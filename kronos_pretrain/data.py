"""Rank-aware sampler over shards. No global window index: every rank draws its own (bucket, series, start)
stream from a seed derived from (seed, epoch, rank), so restarts are exact and no shuffle traffic is needed.

Split is by wall-clock time per window END:  train: end <= train_end ; val: val_start <= end <= val_end.
Configurable (train.*): repr (zscore|logret), time_features (kronos|session), max_gap, alpha / alpha_schedule, cond_bucket.
Each sample = (x [W, d_in] float32, stamps [W, 5] float32, bucket_id int64).
"""
import kpaths  # noqa: F401
import os, json
import numpy as np, torch
from torch.utils.data import IterableDataset
from schema import FREQ_SEC
from represent import represent, n_extra_bars
from timefeat import session_columns
from synthetic import generate_window, SyntheticParams
from schema import stamps_from_ts

def _ts(s): import pandas as pd; return int(pd.Timestamp(s).value // 10**9)

class ShardStore:
    def __init__(self, shard_dir):
        self.dir = shard_dir; self.man = json.load(open(os.path.join(shard_dir, "manifest.json"))); self._mm = {}
        self.bucket_ids = {b: i for i, b in enumerate(sorted(self.man["buckets"]))}     # stable ids for cond_bucket
        self.bucket_ids["SYN"] = len(self.bucket_ids)                                     # synthetic windows (train.synthetic_fraction)
        self.series = {}                                             # bucket -> list of (shard id, offset, length, symbol)
        for b, info in self.man["buckets"].items():
            lst = []
            for s in info["shards"]:
                for r in json.load(open(os.path.join(shard_dir, b, f"{s['id']:04d}.json"))): lst.append((s["id"], r["offset"], r["length"], r["symbol"]))
            self.series[b] = lst
    def arrays(self, b, k):
        key = (b, k)
        if key not in self._mm:
            p = os.path.join(self.dir, b, f"{k:04d}")
            self._mm[key] = tuple(np.load(p + e, mmap_mode="r") for e in (".x.npy", ".ts.npy", ".st.npy")) + ((np.load(p + ".f.npy", mmap_mode="r"),) if self.man.get("factor_names") else ())
        return self._mm[key]
    @property
    def n_factors(self): return len(self.man.get("factor_names") or [])

def is_holdout_symbol(sym, frac, salt="kronos"):
    """Deterministic per-symbol holdout (train.val_symbol_frac): salted md5 in [0,1) < frac. Segments symbol#k share the symbol."""
    import hashlib
    if frac <= 0: return False
    return int(hashlib.md5(f"{salt}:{sym.split('#')[0]}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac

class WindowSampler:
    """Per-bucket temperature balancing p_b ∝ n_b^alpha over the number of valid windows in the split.
    Per-series weight = min(n_windows, cap_windows_per_series) (TimesFM-style cap) so one very long series cannot dominate a bucket.
    Split: by time (window end) and optionally by symbol (val_symbol_frac: those symbols never appear in train; in val they
    contribute every window ending <= val_end, i.e. unseen-symbol validation)."""
    def __init__(self, store, W, split, cfg, alpha=0.5):
        self.store, self.W, self.cfg = store, W, cfg; self.extra = n_extra_bars(cfg)
        t_end, v0, v1 = _ts(cfg["train_end"]), _ts(cfg["val_start"]), _ts(cfg["val_end"])
        frac, salt = cfg.get("val_symbol_frac", 0.0), cfg.get("val_symbol_salt", "kronos"); cap = int(cfg.get("cap_windows_per_series", 0) or 0)
        self.valid = {}                                              # bucket -> list of (series idx, start_lo, start_hi) hi exclusive
        for b, lst in store.series.items():
            v = []
            for i, (k, off, n, sym) in enumerate(lst):
                ts = store.arrays(b, k)[1][off:off + n]
                ends = ts[W - 1:]                                    # window end ts for start = 0..n-W
                hold = is_holdout_symbol(sym, frac, salt)
                if split == "train":
                    if hold: continue
                    lo, hi = self.extra, int(np.searchsorted(ends, t_end, "right"))
                elif hold: lo, hi = self.extra, int(np.searchsorted(ends, v1, "right"))
                else: lo, hi = max(self.extra, int(np.searchsorted(ends, v0, "left"))), int(np.searchsorted(ends, v1, "right"))
                if hi - lo > 0: v.append((i, lo, hi))
            if v: self.valid[b] = v
        self.buckets = sorted(self.valid)
        wts = {b: np.array([min(hi - lo, cap) if cap else hi - lo for _, lo, hi in self.valid[b]], dtype=np.float64) for b in self.buckets}
        self.counts = np.array([wts[b].sum() for b in self.buckets], dtype=np.float64)
        self.n_windows = {b: int(sum(hi - lo for _, lo, hi in self.valid[b])) for b in self.buckets}
        self.cum = {b: np.cumsum(wts[b]) for b in self.buckets}; self.wts = wts
        self.set_alpha(alpha)
    def set_alpha(self, alpha):
        self.alpha = alpha; p = self.counts ** alpha
        mult = self.cfg.get("bucket_weight_mult") or {}                                    # e.g. {USFUT_1D: 2.0}: paper up-weights futures/crypto/forex
        p = p * np.array([float(mult.get(b, 1.0)) for b in self.buckets]); self.p_bucket = p / p.sum()
    def draw(self, rng):
        b = self.buckets[rng.choice(len(self.buckets), p=self.p_bucket)]
        u = rng.rand() * self.cum[b][-1]; j = min(int(np.searchsorted(self.cum[b], u, "right")), len(self.valid[b]) - 1)   # series ∝ capped weight
        i, lo, hi = self.valid[b][j]; start = lo + rng.randint(hi - lo)                                                  # uniform start inside the series
        return b, i, int(start)
    def fetch(self, b, i, start):
        """-> raw x [extra+W, 6], ts [W], stamps [W, 5] (or None if the window violates max_gap)."""
        k, off, n, sym = self.store.series[b][i]; arrs = self.store.arrays(b, k); X, T, S = arrs[:3]
        a, z = off + start - self.extra, off + start + self.W
        x = np.array(X[a:z], dtype=np.float32); ts = np.array(T[a + self.extra:z]); st = np.array(S[a + self.extra:z], dtype=np.int16)
        fac = np.array(arrs[3][a + self.extra:z], dtype=np.float32) if len(arrs) > 3 else np.zeros((self.W, 0), np.float32)
        freq = self.store.man["buckets"][b]["freq"]; mg = (self.cfg.get("max_gap") or {}).get(freq, 0)
        if mg and len(ts) > 1 and np.diff(ts).max() > mg * FREQ_SEC[freq]: return None
        sess = (self.cfg.get("sessions") or {}).get(self.store.man["buckets"][b]["market"])
        if self.cfg.get("time_features", "kronos") == "session" and sess:
            st = st.copy(); st[:, 0], st[:, 1] = session_columns(ts, sess)
        return x, ts, st, fac

class PretrainDataset(IterableDataset):
    """batch_size: when train.context_lengths has several entries, one length is drawn per block of batch_size samples
    (per worker), so every micro-batch is same-length and no padding is ever needed (the model cannot take padding)."""
    def __init__(self, shard_dir, cfg, split, rank, world, seed, n_per_epoch, alpha=0.5, batch_size=1):
        self.store = ShardStore(shard_dir); self.cfg = cfg; W = cfg["lookback"] + cfg["predict"] + 1; self.bs = batch_size
        self.ctx_lens = list(cfg.get("context_lengths") or [cfg["lookback"]]); cw = cfg.get("context_weights") or [1.0] * len(self.ctx_lens)
        self.ctx_p = np.array(cw, dtype=np.float64) / np.sum(cw); assert max(self.ctx_lens) <= cfg["lookback"] and (len(self.ctx_lens) == 1 or cfg["predict"] == 0)
        self.sampler = WindowSampler(self.store, W, split, cfg, alpha)
        self.rank, self.world, self.seed, self.n = rank, world, seed, n_per_epoch; self.epoch = 0; self.skip = 0
    def set_epoch(self, epoch, skip=0): self.epoch, self.skip = epoch, skip
    def set_alpha(self, alpha): self.sampler.set_alpha(alpha)
    def __iter__(self):
        rng = np.random.RandomState((self.seed * 1000003 + self.epoch * 7919 + self.rank) % (2**31 - 1))
        per_rank = self.n // self.world
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        syn_frac = self.cfg.get("synthetic_fraction", 0.0); syn_p = SyntheticParams(**(self.cfg.get("synthetic_params") or {})); zv = self.cfg.get("zero_volume_prob", 0.0)
        m = self.store.n_factors; W = self.sampler.W; extra = self.sampler.extra; L = self.ctx_lens[0]
        for it in range(per_rank):
            if (it // nw) % self.bs == 0: L = int(self.ctx_lens[rng.choice(len(self.ctx_lens), p=self.ctx_p)])   # same length within a worker's batch block
            if syn_frac and rng.rand() < syn_frac:                          # synthetic window (GBM + regimes + GARCH + jumps), bucket SYN, no factors
                xr, tsr = generate_window(W + extra, np.random.default_rng(rng.randint(2**31)), syn_p); b = "SYN"
                got = (xr.astype(np.float32), tsr[extra:], stamps_from_ts(tsr[extra:]), np.full((W, m), np.nan, np.float32))
            else:
                for _ in range(32):                                      # every worker replays the same stream ...
                    b, i, start = self.sampler.draw(rng); got = self.sampler.fetch(b, i, start)
                    if got is not None: break
            if it % nw != wid or it < self.skip or got is None: continue   # ... and owns an interleaved slice; skip = exact resume
            x, ts, st, fac = got
            if zv and rng.rand() < zv: x = x.copy(); x[:, 4:6] = 0.0        # paper regularisation: whole-window volume/amount zeroed
            cfg_s = self.cfg
            if L != self.cfg["lookback"]:                                    # shorter context: keep the LAST L+1 bars (split by window end still holds)
                Wl = L + self.cfg["predict"] + 1; x, st, fac = x[-(Wl + extra):], st[-Wl:], fac[-Wl:]; cfg_s = dict(self.cfg, lookback=L)
            fmask = np.isfinite(fac).astype(np.float32); fac = np.nan_to_num(fac, nan=0.0)
            xr = represent(x, cfg_s)
            if self.cfg.get("factor_mode", "cond") == "repr" and fac.shape[1]: xr = np.concatenate([xr, fac], 1)   # factors as extra input channels
            yield torch.from_numpy(xr), torch.from_numpy(st.astype(np.float32)), torch.tensor(self.store.bucket_ids[b]), torch.from_numpy(fac), torch.from_numpy(fmask)

def alpha_at(cfg, step):
    """train.alpha_schedule: [[step, alpha], ...] piecewise-linear; falls back to train.alpha."""
    sch = cfg.get("alpha_schedule")
    if not sch: return cfg.get("alpha", 0.5)
    xs, ys = [p[0] for p in sch], [p[1] for p in sch]
    return float(np.interp(step, xs, ys))

def summary(shard_dir, cfg):
    st = ShardStore(shard_dir); W = cfg["lookback"] + cfg["predict"] + 1
    for split in ("train", "val"):
        s = WindowSampler(st, W, split, cfg); print(split, {b: f"{n:,}" for b, n in s.n_windows.items()}, "p=", np.round(s.p_bucket, 3))
