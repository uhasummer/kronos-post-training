"""Cross-sectional (per-date) batches with purged / embargoed labels, on top of the pretraining shards.

A batch = every symbol of one bucket on one date (subsampled to max_symbols): context of `lookback` bars ending at
bar t, plus labels computed from bars AFTER t:
  y_ret : open[t+1] -> open[t+1+horizon] log return      (matches Kronos's backtest: signal at t, deal at next open, delay_execution)
  y_vol : log realised vol of close-to-close returns over (t+1 .. t+1+horizon)
Split: train dates have t + 1 + horizon <= train_end (purge: the whole label window lies before the cut);
       val dates start >= train_end + embargo_bars and their label windows end <= val_end.
"""
import sys, os, numpy as np, torch
from collections import defaultdict
from torch.utils.data import IterableDataset
import kpaths  # noqa: F401
from data import ShardStore, is_holdout_symbol, _ts
from represent import represent, n_extra_bars
from schema import FREQ_SEC
from timefeat import session_columns

class DateIndex:
    """bucket -> sorted dates -> list of (series idx, position t) with a full context and a full label window."""
    def __init__(self, store, bucket, cfg):
        self.store, self.b, self.cfg = store, bucket, cfg
        L, H, ex = cfg["lookback"], cfg["horizon"], n_extra_bars(cfg)
        by_date = defaultdict(list)
        for i, (k, off, n, sym) in enumerate(store.series[bucket]):
            if cfg.get("exclude_holdout", True) and is_holdout_symbol(sym, cfg.get("val_symbol_frac", 0.0), cfg.get("val_symbol_salt", "kronos")): continue
            ts = store.arrays(bucket, k)[1][off:off + n]
            for t in range(L - 1 + ex, n - 1 - H):                     # need bars t+1 .. t+1+H
                by_date[int(ts[t])].append((i, t))
        self.dates = sorted(d for d, lst in by_date.items() if len(lst) >= cfg.get("min_symbols", 30)); self.by_date = by_date
        self.freq = store.man["buckets"][bucket]["freq"]
    def split(self, which):
        H = self.cfg["horizon"]; sec = FREQ_SEC[self.freq]; t_end, v0, v1 = _ts(self.cfg["train_end"]), _ts(self.cfg["val_start"]), _ts(self.cfg["val_end"])
        emb = self.cfg.get("embargo_bars", H) * sec
        if which == "train": return [d for d in self.dates if d + (H + 1) * sec * 1.5 <= t_end]      # 1.5x: calendar gaps (weekends) inflate wall-clock span
        if which == "test":                                                                            # cs.test_start / test_end: scored once, after training
            s0, s1 = _ts(self.cfg["test_start"]), _ts(self.cfg["test_end"]); return [d for d in self.dates if d >= s0 and d + (H + 1) * sec * 1.5 <= s1 + (H + 1) * sec * 1.5]
        return [d for d in self.dates if d >= max(v0, t_end + emb) and d + (H + 1) * sec * 1.5 <= v1]

class CrossSectionDataset(IterableDataset):
    """Yields one date per item: dict(x [N,L(+ex),d_in], st [N,L,5], fac [N,L,m], fmask, y_ret [N], y_vol [N], sym [N] (series idx), date)."""
    def __init__(self, shard_dir, cfg, split, rank=0, world=1, seed=0, shuffle=True):
        self.store = ShardStore(shard_dir); self.cfg = cfg; self.rank, self.world, self.seed, self.shuffle = rank, world, seed, shuffle
        self.idx = {b: DateIndex(self.store, b, cfg) for b in cfg.get("buckets", ["CN_1D"])}
        self.items = [(b, d) for b, ix in self.idx.items() for d in ix.split(split)]; self.epoch = 0
        self.n_factors = self.store.n_factors
    def set_epoch(self, e): self.epoch = e
    def __len__(self): return len(self.items) // self.world
    def fetch_date(self, b, d, rng=None):
        ix = self.idx[b]; lst = ix.by_date[d]; L, H, ex = self.cfg["lookback"], self.cfg["horizon"], n_extra_bars(self.cfg)
        if rng is not None and len(lst) > self.cfg.get("max_symbols", 400): lst = [lst[j] for j in sorted(rng.choice(len(lst), self.cfg["max_symbols"], replace=False))]
        xs, sts, facs, yr, yv, syms = [], [], [], [], [], []
        sess = (self.cfg.get("sessions") or {}).get(self.store.man["buckets"][b]["market"])
        for i, t in lst:
            k, off, n, sym = self.store.series[b][i]; arrs = self.store.arrays(b, k); X, T, S = arrs[:3]
            a0 = off + t - (L - 1) - ex; a1 = off + t + 1
            x = np.array(X[a0:a1], dtype=np.float32); st = np.array(S[a0 + ex:a1], dtype=np.int16)
            if self.cfg.get("time_features", "kronos") == "session" and sess: st = st.copy(); st[:, 0], st[:, 1] = session_columns(np.array(T[a0 + ex:a1]), sess)
            fac = np.array(arrs[3][a0 + ex:a1], dtype=np.float32) if len(arrs) > 3 else np.zeros((L, 0), np.float32)
            fut = np.array(X[off + t + 1: off + t + 2 + H], dtype=np.float64)           # bars t+1 .. t+1+H
            if fut.shape[0] < H + 1 or not np.all(fut[:, :4] > 0): continue
            yr.append(np.log(fut[-1, 0] / fut[0, 0]))                                  # open-to-open
            lr = np.diff(np.log(fut[:, 3])); yv.append(np.log(np.sqrt(np.mean(lr ** 2) + 1e-12) * np.sqrt(H) + 1e-8))
            xs.append(represent(x, dict(self.cfg, lookback=L, predict=0))); sts.append(st.astype(np.float32)); facs.append(fac); syms.append(i)
        if not xs: return None
        fac = np.stack(facs); fm = np.isfinite(fac).astype(np.float32); fac = np.nan_to_num(fac, nan=0.0)
        return dict(x=torch.from_numpy(np.stack(xs)), st=torch.from_numpy(np.stack(sts)), fac=torch.from_numpy(fac), fmask=torch.from_numpy(fm),
                    y_ret=torch.tensor(yr, dtype=torch.float32), y_vol=torch.tensor(yv, dtype=torch.float32), sym=torch.tensor(syms), date=d, bucket=b)
    def __iter__(self):
        rng = np.random.RandomState(self.seed * 7919 + self.epoch); order = np.arange(len(self.items))
        if self.shuffle: rng.shuffle(order)
        wi = torch.utils.data.get_worker_info(); wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        for j, o in enumerate(order[self.rank::self.world]):
            if j % nw != wid: continue
            b, d = self.items[o]; got = self.fetch_date(b, d, rng if self.shuffle else None)
            if got is not None: yield got
