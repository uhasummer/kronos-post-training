"""Autoregressive forecast on fixed validation windows vs the persistence baseline (ported idea: other session's evaluate.py).
Works for both representations: predicted close change = represent.close_change (normalised-close units for zscore,
cumulative scaled log-return for logret); persistence predicts zero change. Reports MAE ratio (>= 1 = no better than
doing nothing) and direction accuracy on the last horizon bar. Buckets are mixed in proportion to the val sampler."""
import kpaths  # noqa: F401
import numpy as np, torch
from represent import represent, close_change, n_extra_bars
from model.kronos import auto_regressive_inference

def fixed_windows(dataset, n, seed=1234):
    """Draw n raw (bucket, x_raw, stamps, factors) windows from the dataset's sampler with a fixed rng."""
    rng = np.random.RandomState(seed); s = dataset.sampler; out = []
    while len(out) < n:
        b, i, start = s.draw(rng); got = s.fetch(b, i, start)
        if got is not None: out.append((b,) + got)
    return out

@torch.no_grad()
def forecast_eval(windows, tok, model, cfg, pred_len, dev, bs=256, fmode=None, ftok=None, bucket_ids=None, use_bucket=False):
    L = cfg["lookback"] - pred_len; ex = n_extra_bars(cfg); c = dict(cfg, lookback=L, predict=pred_len)
    assert L >= 8, "pred_len leaves too little context"
    mae_m = mae_p = 0.0; hits = 0; n = 0
    for k in range(0, len(windows), bs):
        chunk = windows[k:k + bs]
        raw = np.stack([w[1][:ex + L + pred_len] for w in chunk]); st = np.stack([w[3][:L + pred_len] for w in chunk]).astype(np.float32)
        xn = np.stack([represent(r[:ex + L], dict(c, predict=0)) for r in raw])
        fac = np.stack([w[4][:L + pred_len] for w in chunk]); fm = np.isfinite(fac).astype(np.float32); fac = np.nan_to_num(fac, nan=0.0); fac[:, L:] = 0; fm[:, L:] = 0
        if fmode == "repr": xn = np.concatenate([xn, fac[:, :L]], -1)
        elif fmode in ("cond", "token") and hasattr(model, "set_factors"):
            ft, fmt = torch.from_numpy(fac).to(dev), torch.from_numpy(fm).to(dev); model.set_factors(ftok.encode(ft, fmt) if fmode == "token" else ft, fmt)
        if use_bucket and hasattr(model, "set_bucket"): model.set_bucket(bucket_ids[chunk[0][0]])
        with torch.autocast("cuda", enabled=False):                                                          # upstream ends with .numpy(): bf16 not allowed
            p = auto_regressive_inference(tok, model, torch.from_numpy(xn).to(dev), torch.from_numpy(st[:, :L]).to(dev), torch.from_numpy(st[:, L:]).to(dev),
                                          max_context=L, pred_len=pred_len, clip=cfg.get("clip", 5.0), T=1.0, top_k=0, top_p=0.9, sample_count=1)
        p = np.asarray(p, dtype=np.float32)[:, -pred_len:, :]
        for r, pp in zip(raw, p):
            pred = close_change(pp, r[ex:], c)                                   # (pred_len,) predicted change vs last context close
            truth = _true_change(r[ex:], c, L, pred_len)
            mae_m += np.abs(pred - truth).mean(); mae_p += np.abs(truth).mean(); hits += int(np.sign(pred[-1]) == np.sign(truth[-1])); n += 1
    if hasattr(model, "set_factors"): model.set_factors(None)
    if hasattr(model, "set_bucket"): model.set_bucket(None)
    return dict(mae_ratio_vs_persistence=(mae_m / n) / max(mae_p / n, 1e-12), direction_acc=hits / n, n=n, pred_len=pred_len)

def _true_change(x, c, L, H):
    """True close change in the same units close_change() produces for the model output."""
    if c.get("repr", "zscore") == "zscore":
        mu, sd = x[:L, 3].mean(), x[:L, 3].std(); return (x[L:L + H, 3] - x[L - 1, 3]) / (sd + 1e-5)
    return np.cumsum(np.log(x[L:L + H, 3] / np.r_[x[L - 1, 3], x[L:L + H - 1, 3]])) * c.get("logret_scale", 100.0)
