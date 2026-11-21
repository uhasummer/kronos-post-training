"""Data representation options (train.repr). Single source of truth for training AND inference preprocessing.

zscore : Kronos default. Window z-score with mean/std from the lookback part, clip. Needs the raw window only.
logret : per-bar relative form, independent of the window: o/h/l/c -> log(x / prev_close); volume, amount ->
         log((v+1) / (median over lookback + 1)). Optional 7th channel = cumulative log close return from window start
         (restores the absolute level the z-score kept). Needs ONE extra bar before the window (prev_close).
"""
import numpy as np

def n_extra_bars(cfg): return 1 if cfg.get("repr", "zscore") == "logret" else 0
def d_in(cfg): return 6 + (1 if cfg.get("repr", "zscore") == "logret" and cfg.get("logret_cum_channel", True) else 0)

def represent(x, cfg):
    """x: float32 [n_extra + W, 6] raw OHLCVA -> float32 [W, d_in] model input."""
    L, clip = cfg["lookback"], cfg.get("clip", 5.0)
    if cfg.get("repr", "zscore") == "zscore":
        mu, sd = x[:L].mean(0), x[:L].std(0)
        return np.clip((x - mu) / (sd + 1e-5), -clip, clip).astype(np.float32)
    prev_c = x[:-1, 3:4]; cur = x[1:]
    px = np.log(np.maximum(cur[:, :4], 1e-12) / np.maximum(prev_c, 1e-12)) * cfg.get("logret_scale", 100.0)   # ~percent units
    med = np.median(cur[:L, 4:6], axis=0, keepdims=True)
    va = np.log((cur[:, 4:6] + 1.0) / (med + 1.0))
    out = [px, va]
    if cfg.get("logret_cum_channel", True): out.append(np.cumsum(px[:, 3:4], axis=0))
    return np.clip(np.concatenate(out, 1), -clip, clip).astype(np.float32)

def close_change(pred, x, cfg):
    """Model output (W_pred, d_in) for the horizon -> predicted close change vs last context close, in a unit comparable
    across representations: zscore -> normalised close units; logret -> cumulative log return (scaled)."""
    L = cfg["lookback"]
    if cfg.get("repr", "zscore") == "zscore":
        mu, sd = x[:L, 3].mean(), x[:L, 3].std()
        return pred[:, 3] - (x[L - 1, 3] - mu) / (sd + 1e-5)
    return np.cumsum(pred[:, 3])
