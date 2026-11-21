"""Statistical tests for publication-grade reporting.
- newey_west_t : t-stat of a mean (e.g. daily IC series) with HAC standard errors (Newey-West, Bartlett kernel)
- diebold_mariano : DM test on two loss series (e.g. QLIKE of model vs GARCH), HAC variance, small-sample (Harvey) correction
- bootstrap_ci : stationary-block bootstrap CI for any statistic of a daily series (Sharpe, mean IC ...)
- model_confidence_set : Hansen-Lunde-Nason MCS (elimination by max-t, bootstrap p-values), returns surviving models
"""
import numpy as np
from scipy import stats

def newey_west_t(x, lags=None):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); m = x.mean(); e = x - m
    L = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    v = (e ** 2).sum() / n
    for l in range(1, L + 1): v += 2 * (1 - l / (L + 1)) * (e[l:] * e[:-l]).sum() / n
    se = np.sqrt(v / n); t = m / se
    return dict(mean=float(m), t=float(t), p=float(2 * (1 - stats.norm.cdf(abs(t)))), se=float(se), n=n, lags=L)

def diebold_mariano(loss_a, loss_b, h=1):
    """H0: equal expected loss. Negative stat = A better (lower loss). Harvey-Leybourne-Newbold correction."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float); d = d[np.isfinite(d)]; n = len(d)
    r = newey_west_t(d, lags=max(h - 1, 0)); dm = r["t"] * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dict(dm=float(dm), p=float(2 * (1 - stats.t.cdf(abs(dm), n - 1))), mean_diff=float(d.mean()), n=n)

def stationary_bootstrap(x, n_boot=2000, block=10, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float); n = len(x); p = 1.0 / block
    for _ in range(n_boot):
        idx = np.empty(n, int); i = rng.integers(n)
        for k in range(n):
            if rng.random() < p: i = rng.integers(n)
            idx[k] = i; i = (i + 1) % n
        yield x[idx]

def bootstrap_ci(x, stat=np.mean, n_boot=2000, block=10, alpha=0.05, seed=0):
    vals = np.array([stat(s) for s in stationary_bootstrap(x, n_boot, block, seed)])
    return dict(point=float(stat(np.asarray(x, float))), lo=float(np.quantile(vals, alpha / 2)), hi=float(np.quantile(vals, 1 - alpha / 2)))

def sharpe(x, periods=252): x = np.asarray(x, float); return float(x.mean() / (x.std() + 1e-12) * np.sqrt(periods))

def model_confidence_set(losses, alpha=0.1, n_boot=1000, block=10, seed=0):
    """losses: dict name -> loss series (aligned). Returns (surviving names, elimination order with p-values)."""
    names = list(losses); L = np.stack([np.asarray(losses[k], float) for k in names], 1); n, m = L.shape
    boots = list(stationary_bootstrap(np.arange(n), n_boot, block, seed)); surv = list(range(m)); order = []
    while len(surv) > 1:
        sub = L[:, surv]; dbar = sub.mean(0) - sub.mean(); 
        def tmax(idx_rows, cols):
            s = L[idx_rows][:, cols]; d = s.mean(0) - s.mean(); v = np.array([np.var(L[idx_rows][:, c] - L[idx_rows][:, cols].mean(1)) for c in cols]) / len(idx_rows) + 1e-12; return d / np.sqrt(v), d
        t_obs, _ = tmax(np.arange(n), surv); T = np.max(t_obs)
        Tb = np.array([np.max(tmax(b.astype(int), surv)[0] - t_obs) for b in boots]); p = float((Tb >= T).mean())
        worst = surv[int(np.argmax(t_obs))]; order.append((names[worst], p))
        if p >= alpha: break
        surv.remove(worst)
    return [names[i] for i in surv], order
