"""Metrics on per-date predictions. alpha: IC / RankIC / ICIR / top-bottom decile spread / turnover proxy;
pricing (quantile): pinball, coverage per quantile, CRPS approx; vol: MSE of log RV and QLIKE, vs EWMA / HAR / GARCH."""
import kpaths  # noqa: F401
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from heads import QUANTILES

def alpha_metrics(frames, min_n=10):
    """frames: list of dict(date, sym, pred, y) -> dict."""
    ics, rics, spreads = [], [], []; prev = None; turn = []
    for f in frames:
        p, y = np.asarray(f["pred"]), np.asarray(f["y"])
        if len(p) < min_n: continue
        ics.append(pearsonr(p, y)[0]); rics.append(spearmanr(p, y)[0])
        k = max(1, len(p) // 10); o = np.argsort(p); spreads.append(y[o[-k:]].mean() - y[o[:k]].mean())
        top = set(np.asarray(f["sym"])[o[-k:]].tolist())
        if prev is not None: turn.append(1 - len(top & prev) / max(len(top), 1))
        prev = top
    r = np.array(rics)
    return dict(IC=float(np.nanmean(ics)), RankIC=float(np.nanmean(r)), ICIR=float(np.nanmean(r) / (np.nanstd(r) + 1e-12)), pos_rate=float((r > 0).mean()),
                decile_spread=float(np.nanmean(spreads)), turnover=float(np.mean(turn)) if turn else float("nan"), n_dates=len(r))

def quantile_metrics(q_pred, y):
    q = np.array(QUANTILES); d = y[:, None] - q_pred; pin = np.maximum(q * d, (q - 1) * d).mean()
    cov = (y[:, None] <= q_pred).mean(0)
    return dict(pinball=float(pin), coverage_err=float(np.abs(cov - q).mean()), coverage=dict(zip([str(x) for x in q], np.round(cov, 3).tolist())))

def vol_metrics(pred_log, y_log):
    e = pred_log - y_log; rv, pv = np.exp(2 * y_log), np.exp(2 * pred_log)
    return dict(mse_logvol=float(np.mean(e ** 2)), qlike=float(np.mean(rv / pv - np.log(rv / pv) - 1)))

def signal_table(frames, store, bucket):
    """-> DataFrame(datetime, instrument, score) for Kronos's TopkDropout backtest (kronos_eval/backtest_kronos.py)."""
    rows = []
    for f in frames:
        for s, p in zip(f["sym"], f["pred"]): rows.append((pd.Timestamp(f["date"], unit="s"), store.series[bucket][int(s)][3].split("#")[0], float(p)))
    return pd.DataFrame(rows, columns=["datetime", "instrument", "score"])
