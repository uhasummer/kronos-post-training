"""Kronos-protocol evaluation of exported signal tables: full-period RankIC / ICIR / Newey-West t and the TopkDropout backtest with
Kronos's own settings (topk 50, n_drop 5, deal at open, delay execution, costs 0.001 / 0.0015, benchmark SH000300).
  python backtest_kronos.py --signals signals/*.parquet --start 2024-04-01 --end 2025-06-05"""
import kpaths  # noqa: F401
import argparse, glob, os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
from stats import newey_west_t, bootstrap_ci, sharpe

def metrics(df):
    ic = df.groupby("datetime").apply(lambda g: spearmanr(g.score, g.y10)[0] if len(g) >= 30 else np.nan).dropna()
    nw = newey_west_t(ic.values); return dict(RankIC=float(ic.mean()), ICIR=float(ic.mean() / ic.std()), NW_t=nw["t"], p=nw["p"], n_dates=int(len(ic))), ic

def backtest(df, start, end, qlib_dir):
    import qlib
    from qlib.config import REG_CN
    from qlib.backtest import backtest as qbt, executor
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.utils.time import Freq
    if not getattr(backtest, "_init", False): qlib.init(provider_uri=os.path.expanduser(qlib_dir), region=REG_CN); backtest._init = True
    s = df.pivot_table(index="datetime", columns="instrument", values="score").sort_index().stack(); s.index.names = ["datetime", "instrument"]; s = s.swaplevel().sort_index()
    strategy = TopkDropoutStrategy(topk=50, n_drop=5, hold_thresh=5, signal=s)
    pm, _ = qbt(strategy=strategy, start_time=start, end_time=end, account=100_000_000, benchmark="SH000300",
                exchange_kwargs={"freq": "day", "limit_threshold": 0.095, "deal_price": "open", "open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5},
                executor=executor.SimulatorExecutor(time_per_step="day", generate_portfolio_metrics=True, delay_execution=True))
    fq = "{0}{1}".format(*Freq.parse("day")); report, _ = pm.get(fq); ex = report["return"] - report["bench"] - report["cost"]; ra = risk_analysis(ex, freq=fq)["risk"]
    ci = bootstrap_ci(ex.values, stat=sharpe, n_boot=500)
    return dict(ann_excess=float(ra["annualized_return"]), IR=float(ra["information_ratio"]), max_dd=float(ra["max_drawdown"]), IR_ci=[round(ci["lo"], 2), round(ci["hi"], 2)], bench_cum=float(report["bench"].sum()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--signals", nargs="+", required=True); ap.add_argument("--start", default="2024-04-01"); ap.add_argument("--end", default="2025-06-05")
    ap.add_argument("--qlib_dir", default="~/.qlib/qlib_data/cn_data"); ap.add_argument("--no_bt", action="store_true"); ap.add_argument("--out", default="kronos_protocol_results.json"); a = ap.parse_args()
    rows = []
    for fp in sorted(sum([glob.glob(p) for p in a.signals], [])):
        df = pd.read_parquet(fp); df = df[(df.datetime >= a.start) & (df.datetime <= a.end)]; m, ic = metrics(df); row = dict(model=os.path.splitext(os.path.basename(fp))[0], **m)
        if not a.no_bt:
            try: row.update(backtest(df, a.start, a.end, a.qlib_dir))
            except Exception as e: row["bt_error"] = str(e)[:120]
        rows.append(row); print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True); ic.to_csv(fp.replace(".parquet", "_rankic_daily.csv"))
    t = pd.DataFrame(rows); print(); print(t.to_string(index=False)); t.to_json(a.out, orient="records", indent=1)

if __name__ == "__main__":
    main()
