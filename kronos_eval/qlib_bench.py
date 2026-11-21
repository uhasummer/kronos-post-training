"""Qlib public-benchmark runner: the same protocol as the Qlib README leaderboard, so our numbers slot into that table.

  python qlib_bench.py --model lightgbm --market csi300 --handler Alpha158            # baseline
  python qlib_bench.py --signal runs/probe_alpha/val_signals_step3000.parquet --market csi300   # our signal, same backtest/metrics

Protocol (Qlib benchmark): train 2008-01-01..2014-12-31, valid 2015-01-01..2016-12-31, test 2017-01-01..2020-08-01;
label = Ref($close,-2)/Ref($close,-1)-1 (next-day open-to-open proxy used by Qlib);  metrics IC / ICIR / RankIC / RankICIR;
backtest TopkDropout(topk 50, n_drop 5), deal at open, costs 0.0005 open / 0.0015 close, min 5, benchmark SH000300.
--period modern switches to train ..2018 / valid 2019-2020 / test 2021-01-01..2025-06-05 for a second, non-overfitted period.
Requires pyqlib + the Qlib CN bundle at ~/.qlib/qlib_data/cn_data.
"""
import kpaths  # noqa: F401
import argparse, json, os, numpy as np, pandas as pd

PERIODS = {"benchmark": dict(train=("2008-01-01", "2014-12-31"), valid=("2015-01-01", "2016-12-31"), test=("2017-01-01", "2020-08-01")),
           "modern": dict(train=("2010-01-01", "2018-12-31"), valid=("2019-01-01", "2020-12-31"), test=("2021-01-01", "2025-06-05"))}
BENCH = {"csi300": "SH000300", "csi500": "SH000905", "csi1000": "SH000852"}

def init(qdir):
    import qlib; from qlib.config import REG_CN
    # mlflow >= 3 refuses the filesystem tracking backend qlib's recorder defaults to -> point it at a local sqlite db
    expm = {"class": "MLflowExpManager", "module_path": "qlib.workflow.expm", "kwargs": {"uri": "sqlite:///" + os.path.abspath("mlflow.db"), "default_exp_name": "Experiment"}}
    qlib.init(provider_uri=os.path.expanduser(qdir), region=REG_CN, exp_manager=expm)

def label_expr(horizon):
    """Qlib default (next-day close-to-close) or the Kronos-protocol label: open[t+1] -> open[t+1+h]."""
    return None if not horizon else [f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"]

def dataset(market, handler, per, horizon=0):
    from qlib.utils import init_instance_by_config
    kw = dict(start_time=per["train"][0], end_time=per["test"][1], fit_start_time=per["train"][0], fit_end_time=per["train"][1], instruments=market)
    if label_expr(horizon): kw["label"] = label_expr(horizon)
    hconf = {"class": handler, "module_path": "qlib.contrib.data.handler", "kwargs": kw}
    return init_instance_by_config({"class": "DatasetH", "module_path": "qlib.data.dataset", "kwargs": dict(handler=hconf, segments=per)})

MODELS = {"lightgbm": {"class": "LGBModel", "module_path": "qlib.contrib.model.gbdt", "kwargs": dict(loss="mse", colsample_bytree=0.8879, learning_rate=0.0421, subsample=0.8789, lambda_l1=205.6999, lambda_l2=580.9768, max_depth=8, num_leaves=210, num_threads=20)},
          "mlp": {"class": "DNNModelPytorch", "module_path": "qlib.contrib.model.pytorch_nn", "kwargs": dict(loss="mse", lr=0.002, optimizer="adam", max_steps=8000, batch_size=8192, GPU=0, pt_model_kwargs=dict(input_dim=158, layers=(256,)))},
          "gru": {"class": "GRU", "module_path": "qlib.contrib.model.pytorch_gru", "kwargs": dict(d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, n_epochs=200, lr=1e-3, early_stop=10, batch_size=800, metric="loss", loss="mse", GPU=0)}}

def signal_metrics(pred, label):
    """pred, label: Series indexed by (datetime, instrument). Daily IC / RankIC and their IR, plus Newey-West t of RankIC."""
    from stats import newey_west_t
    df = pd.concat([pred.rename("p"), label.rename("y")], axis=1).dropna()
    ic = df.groupby(level="datetime").apply(lambda g: g.p.corr(g.y)); ric = df.groupby(level="datetime").apply(lambda g: g.p.corr(g.y, method="spearman"))
    return dict(IC=float(ic.mean()), ICIR=float(ic.mean() / ic.std()), RankIC=float(ric.mean()), RankICIR=float(ric.mean() / ric.std()), RankIC_NW_t=newey_west_t(ric.values)["t"], n_dates=int(len(ric))), ric

def backtest(pred, market, per):
    from qlib.contrib.evaluate import risk_analysis, backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy
    from stats import bootstrap_ci, sharpe
    strategy = TopkDropoutStrategy(topk=50, n_drop=5, signal=pred)
    report, _ = backtest_daily(start_time=per["test"][0], end_time=per["test"][1], strategy=strategy, account=100_000_000, benchmark=BENCH[market],
                               exchange_kwargs=dict(freq="day", limit_threshold=0.095, deal_price="close", open_cost=0.0005, close_cost=0.0015, min_cost=5))
    ex = report["return"] - report["bench"] - report["cost"]; ra = risk_analysis(ex, freq="day")["risk"]
    ci = bootstrap_ci(ex.values, stat=sharpe)
    return dict(ann_excess_w_cost=float(ra["annualized_return"]), IR=float(ra["information_ratio"]), max_dd=float(ra["max_drawdown"]), IR_boot_ci=[round(ci["lo"], 3), round(ci["hi"], 3)]), ex

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qlib_dir", default="~/.qlib/qlib_data/cn_data"); ap.add_argument("--market", default="csi300"); ap.add_argument("--handler", default="Alpha158")
    ap.add_argument("--period", default="benchmark"); ap.add_argument("--model", default=None); ap.add_argument("--signal", default=None, help="parquet with datetime, instrument, score (our model)")
    ap.add_argument("--out", default="bench_results"); ap.add_argument("--name", default=None)
    ap.add_argument("--horizon", type=int, default=0, help="0 = Qlib next-day label; 10 = Kronos protocol open[t+1]->open[t+11]")
    ap.add_argument("--report_start"); ap.add_argument("--report_end", help="also report metrics on this sub-period of the test set (e.g. the Kronos downstream test window)")
    a = ap.parse_args()
    init(a.qlib_dir); per = PERIODS[a.period]; os.makedirs(a.out, exist_ok=True); name = a.name or (a.model or os.path.basename(a.signal))
    if a.horizon: name += f"_h{a.horizon}"
    ds = dataset(a.market, a.handler, per, a.horizon); label = ds.prepare("test", col_set="label").iloc[:, 0]
    if a.model:
        from qlib.utils import init_instance_by_config
        model = init_instance_by_config(MODELS[a.model]); model.fit(ds); pred = model.predict(ds)
    else:
        s = pd.read_parquet(a.signal); s["datetime"] = pd.to_datetime(s["datetime"]); pred = s.set_index(["datetime", "instrument"])["score"]
        pred = pred[(pred.index.get_level_values(0) >= per["test"][0]) & (pred.index.get_level_values(0) <= per["test"][1])]
    pred = pred.rename("score"); pred.index.names = ["datetime", "instrument"]
    m, ric = signal_metrics(pred, label); res = dict(name=name, market=a.market, period=a.period, horizon=a.horizon, **m)
    if a.report_start and a.report_end:
        sub = pred[(pred.index.get_level_values(0) >= a.report_start) & (pred.index.get_level_values(0) <= a.report_end)]
        ms, _ = signal_metrics(sub, label); res.update({f"sub_{k}": v for k, v in ms.items()}); res["sub_period"] = [a.report_start, a.report_end]
    try:
        b, ex = backtest(pred, a.market, per); res.update(b)
    except Exception as e:
        print("backtest skipped:", e); ex = pd.Series(dtype=float)
    print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(a.out, f"{name}_{a.market}_{a.period}.json"), "w"))
    ric.to_csv(os.path.join(a.out, f"{name}_{a.market}_{a.period}_rankic_daily.csv"))
    if len(ex): ex.to_csv(os.path.join(a.out, f"{name}_{a.market}_{a.period}_excess_daily.csv"))

if __name__ == "__main__":
    main()
