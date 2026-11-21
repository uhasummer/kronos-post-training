# kronos_eval — publication-grade evaluation

| file | what |
|---|---|
| `qlib_bench.py` | The Qlib README leaderboard protocol (CSI300/CSI500, Alpha158/Alpha360, train 2008-14 / valid 2015-16 / test 2017-2020.08, IC/ICIR/RankIC/RankICIR, TopkDropout 50/5 backtest with the leaderboard's costs). Runs the public baselines (`--model lightgbm|mlp|gru`; gru wants `--handler Alpha360`) or scores OUR signal table (`--signal *.parquet`) under the identical protocol. `--period modern` = train ..2018 / valid 2019-20 / test 2021-2025.06 as a second, non-overfitted period. Writes per-day RankIC and excess-return series for the tests. |
| `compare.py` | Leaderboard table from `bench_results/`, plus for every model vs the reference (default LightGBM): ΔRankIC with Newey-West t, Diebold-Mariano on daily RankIC, bootstrap CI of IR. |
| `stats.py` | Newey-West t, Diebold-Mariano (HLN-corrected), stationary-block bootstrap CI, Model Confidence Set. Used by both packages. |
| `aggregate.py` | Multi-seed aggregation of `runs/<exp>_seed*/metrics.jsonl` (kronos_post) → mean ± std tables; model selection on validation only (`--select last` or `best:<val key>`). |

Rules baked in: dynamic index membership (qlib resolves constituents by date → no survivorship bias); test period touched once; every headline number
comes with a NW-t or DM p-value and ≥5 seeds; costs and turnover reported; the same script produces baseline and model numbers.
For leaderboard comparability train the alpha head with `cs.horizon: 1` (Qlib's label is next-day); keep `horizon: 10` runs for the Kronos-protocol table.
Pricing: `kronos_post/baselines_vol.py` + `stats.diebold_mariano` / `model_confidence_set` on QLIKE series vs GARCH / HAR-RV / EWMA.
