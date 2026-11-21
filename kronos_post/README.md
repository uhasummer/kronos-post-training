# kronos_post — fine-tuning / post-training of Kronos for alpha finding and pricing

Sits on top of `kronos_pretrain` (symlinked as `pretrain/`): same shards, factors, DeepSpeed helpers, causal validation.

| file | what |
|---|---|
| `data_cs.py` | cross-sectional batches: one date = one micro-batch (all symbols of a bucket, capped by `max_symbols`); labels `y_ret` = open[t+1]→open[t+1+H] log return (Kronos backtest convention), `y_vol` = log realised vol; purged train dates and embargoed val dates |
| `backbone.py` | loads official or own (KronosCond) checkpoints; `hidden_states()` reproduces Kronos.forward up to the final norm; `pool()` last / mean_last / mean |
| `lora.py` | LoRA on q/k/v/out_proj and FFN w1/w2/w3; `adapter: probe | lora | full`; exact merge |
| `heads.py`, `losses.py` | alpha (scalar; ic / rank_ic / listnet / pairwise), quantile (9 monotone quantiles; pinball), vol (log RV; mse) |
| `eval_post.py` | IC / RankIC / ICIR / decile spread / turnover; pinball + coverage; log-vol MSE + QLIKE; signal table export for Kronos's TopkDropout backtest |
| `baselines_vol.py` | EWMA, HAR-RV, GARCH(1,1) (needs `pip install arch`) |
| `train_post.py` | DeepSpeed trainer; `deepspeed --include localhost:1,2 train_post.py --config configs/probe_alpha.yaml` |

Experiment order: `probe_alpha` (frozen official Kronos-small + linear head + IC loss: is there cross-sectional signal in the representation at all?) →
`lora_alpha` → `pricing_vol` / `pricing_quantile` → swap `backbone` for the pretraining arms' checkpoints (`cond: true`, `factor_mode`, `n_factors`).
Backtest a run: `val_signals_step*.parquet` → `kronos_eval/backtest_kronos.py` (qlib TopkDropout, same settings as Kronos's own evaluation).
CPU check: `python smoke_post.py` (after `kronos_pretrain/smoke_local.py`).
