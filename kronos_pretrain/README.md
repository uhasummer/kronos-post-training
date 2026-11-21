# kronos_pretrain — multi-market / multi-frequency Kronos pretraining (DeepSpeed ZeRO-2)

Model code is untouched upstream Kronos (`Kronos/model`). Everything else is ours.

| stage | file | what |
|---|---|---|
| 1 | `ingest.py` | sources → canonical parquet (`data/canonical/{market}/{freq}/{symbol}.parquet`). Handlers: `qlib_bin` (reads Qlib .bin directly), `csv`, `resample` (coarser freqs, bar-end labelled). |
| 2 | `dedupe.py` | row dupes, same-instrument-several-sources, near-duplicate series (chunked hashes of rounded log-returns aligned on wall-clock chunks; ETFs on one index, contract months). |
| 3 | `shard.py` | memmap shards per (market, freq) bucket, series stored contiguously, no materialised windows. |
| 4 | `data.py` + `represent.py` + `timefeat.py` | rank-aware `IterableDataset`: temperature-balanced bucket sampling (p ∝ n^α, optional schedule), time split by window end, `zscore` or `logret` representation, Kronos or session time features, max-gap rejection, exact resume by replaying the rng stream. |
| 5 | `train_tokenizer.py` | BSQ tokenizer, loss identical to Kronos finetune, DeepSpeed ZeRO-2 + bf16 autocast, checkpoints + HF `save_pretrained`. |
| 6 | `train_predictor.py` | autoregressive predictor; frozen tokenizer encodes each window on the fly (tokens depend on per-window normalisation, so they cannot be cached per series). |
| opt | `model_ext.py`, `downstream_val.py` | (market, freq) bucket conditioning for the predictor; zero-shot CSI300 RankIC during training. All-options CPU check: `python smoke_opt.py`. Every key is documented in CONFIG.md. |
| factors | `factors.py` | causal builtin factor library + external factor join, cross-sectional normalisation; fused at `train.factor_mode` = repr / token / cond (see CONFIG.md). Check: `python smoke_factors.py`. |

Run everything (2 GPUs): `GPUS=1,2 ./run_small.sh` — config in `configs/small.yaml`.
Resume: `deepspeed --include localhost:1,2 train_tokenizer.py --config configs/small.yaml --resume`.
Multi-node: add `--hostfile hosts` to the deepspeed launcher; nothing else changes (each rank draws its own sample stream from the shared shard dir).
Local CPU dry run on synthetic data (no DeepSpeed): `python smoke_local.py`.

Scaling knobs: `tokenizer`/`predictor` blocks take the official config.json fields (Kronos-small = d_model 512, 8 layers; Kronos-base = 832, 12 layers);
`rows_per_shard`, `alpha`, `samples_per_epoch`, `total_steps`, `micro_batch × accum × world`.

## From-scratch pretraining with factors (the main experiment)
`python make_arms.py` writes `configs/pretrain_{cond,token,repr}.yaml` from `configs/factor_ablation.yaml` (A-share daily, 18 builtin + 158 Alpha158 factors,
Kronos-Tokenizer-base / Kronos-small sizes, Kronos's own time split). `GPUS=1,2 ./run_arms.sh` builds the shared data once, then trains each arm
(K-line tokenizer → [factor tokenizer] → predictor) and prints the downstream RankIC trajectory with the factors-zeroed ablation. Logs: `runs/pretrain_<arm>/*.log`, TensorBoard: `runs/pretrain_<arm>/tb`.

## Ported from a parallel implementation (2026-09-15)
`cleaning.py` (paper Algorithm 1) and `synthetic.py` (GBM+regimes+GARCH+jumps) ported verbatim; paper Table 5 hyper-parameters,
cosine schedule with 10% warm start, teacher-forced s2 conditioning, causal validation (upstream `is_causal = self.training` hazard),
param-group weight decay, per-series window cap, symbol holdout validation, multi-length contexts without padding, codebook usage/perplexity,
token accuracy, forecast-vs-persistence eval (`eval_forecast.py`), `--overfit_batch` sanity check, grad-norm / throughput logging. Check: `python smoke_merge.py`.
