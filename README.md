# Kronos Pretraining, Post-training and Evaluation

Three packages on top of [Kronos](https://github.com/shiyu-coder/Kronos), a foundation model for financial
candlesticks: **pretraining from scratch, a post-training / fine-tuning framework, and an evaluation
harness** — the parts upstream does not ship. Upstream provides the model definition, two fine-tuning
examples and a web UI; this repository supplies the data pipeline, training code, downstream tasks,
external baselines and statistical testing around it.

## What we validated

Freezing the official Kronos-small backbone — **no weight updates** — and training only a
**513-parameter linear head** with an IC loss, using **no hand-built factors**, still ranks stocks well
out of sample:

| metric | probe_alpha | LightGBM + Alpha158 | Kronos generation path |
|---|---|---|---|
| 2023 validation RankIC | 0.092 (ICIR 0.76) | 0.071 (ICIR 0.66) | — |
| Holdout RankIC (2024-04-01 → 2025-06-05, 284 days, scored once) | **0.1088** | 0.0806 | 0.0324 |
| ICIR | **0.94** | 0.41 | 0.23 |
| Newey–West t | **7.44** (p < 0.001) | 3.29 | 1.70 |
| Days with positive IC | 82% | — | — |
| Decile long–short spread (10-day) | 2.3% | — | — |

Protocol: labels are open[t+1] → open[t+11]; training dates end 2022-12 with purging; a 10-day embargo
precedes the 2023 validation year; the holdout was scored once, after every choice was fixed.

**Takeaway: the alpha is in the pretrained representation, not in the generation.** One forward pass
through a frozen backbone and a linear head already exposes a 0.1-level cross-sectional signal; the
autoregressive "generate K-lines, then read a signal" path retains about a third of it.


## Layout

| package | what it does |
|---|---|
| `kronos_pretrain/` | Full pretraining pipeline: multi-source ingestion, deduplication, cleaning, sharding, a rank-aware sampler, two-stage DeepSpeed ZeRO-2 training, factor fusion, and downstream evaluation during training. |
| `kronos_post/` | Post-training: cross-sectional datasets with purged/embargoed labels, `probe` / `lora` / `full` adaptation, alpha and pricing heads, classical volatility baselines. |
| `kronos_eval/` | Evaluation: the Qlib leaderboard protocol, the Kronos-protocol backtest, Newey-West, Diebold-Mariano, stationary-block bootstrap, Model Confidence Set, multi-seed aggregation. |

Each package has its own README with a file-by-file table. `kronos_pretrain/CONFIG.md` documents every
configuration key; `kronos_pretrain/INSTALL.md` covers setup.

## What this repository adds

- **Data pipeline** — Qlib `.bin`, CSV and yfinance ingestion into one canonical parquet layout; three-level
  dedup (repeated timestamps, multi-source duplicates, near-duplicate series via chunked log-return hashes);
  the paper's Algorithm 1 cleaning; memmapped per-bucket shards.
- **Sampler** — temperature-balanced buckets (`p ∝ n^α`), per-series caps, end-time splits, multi-length
  training without padding, exact resume from the RNG stream.
- **Factors** — 18 causal built-ins plus Alpha158 or your own table; three fusion points via
  `train.factor_mode`: `repr` (extra tokenizer channels), `token` (BSQ-quantised third stream),
  `cond` (MLP bottleneck on the token embedding). Every evaluation re-runs with factors zeroed to report
  their marginal contribution.
- **Downstream model selection** — RankIC under the Kronos protocol during training, official checkpoint
  scored first as a baseline.
- **Evaluation discipline** — date-resolved index membership, one-time test scoring, Newey–West /
  Diebold–Mariano on every headline number, costs and turnover reported, identical code path for baselines.

## Upstream issues fixed

1. The s2 dependency layer sets `is_causal = self.training`, so `model.eval()` lets the s2 head attend to
   future tokens and validation loss peeks ahead. Validation now stays in train mode with all dropout
   temporarily disabled (`common.causal_eval`).
2. Passing `padding_mask` and `is_causal` together makes SDPA raise; the model does not in fact support
   padding. Windows are therefore fixed-length and `padding_mask` is never passed — multi-length training
   uses same-length batches instead.
3. The upstream fine-tuning script conditions s2 on *sampled* s1, while equation (8) of the paper uses the
   ground truth. This is now the `s2_conditioning` option, defaulting to teacher forcing.

## Install

```bash
git clone https://github.com/shiyu-coder/Kronos ~/work/Kronos   # upstream model code
git clone <this-repo> ~/work/kronos-overlay
cp -r ~/work/kronos-overlay/kronos_{pretrain,post,eval} ~/work/
cd ~/work && pip install -r kronos_pretrain/requirements.txt
```

Every entry point calls `import kpaths`, which locates the upstream checkout via `$KRONOS_UPSTREAM`, the
packages' parent directory, `<parent>/Kronos` or `<package>/Kronos`, and puts the sibling packages on
`sys.path`. Override the root with `KRONOS_ROOT`. Run scripts from inside their own package directory,
since data paths are relative. See `kronos_pretrain/INSTALL.md` for details.

## Quick check (CPU, synthetic data, no GPU needed)

```bash
cd kronos_pretrain && python smoke_local.py && python smoke_opt.py && python smoke_factors.py && python smoke_merge.py
cd ../kronos_post && python smoke_post.py
```

These five smoke tests exercise the base pipeline, every optional switch, all three factor modes, the
ported components and post-training, on generated data.

## Running it

```bash
# Small end-to-end run on 2 GPUs
cd kronos_pretrain && GPUS=1,2 ./run_small.sh                 # configs/small.yaml

# The factor-fusion arms
python make_arms.py && GPUS=1,2 ./run_arms.sh                 # from configs/factor_ablation.yaml

# Post-training: frozen backbone + linear head + IC loss
cd ../kronos_post && deepspeed --include localhost:1,2 train_post.py --config configs/probe_alpha.yaml

# Evaluation
cd ../kronos_eval && python qlib_bench.py --model lightgbm    # public baseline
python backtest_kronos.py --signals signals/*.parquet --start 2024-04-01 --end 2025-06-05
```

## Data

No market data and no model checkpoints are redistributed here. A-share daily data comes from the Qlib CN bundle at
`~/.qlib/qlib_data/cn_data`; other markets are fetched by `kronos_pretrain/ingest.py`. Evaluation pickles
referenced by the configs (`data/eval/cn_test.pkl`, `data/eval/cn_val.pkl`) are produced by
`kronos_pretrain/make_pkl.py` from the canonical parquet directory.

## Citation

This work builds on Kronos. If you use it in research, please cite the original paper:

```bibtex
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

## Disclaimer

This repository is research and engineering code, **not investment advice**, and nothing here is a
recommendation to trade. The reported numbers come from a single-seed backtest under the protocol
described above; they do not guarantee future performance and depend on data, cost and execution
assumptions not included here.
No market data and no model checkpoints are distributed in this repository. This project is not
affiliated with or endorsed by the Kronos authors.

## License

MIT — see `LICENSE`. This project builds on Kronos (MIT, Copyright (c) 2025 ShiYu); `NOTICE` records which
files reproduce upstream code and why.
