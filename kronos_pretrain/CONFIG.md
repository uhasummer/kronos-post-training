# Configuration reference (every key in `configs/*.yaml`)

One yaml drives all six stages. Top-level keys: `seed`, `out_dir`, `data`, `sources`, `train`, `tokenizer`,
`tokenizer_train`, `predictor`, `predictor_train`, plus the optional `tokenizer_path`.

## Top level
| key | default | what |
|---|---|---|
| seed | required | Global seed. Sampling-stream seeds are `seed` (tokenizer stage), `seed+100` (predictor stage), and `+1` / `+101` for the matching validation streams |
| out_dir | required | Output root. Creates `tokenizer/` and `predictor/`, each holding DeepSpeed checkpoints (`stepN/`), `latest.json`, and `hf_latest/` (HF format, loadable with `from_pretrained`) |
| tokenizer_path | `{out_dir}/tokenizer/hf_latest` | Frozen tokenizer used by the predictor stage. Set it to `NeoQuasar/Kronos-Tokenizer-base` to skip stage 1 and use the official tokenizer |

## data
| key | default | what |
|---|---|---|
| factor_dir | data/factors | Cache directory for factor tables (Alpha158 and friends) |
| canonical_dir | required | Canonical parquet root, laid out as `{market}/{freq}/{symbol}.parquet` with `catalog.csv` and `catalog_dedup.csv` |
| shard_dir | required | Shard root, laid out as `{market}_{freq}/NNNN.{x,ts,st}.npy` plus `manifest.json` |
| cleaning | {enabled: true} | The three-step cleaning of Algorithm 1 in the Kronos paper (structural-break segmentation, illiquid-period removal, stale-segment filtering). Thresholds default per frequency class; override with `overrides: {5min: {jump_threshold: 0.15, ...}}`. A series split into segments enters sharding as independent series (named `symbol#k`) |
| rows_per_shard | 20000000 | Maximum rows per shard. 20M rows ≈ 480 MB (x) + 160 MB (ts) + 200 MB (st). Shards are memmapped, so the size only affects the file count |
| near_dup_thresh | 0.5 | Near-duplicate rule: if the number of chunk hashes two series share, divided by the chunk count of the shorter one, exceeds this, the shorter series is dropped. It cannot be disabled with 0 — use 1.01 |

## sources (a list; order is the priority for row-level deduplication, earlier sources win)
Every entry needs a `type`. Keys per type:

**qlib_bin** (reads the Qlib binaries directly, without importing qlib)
| key | default | what |
|---|---|---|
| path | required | Qlib data directory containing `calendars/day.txt`, `instruments/*.txt`, `features/{sym}/*.day.bin` |
| instruments | all | A list name under `instruments/` (all, csi300, csi500, csi800, csi1000, csiall). Only the symbol set is used; membership intervals are ignored |
| market | CN | Market label |
| fixed behaviour | | Frequency is always 1D; if more than half of `amount` is zero it is replaced by average price × volume; prices are Qlib's adjusted prices |

**csv**
| key | default | what |
|---|---|---|
| path | required | Glob; may match several files |
| market, freq | required | `freq` is one of `1min/5min/15min/30min/60min/1D` |
| columns | required | Mapping for ts, open, high, low, close, volume, amount to column names. Set amount to null to use average price × volume |
| ts_format | None (inferred) | A pandas date format string, e.g. `"%Y/%m/%d %H:%M"` |
| encoding | utf-8-sig | |
| symbol | file name | Symbol to use when one file holds one instrument |
| symbol_col | none | Column holding the symbol when one file holds several instruments |

**resample** (derive coarser frequencies from finer ones already ingested)
| key | what |
|---|---|
| market, from | Source bucket |
| to | A list such as `[15min, 60min, 1D]`. Intraday bars are resampled right-closed and right-labelled (bar stamped with its end time); 1D groups by date. Buckets with zero volume are dropped |

**yfinance** (US equities and continuous futures contracts; free)
| key | default | what |
|---|---|---|
| market | US | Use `US` for equities and `USFUT` for futures so the two become separate sampling buckets |
| freq | 1D | Maps to `interval`, default 1d. Intraday data only goes back 60 days, which is not enough for pretraining |
| tickers / ticker_file / universe | pick one | An explicit list; a text file with one per line; or `universe: sp500`, which scrapes current members from Wikipedia (survivorship-biased) |
| start, end | 2000-01-01, today | `end` is exclusive, so today's unfinished bar is excluded by default |
| adjust | true | Adjusted prices (true for equities, false for continuous futures contracts) |
| batch | 100 | Symbols per request |
| fixed behaviour | | Yahoo has no turnover, so amount = average price × volume |

To add a source, write a function in `ingest.py` and register it in `HANDLERS`. It must yield
`(market, freq, symbol, DataFrame[ts, open, high, low, close, volume, amount])`; no other stage changes.

## train (windows and splits; shared by four stages)
| key | default | what |
|---|---|---|
| lookback | required | Lookback length. Normalisation mean and variance are computed on this segment only |
| predict | required | Prediction length. Total window = lookback + predict + 1 (the extra 1 is the autoregressive next-token shift). For pretraining, use predict 0 and lookback 512, which matches Kronos's `max_context` |
| clip | 5.0 | Clip to ±clip after normalisation |
| alpha | 0.5 | Bucket sampling temperature, `p_b ∝ n_b^alpha`. 1 = proportional to sample count, 0 = uniform over buckets |
| train_end | required | Training windows must end on or before this date (local wall clock) |
| val_start, val_end | required | Validation windows must end inside this range. Leaving at least one window length between this and `train_end` keeps a validation window's lookback out of the training period |

### Optimisation switches under `train` (all off by default = equivalent to stock Kronos)
| key | default | what |
|---|---|---|
| repr | zscore | Data representation. `zscore` is Kronos's per-window normalisation. `logret` is a per-bar relative representation: OHLC become log(x / previous close) × logret_scale, volume and turnover become log((v+1) / (lookback median + 1)). Under logret a given bar has the same representation in every window (tokens are stable across windows), at the cost of reading one bar before the window |
| logret_cum_channel | true | Under logret, append a 7th channel: cumulative log return from the window start, restoring absolute level information. **When on, `tokenizer.d_in` must be 7** (the trainer asserts this) |
| logret_scale | 100.0 | Scale factor for log returns, bringing them to roughly percentage magnitude |
| time_features | kronos | `kronos` = minute/hour/weekday/day/month. `session` replaces the minute column with "fraction of the trading session elapsed × 59" and the hour column with a session index, leaving the rest. This unifies semantics across markets, including 24-hour ones |
| sessions | none | Required by `session` mode: trading sessions per market, e.g. `CN: [["09:30","11:30"],["13:00","15:00"]]`. Markets with no entry keep the kronos columns. Bars are assigned to a session by their end time |
| max_gap | none | Largest permitted gap per frequency, counted in bars of that frequency on wall-clock time, e.g. `{5min: 3, 1D: 15}`. A window with any adjacent pair exceeding it is discarded and resampled (up to 32 attempts). 15 days at 1D accommodates the Chinese New Year holiday |
| cap_windows_per_series | 0 (off) | Caps each series' (or cleaned segment's) sampling weight inside its bucket at `min(window count, cap)`, TimesFM-style, so one very long series cannot dominate a bucket (the paper reproduction used 200000) |
| val_symbol_frac, val_symbol_salt | 0.0, kronos | Symbol holdout: symbols whose salted md5 falls in [0, frac) are kept out of training entirely and contribute every window ending on or before `val_end` to validation, giving an unseen-symbol validation set. All cleaned segments of a symbol are held out together |
| context_lengths, context_weights | [lookback], uniform | Multi-context-length training: each worker draws a length once per `micro_batch` samples, so a micro-batch is always uniform in length and padding is never used (the model does not support it). Shorter windows keep the last L+1 bars. Requires predict = 0, and the maximum must not exceed lookback |
| zero_volume_prob | 0.0 | Regulariser from the paper: with this probability, zero out volume and turnover for the whole window (0.05 suggested) |
| synthetic_fraction | 0.0 | Mix in this fraction of synthetic windows (GBM log price + two-state volatility regimes + GARCH + Poisson jumps + volume seasonality). Bucket name SYN, no factors. `synthetic_params` overrides the generator's parameters |
| bucket_weight_mult | {} | Extra multipliers on bucket sampling weights, e.g. `{USFUT_1D: 2.0}`, matching the paper's upweighting of futures, crypto and FX |
| alpha_schedule | none | `[[step, alpha], ...]`, piecewise linear in optimisation steps, overriding `alpha`. Example: `[[0, 0.3], [4000, 1.0]]` starts near-uniform across markets and ends proportional to data volume. Updated at the start of each virtual epoch |
| cond_bucket | false | Give the predictor a learnable (market, frequency) bucket embedding, zero-initialised so the starting point matches stock Kronos. The model class becomes `model_ext.KronosCond`; at inference call `set_bucket(id)`. Bucket ids are in `ShardStore.bucket_ids`, sorted by bucket name |

## factors (ingestion and fusion; omit the whole block to train without factors)
| key | default | what |
|---|---|---|
| builtin | [] | `all` or a list of names. 18 built-in causal factors, all computed from OHLCVA: ret_{1,5,10,20,60}, vol_{5,20,60}, ma_ratio_{5,20,60}, hl_range_5, gap, vol_ratio_5_20, amount_z_20, max_dd_20, corr_pv_10, rsi_14. Bar t uses data up to t only |
| external | [] | Your own factor tables. Each entry is `{path (glob, parquet or csv), market, freq, symbol_col or symbol, ts_col, ts_format, columns: [...]}`, aligned to bars on (market, normalised symbol, timestamp); unmatched rows become NaN |
| qlib_alpha158 | none | Compute the 158 raw Alpha158 factors with qlib's own handler over the Qlib bundle, with qlib's normalisation and filling disabled in favour of `norm` below: `{qlib_dir, instruments (all/csi300/...), start, end, market (CN), chunk (400)}`. Cached at `data.factor_dir/alpha158_{instruments}.parquet` and read back on later runs; test-set evaluation reads the same table, so ablating Alpha158 factors is meaningful. All A-shares is roughly 5000 × 3700 × 158, about 11 GB of memory, and takes tens of minutes the first time |
| norm | cs_rank | Cross-sectional normalisation, applied across symbols per timestamp within a (market, frequency) bucket: `cs_rank` → [−1, 1]; `cs_zscore`; `none` |
| clip | 3.0 | Clip after normalisation |
| token_bits, hidden | 8, 64 | Codebook bits and hidden width of the factor tokenizer, used when `factor_mode: token` |

Where factors are fused is set by `train.factor_mode`. All three are configurable so they can be compared:

| factor_mode | fusion point | mechanism | stages needed |
|---|---|---|---|
| repr | representation | Factors become extra input channels of the candlestick tokenizer (`tokenizer.d_in` = base channels + m). The reconstruction loss covers only the base channels by default (`tokenizer_train.recon_factors: true` reconstructs everything). Tokens then carry factor information, so factors must be supplied at inference | 1 (candlestick tokenizer) → 2 |
| token | tokenisation | Factors go through their own small BSQ autoencoder into a third token stream, which the predictor adds to its input through a zero-initialised embedding (conditioning only; it is not predicted) | 1 → 1b (`train_tokenizer.py --factor_tokenizer`, writing `out_dir/factor_tokenizer/`) → 2 |
| cond | predictor conditioning | Factors concatenated with their missingness mask → MLP bottleneck → zero-initialised projection added to the token embedding, with element-wise and whole-group dropout (`predictor_train.cond_factors: {hidden: 64, dropout: 0.25, group_dropout: 0.2}`) | 1 → 2 |

The final factor column order is written by `factors.py` to `canonical_dir/factor_names.json` (built-ins first,
external second) and read by every later stage. All three modes fill NaN factors with 0 and carry a mask
(repr mode only fills with 0). For `cond` and `token`, `downstream_val` automatically re-runs with the factors
zeroed and logs both the RankIC and the difference. Test-set factors are recomputed by the built-in factor
library and normalised cross-sectionally within the test universe; external factors are unavailable on the
test set and recorded as missing. `downstream_val.min_per_date` (default 30) sets how many symbols a date
needs before it counts toward RankIC.

## tokenizer (the official config.json fields, passed through to KronosTokenizer)
`d_in` (6), `d_model`, `n_heads`, `ff_dim`, `n_enc_layers`, `n_dec_layers`, `ffn_dropout_p`, `attn_dropout_p`,
`resid_dropout_p`, `s1_bits`, `s2_bits` (coarse and fine codebook bits, 10+10 officially), `beta`, `gamma0`,
`gamma`, `zeta` (BSQ loss coefficients), `group_size`. Official Tokenizer-base: 256 / 4 heads / 512 / 4+4 layers.

## predictor (the official config.json fields, passed through to Kronos)
`s1_bits`, `s2_bits` (must match the tokenizer), `n_layers`, `d_model`, `n_heads`, `ff_dim`, `ffn_dropout_p`,
`attn_dropout_p`, `resid_dropout_p`, `token_dropout_p`, `learn_te`. Official sizes: mini roughly 4 layers at
256, small 8 layers at 512 (24.7M), base 12 layers at 832 (102M).

## tokenizer_train / predictor_train (the training loop; both blocks share this structure)
| key | default | what |
|---|---|---|
| micro_batch | required | Samples per GPU per step |
| accum | 1 | Gradient accumulation. Effective batch = micro_batch × accum × world size |
| lr_schedule | cosine | `cosine`: linear warmup from `warmup_start_frac` × peak to peak, then cosine decay to `cos_min_ratio` × peak (the paper's recipe). `linear`: DeepSpeed WarmupDecayLR |
| s2_conditioning (predictor_train only) | teacher | `teacher`: the s2 head is conditioned on the ground-truth s1 (equation 8 of the paper). `sampled`: conditioned on the model's sampled s1 (the upstream fine-tuning script's default) |
| lr | required | Peak learning rate (official fine-tuning: tokenizer 2e-4, predictor 4e-5; pretraining a predictor from scratch tolerates up to 4e-4) |
| warmup_steps | 500 | Linear warmup steps |
| total_steps | required | Total optimisation steps, and the end point of linear decay (DeepSpeed WarmupDecayLR) |
| weight_decay | 0.1 / 0.01 | AdamW with β fixed at (0.9, 0.95). Applied only to non-embedding parameters of rank ≥ 2; biases, norms and embeddings are not decayed (the Moirai / nanoGPT convention) |
| grad_clip | 2.0 | Gradient clipping (officially 2.0 for the tokenizer, 3.0 for the predictor) |
| samples_per_epoch | required | Samples in one "virtual epoch", summed over GPUs. It only affects how often the random stream resets and the epoch counter in checkpoints |
| val_samples | required | Windows drawn per validation pass, summed over GPUs. Validation samples buckets uniformly (alpha fixed at 1.0, i.e. proportional to sample count) |
| log_every / val_every / save_every | required | Counted in optimisation steps |
| workers | 4 | DataLoader processes per GPU. Each replays the same random stream and takes an interleaved slice, so the worker count does not change the data order |
| bit_entropy_weight (tokenizer_train only) | 0.0 | Codebook-utilisation regulariser: for each BSQ bit, compute the in-batch binary entropy H of P(+1) and penalise Σ(1−H). Start between 0.05 and 0.2. It addresses large numbers of codes never activating on a single market |
| distill (predictor_train only) | none | Distillation: `{teacher_path, weight (0.5), temperature (1.0), teacher_cond (false)}`. Teacher and student share a tokenizer; the loss is (1−w)·cross-entropy + w·KL(teacher logits ‖ student logits), averaged over the s1 and s2 heads, with the teacher taking its s2 condition from teacher forcing. The teacher must be an HF-format directory or a Hub name such as NeoQuasar/Kronos-base; the student's architecture is whatever the `predictor` block says |
| forecast_eval (predictor_train only) | none | Autoregressive forecasting on fixed validation windows, compared against persistence: `{n_windows: 256, pred_len: 16, every: <steps>}`. Reports the MAE ratio (≥ 1 means worse than not moving) and last-step directional accuracy, with units aligned to `repr` automatically and buckets mixed in validation-sampling proportions. This is a general check and does not need Qlib |
| downstream_val (predictor_train only) | none | Zero-shot cross-sectional RankIC every `every` steps. Keys: `every` (set it to one virtual epoch's steps to evaluate once per epoch); `pkl` (the CSI300 test set, used for reporting); `val_pkl` (the validation set, used by the stopping rule); `n_dates` (30: pick 30 evenly spaced signal dates and take every symbol on each, roughly 30×300 windows — sampling windows at random leaves only a few symbols per day, which makes cross-sectional RankIC uncomputable); `lookback` (90); `horizon` (10); `market` (CN); `bucket`; `min_per_date` (30); `eval_bs` (1500: windows per autoregressive call, faster when larger if memory allows); `reference` (true: before training, score the official Kronos-small on exactly the same windows as a baseline, and log the difference at every evaluation); `report_test` (false: during training only the validation set is printed, and the test set is scored once at the final evaluation or on early stopping; true prints it every time); `stop_on` (val / test / null: stop and checkpoint once RankIC beats the official baseline by `margin` for `patience` consecutive evaluations — stopping on `test` means selecting the model on the test set and makes reported numbers optimistic, hence the default `val`). Runs on rank 0 only, with the stop signal broadcast to all ranks |

### Two hazards in the upstream model code (worked around here)
1. The cross-attention in the s2 dependency layer sets `is_causal = self.training`, so under `model.eval()` the
   validation loss peeks at future s1. Validation here goes through `common.causal_eval`, which keeps train mode
   and temporarily zeroes all dropout. Autoregressive inference only reads the last position and is unaffected.
2. Passing `padding_mask` and `is_causal` together makes SDPA raise; the model does not actually support
   padding. Windows here are fixed-length and `padding_mask` is never passed.

## ds (DeepSpeed overrides)
| key | default | what |
|---|---|---|
| zero_stage | 2 | ZeRO stage. 2 is enough up to roughly 500M parameters; use 3 only when a single GPU cannot hold the model |
| tensorboard | true | DeepSpeed's own TensorBoard logging (loss, lr, throughput) written to `out_dir/tb/`; view with `tensorboard --logdir runs/<name>/tb` |
| bf16_weights | false | When true, weights and optimizer states are themselves bf16 (DeepSpeed bf16 mode), halving ZeRO-2 communication. When false, weights stay fp32 and only the forward pass runs under autocast |

## ds_config_zero2.json
A template. At runtime the training block and the `ds` block override micro batch, accumulation,
`gradient_clipping`, optimizer, scheduler, ZeRO stage and bf16. Only `reduce_bucket_size` and
`allgather_bucket_size` still need editing by hand.

## Command line
`--config` is required. `--resume` restores exactly from `out_dir/{stage}/latest.json` (model, optimizer,
scheduler, epoch, and samples consumed within the epoch). `--max_steps N` temporarily overrides `total_steps`
for smoke runs. `--overfit_batch` feeds the first batch at every step; the loss must collapse to near zero or
the pipeline is broken. `--local_rank` is injected by the deepspeed launcher.

`run_small.sh` reads two environment variables: `CFG` (default `configs/small.yaml`) and `GPUS` (default `1,2`,
passed to `deepspeed --include localhost:`).

## Deliberately not configurable (these need model-architecture changes; listed so they are not mistaken for switches)
Multi-scale patch tokens; a separate codebook for volume; FiLM modulation by factors (it would let one token
decode to different prices, judged not worth it); sequence packing (Kronos's attention only supports a key
padding mask, not a block-diagonal one); joint fine-tuning of tokenizer and predictor (the predictor's inputs
are integer ids, so there is no differentiable path); activation checkpointing; FlashAttention.
