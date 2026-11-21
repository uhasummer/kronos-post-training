#!/bin/bash
# Small end-to-end run: data pipeline (CPU) then 2-GPU DeepSpeed for tokenizer and predictor.
set -e; cd "$(dirname "$0")"
source ~/miniforge3/bin/activate tsfm; export PYTHONNOUSERSITE=1
CFG=${CFG:-configs/small.yaml}; GPUS=${GPUS:-"1,2"}; MS=${MAX_STEPS:+--max_steps $MAX_STEPS}   # MAX_STEPS=30 = quick DeepSpeed link check
export TABPFN_TOKEN=; test -d Kronos || git clone -q --depth 1 https://github.com/shiyu-coder/Kronos.git
python -c "import deepspeed" 2>/dev/null || pip install deepspeed pyyaml pyarrow einops yfinance lxml
echo "== ingest";  python ingest.py --config $CFG
echo "== dedupe";  python dedupe.py --config $CFG
echo "== factors"; python factors.py --config $CFG
echo "== shard";   python shard.py  --config $CFG
echo "== sampler"; python -c "import yaml,data; c=yaml.safe_load(open('$CFG')); data.summary(c['data']['shard_dir'], c['train'])"
echo "== tokenizer"; deepspeed --include localhost:$GPUS --master_port 29571 train_tokenizer.py --config $CFG $MS
if python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); sys.exit(0 if c.get('factors') and c['train'].get('factor_mode')=='token' else 1)"; then
  echo "== factor tokenizer"; deepspeed --include localhost:$GPUS --master_port 29573 train_tokenizer.py --config $CFG --factor_tokenizer $MS
fi
echo "== predictor"; deepspeed --include localhost:$GPUS --master_port 29572 train_predictor.py --config $CFG $MS
echo SMALL_RUN_DONE
