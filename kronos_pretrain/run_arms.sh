#!/bin/bash
# From-scratch pretraining with factors, three fusion arms, 2 GPUs. Data pipeline runs once (shared shards).
set -eo pipefail; cd "$(dirname "$0")"
source ~/miniforge3/bin/activate tsfm; export PYTHONNOUSERSITE=1
GPUS=${GPUS:-"1,2"}; ARMS=${ARMS:-"cond token repr"}
test -d Kronos || git clone -q --depth 1 https://github.com/shiyu-coder/Kronos.git
python -c "import deepspeed" 2>/dev/null || pip install deepspeed pyyaml pyarrow einops scipy
for a in cond token repr none offtok; do [ -f configs/pretrain_$a.yaml ] || { python make_arms.py; break; }; done   # generate arm configs only if missing (hand-edited ones survive)
CFG=configs/pretrain_cond.yaml
[ -f data/shards_a158/manifest.json ] || {
  echo "== ingest";  python ingest.py  --config $CFG
  echo "== dedupe";  python dedupe.py  --config $CFG
  echo "== factors"; python factors.py --config $CFG          # builtin + Alpha158 (cached after first run)
  echo "== shard";   python shard.py   --config $CFG
  python -c "import yaml,data; c=yaml.safe_load(open('$CFG')); data.summary(c['data']['shard_dir'], c['train'])"
}
for ARM in $ARMS; do
  CFG=configs/pretrain_$ARM.yaml; echo "==== ARM $ARM ($CFG) ===="
  OUT=runs/pretrain_$ARM; mkdir -p $OUT
  TOKP=$(python -c "import yaml; print(yaml.safe_load(open('$CFG')).get('tokenizer_path') or '')")
  if [ -n "$TOKP" ]; then
    echo "== K-line tokenizer: using $TOKP from config"
  elif [ -f $OUT/tokenizer/hf_latest/model.safetensors ]; then
    echo "== K-line tokenizer: reusing $OUT/tokenizer/hf_latest"
  else
    echo "== K-line tokenizer";  deepspeed --include localhost:$GPUS --master_port $((29500 + RANDOM % 400)) train_tokenizer.py --config $CFG 2>&1 | tee $OUT/tokenizer.log
  fi
  if [ "$ARM" = "token" ]; then
    echo "== factor tokenizer";  deepspeed --include localhost:$GPUS --master_port $((29500 + RANDOM % 400)) train_tokenizer.py --config $CFG --factor_tokenizer 2>&1 | tee $OUT/factor_tokenizer.log
  fi
  echo "== predictor";           deepspeed --include localhost:$GPUS --master_port $((29500 + RANDOM % 400)) train_predictor.py --config $CFG 2>&1 | tee $OUT/predictor.log
  echo "---- $ARM downstream RankIC trajectory ----"; grep -h "downstream RankIC" $OUT/predictor.log | tail -25 || true
done
echo ARMS_DONE
