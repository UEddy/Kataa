#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Run-1 step 0: which warm-start checkpoint to start GRPO from.
# Samples 4 rollouts (temperature 0.9, as in GRPO) on 40 val tasks for base Tiny Aya and
# warm-start checkpoints 25 / 75 / 150, and reports accuracy, evidence, unsafe rate and the share
# of task groups whose naive reward varies (the learning signal run 1 would get).
# Checkpoints are merged first (vLLM 0.22.1 LoRA crashes on Tiny Aya); one model per process.
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
WS=${WARMSTART:-warmstart/4ef6d074ec4d}
OUT=eval/step0/${KATAA_SHA}
mkdir -p /work/logs /work/ck /work/merged
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
python -c "from huggingface_hub import snapshot_download as s; s('${KATAA_RUNS_REPO}', local_dir='/work/ck', allow_patterns=['$WS/checkpoint-25/adapter*', '$WS/checkpoint-75/adapter*', '$WS/adapter*'])"
EVAL="python -m kataa.eval.vllm_eval --split val --n-tasks 40 --samples 4 --temperature 0.9 --out /work/eval --hub-path $OUT"
$EVAL --name base 2>&1 | tee /work/logs/base.log
for pair in ws25:$WS/checkpoint-25 ws75:$WS/checkpoint-75 ws150:$WS; do
  name=${pair%%:*}; path=${pair#*:}
  python -m kataa.train.merge --adapter "/work/ck/$path" --out "/work/merged/$name" 2>&1 | tail -1 | tee -a /work/logs/merge.log
  $EVAL --name "$name" --model "/work/merged/$name" 2>&1 | tee "/work/logs/$name.log"
  rm -rf "/work/merged/$name"
done
grep -h "^\[eval\]" /work/logs/*.log | tee /work/logs/summary.log
