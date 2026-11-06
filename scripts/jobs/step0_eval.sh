#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Run-1 step 0: which warm-start checkpoint to start GRPO from.
# Samples 4 rollouts (temperature 0.9, as in GRPO) on 40 val tasks for base Tiny Aya and
# warm-start checkpoints 25 / 75 / 150, and reports accuracy, evidence, unsafe rate and the share
# of task groups whose naive reward varies (the learning signal run 1 would get).
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
WS=${WARMSTART:-warmstart/4ef6d074ec4d}
OUT=eval/step0/${KATAA_SHA}
mkdir -p /work/logs /work/ck
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("${KATAA_RUNS_REPO}", local_dir="/work/ck", allow_patterns=[
    "$WS/checkpoint-25/adapter*", "$WS/checkpoint-75/adapter*", "$WS/adapter*"])
PY
(time python -m kataa.eval.vllm_eval --split val --n-tasks 40 --samples 4 --temperature 0.9 \
   --include-base --adapter "ws25=/work/ck/$WS/checkpoint-25" \
   --adapter "ws75=/work/ck/$WS/checkpoint-75" --adapter "ws150=/work/ck/$WS" \
   --out /work/eval --hub-path "$OUT") 2>&1 | tee /work/logs/step0.log
echo "step0 exit: ${PIPESTATUS[0]}" | tee -a /work/logs/step0.log
