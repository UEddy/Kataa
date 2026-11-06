#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Full warm start: 1200 scripted ideal runs, 1 epoch, effective
# batch 8 -> 150 steps. A checkpoint is saved and pushed every 25 steps (step 75 = the mid-run
# checkpoint kept for run-1 plan B); all checkpoints stay on the Hub under warmstart/<sha>/.
set -uo pipefail
source "$(dirname "$0")/job_setup.sh"
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=warmstart/${KATAA_SHA}
mkdir -p /work/logs
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
(time python -m kataa.train.warmstart --out /work/ws --hub-path "$OUT" --epochs 1 \
   --batch 1 --grad-accum 8 --lr 1e-4 --save-steps 25) 2>&1 | tee /work/logs/warmstart.log
echo "warmstart exit: ${PIPESTATUS[0]}" | tee -a /work/logs/warmstart.log
