#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). One GRPO run from a warm-start checkpoint, colocated vLLM.
# Required env: RUN (e.g. run1), REWARD (naive|hardened), START (Hub path of the adapter dir,
# e.g. warmstart/4ef6d074ec4d or warmstart/4ef6d074ec4d/checkpoint-75).
# 120 steps x 8 rollouts (2 tasks x 4 samples), one rollout per micro-batch (4 per micro-batch
# ran out of memory in backward on Tiny Aya's 262K-vocab logits, job 6aedc458). Pushed every 20.
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${RUN}/${KATAA_SHA}
mkdir -p /work/logs /work/start
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
echo "run=$RUN reward=$REWARD start=$START" | tee /work/logs/config.log
python -c "from huggingface_hub import snapshot_download as s; s('${KATAA_RUNS_REPO}', local_dir='/work/start', allow_patterns=['${START}/adapter*'])"
python -m kataa.train.merge --adapter "/work/start/${START}" --out /work/start-merged  # this run's LoRA sits on top of this 2>&1 | tee -a /work/logs/config.log
(time python -m kataa.train.grpo --model /work/start-merged --reward "$REWARD" --out /work/run \
   --hub-path "$OUT" --max-steps ${MAX_STEPS:-120} --batch 1 --grad-accum 8 --num-generations 4 \
   --lr 1e-5 --beta 0.04 --temperature 0.9 --max-completion-length 512 --save-steps 20 \
   --vllm --vllm-mem 0.35) 2>&1 | tee /work/logs/grpo.log
echo "grpo exit: ${PIPESTATUS[0]}" | tee -a /work/logs/grpo.log
