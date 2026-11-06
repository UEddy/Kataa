#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Measures GRPO rollout speed with colocated vLLM (vs ~45 s per
# rollout with plain transformers generation in smoke/b10c3e66637d).
#   1. 8-step warm start on 16 tasks, merged (the starting model for GRPO).
#   2. GRPO, hardened reward, vLLM colocate: 2 steps x (1 prompt x 4 generations).
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=smoke-vllm/${KATAA_SHA}
mkdir -p /work/logs
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" 2>&1 | tail -1 | tee /work/logs/versions.log

(time python -m kataa.train.warmstart --out /work/ws --hub-path "$OUT/warmstart" --n-tasks 16 \
   --max-steps 8 --grad-accum 2 --merge-to /work/ws-merged) 2>&1 | tee /work/logs/b1_warmstart.log
echo "B1 exit: ${PIPESTATUS[0]}" | tee -a /work/logs/b1_warmstart.log

(time python -m kataa.train.grpo --model /work/ws-merged --reward hardened --out /work/grpo \
   --hub-path "$OUT/grpo" --n-tasks 8 --max-steps 2 --batch 4 --num-generations 4 \
   --max-completion-length 512 --vllm --vllm-mem 0.35) 2>&1 | tee /work/logs/b2_grpo_vllm.log
echo "B2 exit: ${PIPESTATUS[0]}" | tee -a /work/logs/b2_grpo_vllm.log
