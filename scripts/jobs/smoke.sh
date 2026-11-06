#!/usr/bin/env bash
# Runs in: HF Job (t4-small). Nov 9 smoke test:
#   A. AgentTune's own Tiny Aya README example, unchanged (10 GRPO steps).
#   B. Ours: warm start for a few steps on 16 tasks, then 2 GRPO steps with the hardened reward.
# Logs and adapters go to JustEddy1/kataa-runs under smoke/<sha>/ as the job runs.
set -uo pipefail
source "$(dirname "$0")/job_setup.sh"
OUT=smoke/${KATAA_SHA}
mkdir -p /work/logs
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
python -c "from huggingface_hub import hf_hub_download as d; d('CohereLabs/tiny-aya-global', 'config.json'); print('tiny-aya access ok')" 2>&1 | tail -1 | tee /work/logs/access.log

echo "=== A: AgentTune README example ===" 
(cd /work && time python /work/kataa/scripts/jobs/agenttune_tiny_aya_example.py) 2>&1 | tee /work/logs/a_example.log
echo "A exit: ${PIPESTATUS[0]}" | tee -a /work/logs/a_example.log

echo "=== B1: warm start (smoke) ==="
(time python -m kataa.train.warmstart --out /work/ws --hub-path "$OUT/warmstart" --n-tasks 16 \
   --max-steps 8 --grad-accum 2 --merge-to /work/ws-merged) 2>&1 | tee /work/logs/b1_warmstart.log
echo "B1 exit: ${PIPESTATUS[0]}" | tee -a /work/logs/b1_warmstart.log

echo "=== B2: GRPO, hardened reward (smoke) ==="
(time python -m kataa.train.grpo --model /work/ws-merged --reward hardened --out /work/grpo \
   --hub-path "$OUT/grpo" --n-tasks 8 --max-steps 2 --batch 2 --num-generations 2 \
   --max-completion-length 512) 2>&1 | tee /work/logs/b2_grpo.log
echo "B2 exit: ${PIPESTATUS[0]}" | tee -a /work/logs/b2_grpo.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a /work/logs/b2_grpo.log
