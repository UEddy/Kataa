#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Targeted DPO repair of run 2 (kataa/train/dpo.py has the agreed design).
#   1. merge WS2 + run 2 (run 2's LoRA was trained on the merged warm start)
#   2. collect: run 2 greedy on all 1,200 TRAIN tasks (no test, held-out or val)
#   3. pairs from its unsafe decisions (cut at the decision) + DPO with a small SFT term
#      (stops before training if fewer than 100 pairs)
#   4. merge the DPO adapter on top, evaluate greedy on held-out (300); run 2 is re-evaluated in
#      the same job as a reproducibility check (reported only)
#   5. the pre-agreed acceptance gate against run 2's published held-out rows
# Everything is pushed to <runs repo>/dpo/<sha>/ as it goes. Optional env: LR, EPOCHS, SFT_WEIGHT.
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WS=warmstart/f7118070bac9 RUN2=run2/169140127fca
OUT=dpo/${KATAA_SHA}
mkdir -p /work/logs /work/ck /work/merged /work/eval
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
stamp() { echo "[time] $(date -u +%H:%M:%S) $*" | tee -a /work/logs/timing.log; }

stamp start
python -c "from huggingface_hub import snapshot_download as s; s('${KATAA_RUNS_REPO}', local_dir='/work/ck', allow_patterns=['$WS/adapter*', '$RUN2/adapter*'])"
python -m kataa.train.merge --adapter /work/ck/$WS --adapter /work/ck/$RUN2 --out /work/merged/run2 2>&1 | tail -1 | tee -a /work/logs/merge.log
stamp merged-run2

python -m kataa.eval.vllm_eval --name run2__train_n0_s1_t0 --model /work/merged/run2 --split train \
  --samples 1 --temperature 0 --out /work/eval --hub-path "$OUT/eval" 2>&1 | tee /work/logs/collect.log | grep "^\[eval\]"
[ -s /work/eval/run2__train_n0_s1_t0.rows.jsonl ] || { echo "collect failed"; exit 1; }
stamp collected

(time python -m kataa.train.dpo --model /work/merged/run2 \
   --rows /work/eval/run2__train_n0_s1_t0.rows.jsonl \
   --trajectories /work/eval/run2__train_n0_s1_t0.trajectories.jsonl \
   --out /work/dpo --hub-path "$OUT/adapter" --lr ${LR:-2e-5} --epochs ${EPOCHS:-2} \
   --sft-weight ${SFT_WEIGHT:-0.2}) 2>&1 | tee /work/logs/dpo.log
rc=${PIPESTATUS[0]}; echo "dpo exit: $rc" | tee -a /work/logs/dpo.log
[ "$rc" = "0" ] || exit "$rc"
stamp trained

python -m kataa.train.merge --base /work/merged/run2 --adapter /work/dpo --out /work/merged/dpo 2>&1 | tail -1 | tee -a /work/logs/merge.log
stamp merged-dpo
python -m kataa.eval.vllm_eval --name dpo__heldout_n0_s1_t0 --model /work/merged/dpo --split heldout \
  --samples 1 --temperature 0 --out /work/eval --hub-path "$OUT/eval" 2>&1 | tee /work/logs/eval_dpo.log | grep "^\[eval\]"
stamp evaluated-dpo
python -m kataa.eval.vllm_eval --name run2rerun__heldout_n0_s1_t0 --model /work/merged/run2 --split heldout \
  --samples 1 --temperature 0 --out /work/eval --hub-path "$OUT/eval" 2>&1 | tee /work/logs/eval_run2.log | grep "^\[eval\]"
stamp evaluated-run2-rerun

python -m kataa.eval.gate --dpo /work/eval/dpo__heldout_n0_s1_t0.rows.jsonl \
  --rerun /work/eval/run2rerun__heldout_n0_s1_t0.rows.jsonl --out /work/eval/gate.json \
  --hub-path "$OUT/eval" 2>&1 | tee /work/logs/gate.log
stamp done
