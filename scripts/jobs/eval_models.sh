#!/usr/bin/env bash
# Runs in: HF Job (l40sx1). Evaluate LoRA adapters stored in the runs repo (merged first; vLLM
# 0.22.1 LoRA crashes on Tiny Aya). Env:
#   MODELS="name:hub/path|name2:path2" (| separates entries; launch.sh splits JOB_ENV on spaces).
#   A GRPO run is "name:warmstart/path+run/path": adapters merge in order (the run's LoRA was
#   trained on top of the merged warm start).   ("base" as the name with path "-" = base Tiny Aya)
#   EVALS="split:n_tasks:samples:temperature|..." (n_tasks 0 = whole split), TAG (output folder)
set -uo pipefail
export KATAA_VLLM=1
source "$(dirname "$0")/job_setup.sh"
set +e
OUT=eval/${TAG}/${KATAA_SHA}
mkdir -p /work/logs /work/ck /work/merged
push_logs() { python -c "from kataa.train.hub import push_folder; push_folder('/work/logs', '$OUT/logs', '$OUT: logs')"; }
trap push_logs EXIT
cd /work/kataa && export PYTHONPATH=/work/kataa
for pair in ${MODELS//|/ }; do
  name=${pair%%:*}; path=${pair#*:}
  if [ "$path" = "-" ]; then model=CohereLabs/tiny-aya-global; else
    args=""
    for p in ${path//+/ }; do
      python -c "from huggingface_hub import snapshot_download as s; s('${KATAA_RUNS_REPO}', local_dir='/work/ck', allow_patterns=['$p/adapter*'])"
      args="$args --adapter /work/ck/$p"
    done
    python -m kataa.train.merge $args --out "/work/merged/$name" 2>&1 | tail -1 | tee -a /work/logs/merge.log
    model=/work/merged/$name
  fi
  for ev in ${EVALS//|/ }; do
    IFS=: read -r split n samples temp <<< "$ev"
    nt=""; [ "$n" != "0" ] && nt="--n-tasks $n"
    python -m kataa.eval.vllm_eval --name "${name}__${split}_n${n}_s${samples}_t${temp}" --model "$model" \
      --split "$split" $nt --samples "$samples" --temperature "$temp" --out /work/eval \
      --hub-path "$OUT" 2>&1 | tee -a "/work/logs/${name}.log" | grep "^\[eval\]"
  done
  rm -rf "/work/merged/$name"
done
