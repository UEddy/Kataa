#!/usr/bin/env bash
# Runs in: cloud session. Upload a snapshot of HEAD to the runs repo, then launch a job script.
#   [JOB_ENV="K=V K2=V2"] scripts/jobs/launch.sh <job-script> <flavor> <timeout> [--dry-run]
# The HF token is passed only with `--secrets HF_TOKEN` (read from this shell's environment,
# encrypted by HF Jobs); it is never written to the command line or a file.
set -euo pipefail
cd "$(dirname "$0")/../.."
SCRIPT=$1 FLAVOR=$2 TIMEOUT=$3 EXTRA=${4:-}
[ -z "$(git status --porcelain)" ] || { echo "commit first: working tree is dirty"; exit 1; }
SHA=$(git rev-parse --short=12 HEAD)
RUNS=${KATAA_RUNS_REPO:-JustEddy1/kataa-runs}
git archive --format=tar.gz -o "/tmp/kataa-${SHA}.tar.gz" HEAD
hf upload "$RUNS" "/tmp/kataa-${SHA}.tar.gz" "code/${SHA}.tar.gz" --commit-message "code ${SHA}" >/dev/null
echo "code snapshot: $RUNS/code/${SHA}.tar.gz"
hf jobs run $EXTRA --detach --flavor "$FLAVOR" --timeout "$TIMEOUT" --name "kataa-$(basename "$SCRIPT" .sh)-${SHA}" \
  --secrets HF_TOKEN --env "KATAA_SHA=${SHA}" --env "KATAA_RUNS_REPO=${RUNS}" \
  $(for kv in ${JOB_ENV:-}; do printf -- '--env %s ' "$kv"; done) \
  python:3.12 bash -c "pip install -q huggingface_hub && \
    hf download ${RUNS} code/${SHA}.tar.gz --local-dir /tmp/c >/dev/null && \
    mkdir -p /work/kataa && tar -xzf /tmp/c/code/${SHA}.tar.gz -C /work/kataa && \
    bash /work/kataa/${SCRIPT}"
