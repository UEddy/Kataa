#!/usr/bin/env bash
# Runs in: cloud session (paste into the environment's setup script, or run by hand).
# Installs Kataa's CPU-side dependencies into /opt/kataa-venv. No secrets are read or written.
# Measured 2026-11-06: 35 s from a cold cache (CPU torch from download.pytorch.org).
set -euo pipefail

AGENTTUNE_REF=d4db4c12d52342227c4113a70183d0f64aa2f345   # Lexsi-Labs/AgentTune main, 2026-10-07
AUDITKIT_REF=71316da345784ff220d5ed59337441a59ab31526    # Lexsi-Labs/AuditKIT main, 2026-10-07 (v1.1.3)
DEPS=${KATAA_DEPS:-/opt/kataa-deps}
VENV=${KATAA_VENV:-/opt/kataa-venv}

mkdir -p "$DEPS"
fetch() {  # fetch <repo> <ref> <dir>
  if [ ! -d "$3/.git" ]; then git clone -q --filter=blob:none "https://github.com/$1.git" "$3"; fi
  git -C "$3" checkout -q "$2" 2>/dev/null || { git -C "$3" fetch -q origin; git -C "$3" checkout -q "$2"; }
}
fetch Lexsi-Labs/AgentTune "$AGENTTUNE_REF" "$DEPS/AgentTune" &
fetch Lexsi-Labs/AuditKIT  "$AUDITKIT_REF"  "$DEPS/AuditKIT" &
wait

command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.13 "$VENV"
export VIRTUAL_ENV="$VENV"
# CPU build of the exact torch AgentTune pins (2.11.0; "+cpu" satisfies ==2.11.0). Installing it
# first keeps the ~4 GB CUDA wheels out of the cloud session. GPU jobs install CUDA torch themselves.
uv pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -q -e "$DEPS/AgentTune" "$DEPS/AuditKIT[transformers]" pytest mistralai
"$VENV/bin/python" -c "import torch, transformers, trl, agenttune, auditkit; print('ok', torch.__version__, transformers.__version__, trl.__version__)"
