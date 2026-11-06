#!/usr/bin/env bash
# Runs in: cloud session (paste into the environment's setup script, or run by hand).
# Installs Kataa's CPU-side dependencies into /opt/kataa-venv. Target: under ~5 minutes.
# No secrets are read or written here.
set -euo pipefail

AGENTTUNE_REF=d4db4c12d52342227c4113a70183d0f64aa2f345   # Lexsi-Labs/AgentTune main, 2026-10-07
AUDITKIT_REF=71316da345784ff220d5ed59337441a59ab31526    # Lexsi-Labs/AuditKIT main, 2026-10-07 (v1.1.3)
DEPS=/opt/kataa-deps
VENV=/opt/kataa-venv

mkdir -p "$DEPS"
fetch() {  # fetch <repo> <ref> <dir>
  if [ ! -d "$3/.git" ]; then git clone -q "https://github.com/$1.git" "$3"; fi
  git -C "$3" fetch -q origin "$2" 2>/dev/null || git -C "$3" fetch -q origin
  git -C "$3" checkout -q "$2"
}
fetch Lexsi-Labs/AgentTune "$AGENTTUNE_REF" "$DEPS/AgentTune"
fetch Lexsi-Labs/AuditKIT  "$AUDITKIT_REF"  "$DEPS/AuditKIT"

command -v uv >/dev/null || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.13 "$VENV"
export VIRTUAL_ENV="$VENV"
# AgentTune pins torch==2.11.0 exactly; PyPI is the only torch index reachable from the cloud
# session (download.pytorch.org is blocked), so this pulls the CUDA build. It imports fine on CPU.
uv pip install -q -e "$DEPS/AgentTune" "$DEPS/AuditKIT[transformers]" pytest
# Measured 2026-11-06: about 6.5 min on a fresh container, mostly the ~4 GB CUDA torch wheels.
"$VENV/bin/python" -c "import torch, transformers, trl, agenttune, auditkit; print('ok', torch.__version__, transformers.__version__, trl.__version__)"
