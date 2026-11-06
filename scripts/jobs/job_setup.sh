#!/usr/bin/env bash
# Runs in: HF Job. Shared setup: CUDA torch and pinned AgentTune. The Kataa code snapshot was
# already unpacked to /work/kataa by scripts/jobs/launch.sh. HF_TOKEN arrives via HF Jobs'
# encrypted secrets and is never echoed.
set -euo pipefail
AGENTTUNE_REF=d4db4c12d52342227c4113a70183d0f64aa2f345
export PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
pip install -q uv
mkdir -p /work && cd /work
git clone -q https://github.com/Lexsi-Labs/AgentTune.git && git -C AgentTune checkout -q "$AGENTTUNE_REF"
uv venv -q --python 3.12 /opt/v && export VIRTUAL_ENV=/opt/v PATH=/opt/v/bin:$PATH
if [ "${KATAA_VLLM:-0}" = "1" ]; then
  # AgentTune's [vllm] extra with the vLLM version its pyproject says it resolved and tested
  # (0.22.1; unpinned it now resolves 0.23.0). Default PyPI torch 2.11.0 (CUDA 13; driver 580).
  uv pip install -q -e "/work/AgentTune[vllm]" "vllm==0.22.1"
  # FlashInfer's sampler JIT-compiles a CUDA kernel on first use and needs nvcc, which this image
  # lacks (job 6aedbe20c78863f9c6341fa5). Use vLLM's PyTorch sampler instead; fail fast if this
  # vLLM version does not know the switch.
  export VLLM_USE_FLASHINFER_SAMPLER=0
  python -c "import vllm.envs as e; assert e.VLLM_USE_FLASHINFER_SAMPLER is False, e.VLLM_USE_FLASHINFER_SAMPLER; print('flashinfer sampler off')"
else
  uv pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q -e /work/AgentTune
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
