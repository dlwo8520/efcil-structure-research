#!/usr/bin/env bash
set -euo pipefail
PILOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PILOT_ROOT"
# Packages are installed only into the new folder, never into the original cds environment.
if [[ ! -x .venv_generator/bin/python ]]; then
  ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python -m venv --system-site-packages .venv_generator
fi
.venv_generator/bin/python -m pip install -r requirements-generator.txt
CUDA_VISIBLE_DEVICES='' .venv_generator/bin/python -c 'import torch, diffusers, transformers, accelerate; print("Generation environment:", torch.__version__, diffusers.__version__, transformers.__version__, accelerate.__version__); assert torch.__version__.startswith("2.4.1"), "Expected original cds PyTorch 2.4.1"'
echo "Environment ready; no images generated and no GPU training started."
