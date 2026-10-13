#!/usr/bin/env bash
set -euo pipefail
BOB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
cd "$BOB_ROOT"
# New project-scoped environment only; cds and the old generator env stay intact.
if [[ ! -x "$BOB_ROOT/.bob_env/bin/python" ]]; then
  conda create -y -p "$BOB_ROOT/.bob_env" python=3.10 pip
fi
"$BOB_ROOT/.bob_env/bin/python" -m pip install -r requirements-bob.txt
# Downloads the caption model once and pins its resolved revision in model_lock.
# No caption inference, LoRA training, image generation or Structure training here.
CUDA_VISIBLE_DEVICES='' "$BOB_ROOT/.bob_env/bin/python" -m fulltraj.bob_worker assets
echo 'BOB environment and frozen caption model ready. No experiments started.'
