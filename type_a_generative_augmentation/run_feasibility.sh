#!/usr/bin/env bash
set -euo pipefail
PILOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PILOT_GPU="${1:?Usage: bash run_feasibility.sh GPU_NUMBER}"
[[ "$PILOT_GPU" =~ ^[0-9]+$ ]] || { echo "GPU number must be nonnegative" >&2; exit 2; }
cd "$PILOT_ROOT"
mkdir -p logs
exec 9>logs/pilot.queue.lock
flock -n 9 || { echo "Another Type A pilot stage holds this folder's queue lock" >&2; exit 1; }
if [[ ! -x .venv_generator/bin/python ]]; then
  echo "Run bash setup_generator_env.sh first. No GPU experiment was started." >&2
  exit 1
fi
set +u
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
set -u
export PYTHONUNBUFFERED=1
python -u run.py --stage preflight
CUDA_VISIBLE_DEVICES='' .venv_generator/bin/python -c 'import torch, diffusers, transformers, accelerate; from diffusers import StableDiffusionImg2ImgPipeline; print("Generator environment/imports PASS:", diffusers.__version__, transformers.__version__, accelerate.__version__)'
python -u run.py --stage audit --gpu "$PILOT_GPU"
python -u run.py --stage prepare
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
.venv_generator/bin/python -u run.py --stage generate --gpu "$PILOT_GPU"
if [[ -f exploratory_config.json ]]; then
  echo "Generation complete. run_matched.sh performs automatic sanity before exploratory training. Contact sheets are artifacts only."
else
  echo "STOP FOR HUMAN REVIEW: contact_sheets/ and quality_review.csv. Training has NOT started."
fi
