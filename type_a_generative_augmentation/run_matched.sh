#!/usr/bin/env bash
set -euo pipefail
PILOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PILOT_GPU="${1:?Usage: bash run_matched.sh GPU_NUMBER}"
[[ "$PILOT_GPU" =~ ^[0-9]+$ ]] || { echo "GPU number must be nonnegative" >&2; exit 2; }
cd "$PILOT_ROOT"
mkdir -p logs
exec 9>logs/pilot.queue.lock
flock -n 9 || { echo "Another Type A pilot stage holds this folder's queue lock" >&2; exit 1; }
set +u
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
set -u
export PYTHONUNBUFFERED=1
if [[ -f exploratory_config.json ]]; then
  if [[ ! -x .venv_generator/bin/python ]]; then
    echo "Exploratory sanity requires the installed .venv_generator" >&2
    exit 1
  fi
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  .venv_generator/bin/python -u run.py --stage auto-sanity --gpu "$PILOT_GPU"
fi
python -u run.py --stage quality-check
python -u run.py --stage train --gpu "$PILOT_GPU"
echo "Matched pilot and report complete: final_report.md"
