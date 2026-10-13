#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_rp_protoclad_task5.sh GPU_ID [--dry-run]" >&2
    exit 2
fi

GPU_ID="$1"
shift
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one non-negative integer" >&2
    exit 2
fi

ROOT="${RESEARCH_WORKSPACE}/structure_rp_protoclad_task5"
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd "$ROOT"
exec python scripts/run_rp_experiments.py --gpu "$GPU_ID" "$@"
