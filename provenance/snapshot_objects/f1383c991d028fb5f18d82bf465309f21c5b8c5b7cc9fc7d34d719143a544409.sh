#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash run_reduced_experiments.sh GPU_ID [--dry-run]" >&2
    exit 2
fi

GPU_ID="$1"
shift
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be one non-negative integer: ${GPU_ID}" >&2
    exit 2
fi

PROJECT_ROOT="${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep"
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd "${PROJECT_ROOT}"

exec python scripts/run_reduced_experiments.py \
    --gpu "${GPU_ID}" \
    --imageneta-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
    --cars-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 \
    --output-root "${PROJECT_ROOT}/outputs" \
    --seed 1993 \
    "$@"
