#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash run_stored_apr_rp.sh GPU_ID [--dry-run]" >&2
    exit 2
fi

GPU_ID="$1"
shift
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_stored_apr_rp
exec python run_all.py --gpu "$GPU_ID" "$@"
