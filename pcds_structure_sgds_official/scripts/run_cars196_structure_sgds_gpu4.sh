#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${RESEARCH_WORKSPACE}/pcds_structure_sgds_official"
PYTHON_BIN="${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python"
DATA_ROOT="${RESEARCH_WORKSPACE}/cds-main/data/cars196"
OUTPUT_BASE="$REPO_ROOT/outputs/cars196_10task_official_structure/seed_0"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing official CDS Python environment: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -d "$DATA_ROOT/train" || ! -d "$DATA_ROOT/test" ]]; then
    echo "Missing prepared Cars-196 split at $DATA_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"
mkdir -p logs "$OUTPUT_BASE"

export CUDA_VISIBLE_DEVICES=4
echo "Physical GPU visibility: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --id=4 \
    --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

"$PYTHON_BIN" -m unittest \
    tests.test_sgds_activation_guidance \
    tests.test_vit_sgds_integration
"$PYTHON_BIN" scripts/check_protocol.py | tee logs/cars196_structure_protocol.log

run_method() {
    local method="$1"
    local config="$2"
    local log_file="$3"
    if [[ -f "$OUTPUT_BASE/$method/COMPLETED.json" ]]; then
        echo "Skipping completed method: $method"
        return
    fi
    "$PYTHON_BIN" main.py --config "$config" 2>&1 | tee "$log_file"
}

run_method \
    "pcds_structure" \
    "exps_jsons/cars196_structure_pcds_seed0.json" \
    "logs/cars196_structure_pcds_seed0.log"

run_method \
    "pcds_structure_sgds" \
    "exps_jsons/cars196_structure_pcds_sgds_seed0.json" \
    "logs/cars196_structure_pcds_sgds_seed0.log"

"$PYTHON_BIN" scripts/analyze_structure_sgds.py
echo "Complete: $REPO_ROOT/RESULTS_STRUCTURE_SGDS.md"
