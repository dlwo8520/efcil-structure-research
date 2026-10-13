#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

export IMAGENETA_ROOT="${IMAGENETA_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a}"
export CARS196_ROOT="${CARS196_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" scripts/check_ptm_guided_datasets.py --phase1

configs=(
  exps_jsons/ptm_guided_imageneta_10task_adapter_baseline.json
  exps_jsons/ptm_guided_imageneta_10task_ptm_place_only.json
  exps_jsons/ptm_guided_imageneta_10task_ptm_place_move.json
  exps_jsons/ptm_guided_cars196_10task_adapter_baseline.json
  exps_jsons/ptm_guided_cars196_10task_ptm_place_only.json
  exps_jsons/ptm_guided_cars196_10task_ptm_place_move.json
)

for config in "${configs[@]}"; do
  echo "[Phase 1] Starting $config"
  CUDA_VISIBLE_DEVICES=4 "$python_bin" main.py --config "$config"
done

"$python_bin" scripts/summarize_ptm_guided_results.py
