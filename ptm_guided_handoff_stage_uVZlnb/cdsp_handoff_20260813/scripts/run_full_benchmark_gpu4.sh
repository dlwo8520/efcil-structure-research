#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

export IMAGENETA_ROOT="${IMAGENETA_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a}"
export CARS196_ROOT="${CARS196_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196}"
: "${OBJECTNET_ROOT:?Set OBJECTNET_ROOT to the licensed processed ObjectNet CIL root}"
: "${OMNIBENCHMARK_ROOT:?Set OMNIBENCHMARK_ROOT to the processed OmniBenchmark root}"
: "${IMAGENETR_ROOT:?Set IMAGENETR_ROOT to the processed ImageNet-R root}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" scripts/check_ptm_guided_datasets.py --all

configs=(
  exps_jsons/ptm_guided_imageneta_5task_ptm_place_move.json
  exps_jsons/ptm_guided_imageneta_20task_ptm_place_move.json
  exps_jsons/ptm_guided_cars196_5task_ptm_place_move.json
  exps_jsons/ptm_guided_cars196_20task_ptm_place_move.json
  exps_jsons/ptm_guided_objectnet_10task_ptm_place_move.json
  exps_jsons/ptm_guided_omnibenchmark_10task_ptm_place_move.json
  exps_jsons/ptm_guided_imagenetr_10task_ptm_place_move.json
)

for config in "${configs[@]}"; do
  echo "[Full benchmark] Starting $config"
  CUDA_VISIBLE_DEVICES=4 "$python_bin" main.py --config "$config"
done

"$python_bin" scripts/summarize_ptm_guided_results.py
