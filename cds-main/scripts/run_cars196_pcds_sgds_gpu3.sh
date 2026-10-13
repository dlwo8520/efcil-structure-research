#!/usr/bin/env bash
set -euo pipefail

cd ${RESEARCH_WORKSPACE}/cds-main

cars_root=${RESEARCH_WORKSPACE}/cds-main/data/cars196
if [[ ! -d "${cars_root}/train" || ! -d "${cars_root}/test" ]]; then
  echo "Cars-196 data is missing. Expected ${cars_root}/train/<class>/ and ${cars_root}/test/<class>/." >&2
  exit 2
fi

nvidia-smi -i 3 --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader

export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1

${RESEARCH_WORKSPACE}/miniconda3/envs/clip/bin/python main.py \
  --config exps_jsons/cars196_pcds.json \
  --config exps_jsons/cars196_pcds_structure_sgds.json
