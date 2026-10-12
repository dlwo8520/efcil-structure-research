#!/usr/bin/env bash
set -euo pipefail

repo_dir=${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic
python_bin=${RESEARCH_WORKSPACE}/miniconda3/envs/clip/bin/python
config_path=configs/pcds_efcil_cifar100_10task.yaml

cd "$repo_dir"
CUDA_VISIBLE_DEVICES=1 "$python_bin" pcds_static_check.py --config "$config_path"
CUDA_VISIBLE_DEVICES=1 "$python_bin" train_pcds.py \
  --config "$config_path" \
  --method pcds_full
"$python_bin" analyze_pcds.py --config "$config_path"
