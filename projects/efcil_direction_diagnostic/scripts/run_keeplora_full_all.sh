#!/usr/bin/env bash
set -euo pipefail

repo_dir=${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic
python_bin=${RESEARCH_WORKSPACE}/miniconda3/envs/clip/bin/python
config_path=configs/keeplora_efcil_cifar100_10task.yaml

cd "$repo_dir"
for method in vanilla_lora keep_lora_wp_only keep_lora_m_only keep_lora_full; do
  CUDA_VISIBLE_DEVICES=1 "$python_bin" train_keeplora.py --config "$config_path" --method "$method"
done
CUDA_VISIBLE_DEVICES=1 "$python_bin" analyze_keeplora.py --config "$config_path"
