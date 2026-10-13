#!/usr/bin/env bash
set -euo pipefail

repo_dir="${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic"
config_path="${repo_dir}/configs/cifar100_10task.yaml"
seed_root="${repo_dir}/outputs/cifar100_10task/seed_0"

run_method() {
  local method="$1"
  if [[ -f "${seed_root}/${method}/COMPLETED.json" ]]; then
    echo "Skipping completed ${method}"
    return
  fi
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n clip \
    python "${repo_dir}/train.py" --config "${config_path}" --method "${method}" --resume
}

# Vanilla generates the 25%-magnitude control; that control generates the
# exactly matched target for the orthogonal-direction condition.
run_method vanilla_lora
run_method small_update_lora
run_method orthogonal_lora
run_method orthogonal_lora_norm_matched

conda run --no-capture-output -n clip python "${repo_dir}/analyze.py" --config "${config_path}"
