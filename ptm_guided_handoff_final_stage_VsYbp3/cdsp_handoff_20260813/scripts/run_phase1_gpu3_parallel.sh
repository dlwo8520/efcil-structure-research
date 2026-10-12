#!/usr/bin/env bash
set -euo pipefail

# Run the three Phase-1 methods concurrently on one physical GPU while keeping
# datasets sequential: ImageNet-A first, then Cars-196.  Each JSON still runs
# seeds 1993, 1996, and 1997 sequentially inside its own Python process.

project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"

export IMAGENETA_ROOT="${IMAGENETA_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a}"
export CARS196_ROOT="${CARS196_ROOT:-${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196}"
python_bin="${PYTHON_BIN:-python}"
physical_gpu="${PHYSICAL_GPU:-3}"
run_stamp=$(date +%Y%m%d_%H%M%S)
console_log_dir="logs/ptm_guided/parallel_gpu${physical_gpu}/${run_stamp}"
dry_run=false

if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "$console_log_dir"
"$python_bin" scripts/check_ptm_guided_datasets.py --phase1

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU check] Physical GPU ${physical_gpu} before launch"
  nvidia-smi --id="$physical_gpu" \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
fi

active_pids=()

cleanup() {
  if (( ${#active_pids[@]} > 0 )); then
    echo "[Cleanup] Stopping ${#active_pids[@]} active method process(es)..." >&2
    kill "${active_pids[@]}" 2>/dev/null || true
    wait "${active_pids[@]}" 2>/dev/null || true
    active_pids=()
  fi
}

handle_signal() {
  trap - INT TERM EXIT
  cleanup
  exit 130
}

trap handle_signal INT TERM
trap cleanup EXIT

run_dataset() {
  local dataset_label="$1"
  shift
  local entries=("$@")
  local entry method config pid
  local labels=()
  local group_failed=0

  active_pids=()
  echo "[Dataset] ${dataset_label}: launching ${#entries[@]} methods on physical GPU ${physical_gpu}"

  for entry in "${entries[@]}"; do
    method="${entry%%:*}"
    config="${entry#*:}"
    if [[ ! -f "$config" ]]; then
      echo "[Error] Missing config: $config" >&2
      return 1
    fi

    if [[ "$dry_run" == true ]]; then
      echo "[Dry run] CUDA_VISIBLE_DEVICES=${physical_gpu} ${python_bin} -u main.py --config ${config}"
      continue
    fi

    echo "[Launch] ${dataset_label}/${method} -> ${console_log_dir}/${dataset_label}_${method}.log"
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="$physical_gpu" PYTHONUNBUFFERED=1 \
        "$python_bin" -u main.py --config "$config" 2>&1 \
        | sed -u "s|^|[${dataset_label}/${method}] |" \
        | tee "$console_log_dir/${dataset_label}_${method}.log"
    ) &
    pid=$!
    active_pids+=("$pid")
    labels+=("$method")
  done

  if [[ "$dry_run" == true ]]; then
    echo "[Dry run] ${dataset_label}: all three commands would run concurrently"
    return 0
  fi

  for ((entry_index=0; entry_index<${#active_pids[@]}; entry_index++)); do
    pid="${active_pids[$entry_index]}"
    method="${labels[$entry_index]}"
    if wait "$pid"; then
      echo "[Complete] ${dataset_label}/${method}"
    else
      echo "[Failed] ${dataset_label}/${method} (see console log)" >&2
      group_failed=1
    fi
  done
  active_pids=()

  if (( group_failed != 0 )); then
    echo "[Abort] ${dataset_label} had a failed method; the next dataset will not start." >&2
    return 1
  fi
  echo "[Dataset complete] ${dataset_label}: all methods finished"
}

run_dataset "imageneta" \
  "adapter_baseline:exps_jsons/ptm_guided_imageneta_10task_adapter_baseline.json" \
  "ptm_place_only:exps_jsons/ptm_guided_imageneta_10task_ptm_place_only.json" \
  "ptm_place_move:exps_jsons/ptm_guided_imageneta_10task_ptm_place_move.json"

run_dataset "cars196" \
  "adapter_baseline:exps_jsons/ptm_guided_cars196_10task_adapter_baseline.json" \
  "ptm_place_only:exps_jsons/ptm_guided_cars196_10task_ptm_place_only.json" \
  "ptm_place_move:exps_jsons/ptm_guided_cars196_10task_ptm_place_move.json"

if [[ "$dry_run" == false ]]; then
  "$python_bin" scripts/summarize_ptm_guided_results.py
  echo "[Done] Phase 1 complete. Parallel console logs: $console_log_dir"
fi

trap - INT TERM EXIT
