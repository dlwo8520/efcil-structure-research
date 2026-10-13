#!/usr/bin/env bash
set -euo pipefail
BOB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd "$BOB_ROOT"
BOB_GPU="${1:-}"
if [[ -z "$BOB_GPU" ]]; then
  read -rp '사용할 GPU 번호: ' BOB_GPU
fi
if [[ ! "$BOB_GPU" =~ ^[0-9]+$ ]]; then
  echo 'GPU 번호는 하나의 0 이상 정수로 입력하세요.' >&2
  exit 2
fi
exec python -u run.py --stage full --gpu "$BOB_GPU"
