#!/usr/bin/env bash
set -euo pipefail
BOB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BOB_ROOT"
BOB_GPU="${1:-}"
if [[ -z "$BOB_GPU" ]]; then
  read -rp '사용할 GPU 번호: ' BOB_GPU
fi
if [[ ! "$BOB_GPU" =~ ^[0-9]+$ ]]; then
  echo 'GPU 번호는 하나의 0 이상 정수로 입력하세요.' >&2
  exit 2
fi
mkdir -p logs
BOB_STAMP="$(date +%Y%m%d_%H%M%S)"
BOB_LOG="logs/full_gpu${BOB_GPU}_${BOB_STAMP}.log"
nohup bash run_full.sh "$BOB_GPU" > "$BOB_LOG" 2>&1 &
BOB_PID=$!
printf '%s\n' "$BOB_PID" > "logs/full_gpu${BOB_GPU}_${BOB_STAMP}.pid"
echo "PID=$BOB_PID"
echo "LOG=$BOB_ROOT/$BOB_LOG"
echo "진척도: python run.py --stage status"
echo "로그: tail -f '$BOB_ROOT/$BOB_LOG'"
