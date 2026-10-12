#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python}"

cd "$ROOT"
exec "$PYTHON" -u run_ten_task_hypothesis_revalidation.py --gpu 4 "$@"
