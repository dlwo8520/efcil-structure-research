#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python}"
cd "$ROOT"
exec "$PYTHON" -u run_paired_source_prediction_diagnostic.py --gpu 4 "$@"
