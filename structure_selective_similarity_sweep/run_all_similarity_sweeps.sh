#!/usr/bin/env bash
set -Eeuo pipefail

echo "The legacy 108-run sweep is disabled for reduced Stage 1." >&2
echo "Use: bash run_reduced_experiments.sh GPU_ID" >&2
exit 2
