#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/run_sweeps.py" "$@"
