#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python run_representation_relation_diagnostic.py \
  --gpu 2 \
  --protocol IA5 \
  --protocol CAR5 \
  "$@"
