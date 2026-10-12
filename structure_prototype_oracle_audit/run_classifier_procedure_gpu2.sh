#!/usr/bin/env bash
set -euo pipefail

ROOT="${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit"
CONDA_SH="${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh"
OUTPUT_ROOT="${ROOT}/classifier_procedure_diagnostic_gpu2"
LOG_FILE="${ROOT}/logs/classifier_procedure_t5_gpu2.log"

source "${CONDA_SH}"
conda activate cds
cd "${ROOT}"
mkdir -p "${ROOT}/logs" "${OUTPUT_ROOT}"

python run_classifier_procedure_diagnostic.py \
  --protocol IA5 \
  --protocol CAR5 \
  --gpu 2 \
  --seed 1993 \
  --output-root "${OUTPUT_ROOT}" \
  2>&1 | tee "${LOG_FILE}"

