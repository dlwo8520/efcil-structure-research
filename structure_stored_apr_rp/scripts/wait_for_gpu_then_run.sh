#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash scripts/wait_for_gpu_then_run.sh GPU_ID" >&2
    exit 2
fi

GPU_ID="$1"
cd ${RESEARCH_WORKSPACE}/structure_stored_apr_rp
while [[ -n "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "$(date -Is) GPU $GPU_ID busy; waiting 60 seconds" 
    sleep 60
done
echo "$(date -Is) GPU $GPU_ID appears free; confirming after 60 seconds"
sleep 60
if [[ -n "$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader)" ]]; then
    exec bash scripts/wait_for_gpu_then_run.sh "$GPU_ID"
fi
echo "$(date -Is) GPU $GPU_ID has no compute process; launching StoredAPR queue"
exec bash run_stored_apr_rp.sh "$GPU_ID"
