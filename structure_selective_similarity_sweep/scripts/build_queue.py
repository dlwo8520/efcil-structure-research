#!/usr/bin/env python3
"""Generate the reproducible 108-run Stage-1 experiment queue."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective.registry import PROTOCOL_ORDER, method_registry


def read_runtime(path, field, fallback_minutes):
    path = Path(path)
    if not path.is_file():
        return fallback_minutes
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))[field]) / 60.0
    except (KeyError, TypeError, ValueError):
        return fallback_minutes


def runtime_estimates():
    ia5 = read_runtime(
        ROOT.parent / "structure_similarity_control_experiments/outputs/imageneta/C0/seed_1993/summary.json",
        "total_training_seconds", 56.0,
    )
    car10 = read_runtime(
        ROOT.parent / "pcds_structure_sgds_official/outputs/cars196_10task_official_structure/seed_0/pcds_structure/summary.json",
        "elapsed_seconds", 204.0,
    )
    return {"IA5": ia5, "IA10": ia5 * 1.70, "CAR5": car10 * 0.58, "CAR10": car10}


def status(run_dir, expected_tasks):
    path = run_dir / "summary.json"
    if not path.is_file():
        return "pending"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") == "complete" and int(summary.get("completed_tasks", 0)) == expected_tasks:
        return "complete"
    return f"partial ({summary.get('completed_tasks', 0)}/{expected_tasks})"


def main():
    output_root = ROOT / "outputs"
    estimates = runtime_estimates()
    lines = [
        "# Similarity-selective Stage-1 experiment queue", "",
        "총 108 runs = 4 protocols × (C0 1 + A 5 + B 9 + C 12), seed 1993.", "",
        "예상 시간은 기존 IA5 C0 `total_training_seconds`와 기존 Cars-196 10-task Structure "
        "`elapsed_seconds`에서 자동으로 읽었습니다. IA10은 IA5의 1.70배, CAR5는 CAR10의 "
        "0.58배로 외삽한 계획값이며 실제 wall-clock 보장은 아닙니다.", "",
        "| Protocol | Per-run estimate | 27-run estimate |", "|---|---:|---:|",
    ]
    for protocol in PROTOCOL_ORDER:
        minutes = estimates[protocol]
        lines.append(f"| {protocol} | {minutes:.1f} min | {minutes * 27 / 60:.1f} h |")
    total_hours = sum(estimates.values()) * 27 / 60
    lines.extend(
        ["", f"전체 단일 GPU 예상: **{total_hours:.1f} h ({total_hours / 24:.1f} days)**.", "",
         "| order | protocol | dataset | num_tasks | run_id | method | P | alpha | lambda_clad | seed | command | estimated_time | status | output_dir |",
         "|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---:|---|---|" ]
    )
    order = 0
    for protocol in PROTOCOL_ORDER:
        dataset = "ImageNet-A" if protocol.startswith("IA") else "Cars-196"
        tasks = 5 if protocol.endswith("5") else 10
        root = ("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)) if protocol.startswith("IA") else ("${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))
        for run_id, config in method_registry().items():
            order += 1
            run_dir = output_root / protocol / run_id / "seed_1993"
            command = f"python run.py --protocol {protocol} --run-id {run_id} --gpu 1 --data-root {root} --seed 1993"
            value = lambda item: "-" if item is None else item
            lines.append(
                f"| {order} | {protocol} | {dataset} | {tasks} | {run_id} | {config['method']} | "
                f"{value(config['top_p'])} | {value(config['alpha'])} | {value(config['lambda_clad'])} | "
                f"1993 | `{command}` | {estimates[protocol]:.1f} min | {status(run_dir, tasks)} | `{run_dir}` |"
            )
    path = ROOT / "experiment_queue.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {path} rows={order}")


if __name__ == "__main__":
    main()
