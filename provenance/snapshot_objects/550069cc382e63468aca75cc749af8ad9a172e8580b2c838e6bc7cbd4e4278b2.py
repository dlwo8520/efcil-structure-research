#!/usr/bin/env python3
"""Show queue/task/epoch progress without touching running processes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective.registry import PROTOCOL_ORDER, method_registry


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=1993)
    cli = parser.parse_args()
    output = Path(cli.output_root).expanduser().resolve()
    totals = {"complete": 0, "running": 0, "failed": 0, "pending": 0, "partial": 0}
    for protocol in PROTOCOL_ORDER:
        expected = 5 if protocol.endswith("5") else 10
        for run_id in method_registry():
            run_dir = output / protocol / run_id / f"seed_{cli.seed}"
            meta, summary = read(run_dir / "run_metadata.json"), read(run_dir / "summary.json")
            if summary.get("status") == "complete" and int(summary.get("completed_tasks", 0)) == expected:
                state = "complete"
            elif meta.get("status") in {"running", "failed", "interrupted"}:
                state = "failed" if meta["status"] == "failed" else ("running" if meta["status"] == "running" else "partial")
            elif summary:
                state = "partial"
            else:
                state = "pending"
            totals[state] += 1
            if state in {"running", "failed", "partial"}:
                print(
                    f"{state.upper():8s} {protocol:5s} {run_id:17s} "
                    f"task={meta.get('last_recorded_task', meta.get('completed_tasks', 0))}/{expected} "
                    f"epoch={meta.get('last_recorded_epoch', '-')} reason={meta.get('failure_reason', '')}"
                )
    print(" ".join(f"{key}={value}" for key, value in totals.items()) + " total=108")


if __name__ == "__main__":
    main()
