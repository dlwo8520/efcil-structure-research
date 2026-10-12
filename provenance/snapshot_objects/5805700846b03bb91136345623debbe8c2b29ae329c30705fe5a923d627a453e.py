#!/usr/bin/env python3
"""Print task/epoch progress for C0--C4 and smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--dataset", default="imageneta")
    return parser.parse_args()


def read_json(path):
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def last_epoch(path):
    if not path.is_file():
        return "-"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("stage") == "feature_training"]
    if not rows:
        return "-"
    row = max(rows, key=lambda item: (int(item["task_id"]), int(item["epoch"])))
    return f"T{row['task_id']}:E{row['epoch']}"


def main():
    cli = parse_args()
    dataset_root = Path(cli.output_root).expanduser().resolve() / cli.dataset
    print(f"{'METHOD':<10} {'SEED':<6} {'STATUS':<10} {'TASKS':<7} {'LAST':<9} UPDATED")
    for run_dir in sorted(dataset_root.glob("*/seed_*")):
        meta = read_json(run_dir / "run_metadata.json")
        print(
            f"{run_dir.parent.name:<10} {run_dir.name.removeprefix('seed_'):<6} "
            f"{meta.get('status', 'unknown'):<10} "
            f"{str(meta.get('completed_tasks', 0)) + '/5':<7} "
            f"{last_epoch(run_dir / 'epoch_metrics.csv'):<9} "
            f"{meta.get('updated_at_utc', '-')}"
        )


if __name__ == "__main__":
    main()
