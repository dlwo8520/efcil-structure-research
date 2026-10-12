"""Protocol-aware aggregate exports (C0 is the baseline)."""

from __future__ import annotations

import json
from pathlib import Path

from utils.experiment_io import read_csv, write_csv


def _read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _final_task_row(run_dir):
    rows = [row for row in read_csv(Path(run_dir) / "task_summary.csv") if row.get("eval_stage") == "task_final"]
    return max(rows, key=lambda row: int(row["task_id"])) if rows else None


def export_protocol(output_root, protocol):
    protocol_root = Path(output_root) / protocol
    metrics = []
    if not protocol_root.is_dir():
        return metrics
    for run_dir in sorted(protocol_root.glob("*/seed_*")):
        summary = _read_json(run_dir / "summary.json", {})
        config = _read_json(run_dir / "resolved_config.json", {})
        final = _final_task_row(run_dir)
        if summary.get("status") != "complete" or final is None:
            continue
        metrics.append(
            {
                "protocol": protocol,
                "dataset": config.get("source_dataset"),
                "num_tasks": config.get("expected_tasks"),
                "run_id": summary["experiment_id"],
                "method": config.get("method"),
                "P": config.get("top_p"),
                "alpha": config.get("alpha"),
                "lambda_clad": config.get("lambda_clad"),
                "seed": int(summary["seed"]),
                "Inc-Acc": summary["Inc-Acc"],
                "Last-Acc": summary["Last-Acc"],
                "Forgetting": summary["Forgetting"],
                "Old": final.get("old_accuracy"),
                "New": final.get("new_accuracy"),
                "completed_tasks": summary["completed_tasks"],
                "total_training_seconds": summary.get("total_training_seconds"),
                "status": summary["status"],
            }
        )
    metrics.sort(key=lambda row: (row["run_id"], row["seed"]))
    baseline = {
        int(row["seed"]): row for row in metrics if row["run_id"] == "C0"
    }
    for row in metrics:
        control = baseline.get(int(row["seed"]))
        for field in ("Inc-Acc", "Last-Acc", "Forgetting", "Old", "New"):
            left, right = row.get(field), control.get(field) if control else None
            row[f"delta_{field}"] = (
                float(left) - float(right) if left not in (None, "") and right not in (None, "") else None
            )
        row["guardrail_pass"] = (
            row["delta_Old"] is not None
            and row["delta_New"] is not None
            and row["delta_Forgetting"] is not None
            and row["delta_Old"] >= -0.5
            and row["delta_New"] >= -0.5
            and row["delta_Forgetting"] <= 0.5
        )
    write_csv(protocol_root / "metrics_per_seed.csv", metrics)
    return metrics


def export_all(output_root, protocols):
    rows = []
    for protocol in protocols:
        rows.extend(export_protocol(output_root, protocol))
    write_csv(Path(output_root) / "metrics_per_seed.csv", rows)
    return rows
