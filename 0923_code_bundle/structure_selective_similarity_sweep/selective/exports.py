"""Protocol-aware aggregate exports with vanilla Structure as the baseline."""

from __future__ import annotations

import json
from pathlib import Path

from utils.experiment_io import read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]
IA5_STRUCTURE_DIR = (
    ROOT.parent
    / "structure_e3_followup"
    / "outputs"
    / "imageneta"
    / "B0_STRUCTURE"
    / "seed_1993"
)


def _read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _final_task_row(run_dir):
    rows = [row for row in read_csv(Path(run_dir) / "task_summary.csv") if row.get("eval_stage") == "task_final"]
    return max(rows, key=lambda row: int(row["task_id"])) if rows else None


def _metric_row(protocol, run_dir, *, run_id=None, method=None):
    summary = _read_json(Path(run_dir) / "summary.json", {})
    config = _read_json(Path(run_dir) / "resolved_config.json", {})
    final = _final_task_row(run_dir)
    if summary.get("status") != "complete" or final is None:
        return None
    return {
        "protocol": protocol,
        "dataset": config.get("source_dataset", "imageneta" if protocol == "IA5" else None),
        "num_tasks": config.get("expected_tasks", 5 if protocol == "IA5" else None),
        "run_id": run_id or summary["experiment_id"],
        "method": method or config.get("method"),
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


def export_protocol(output_root, protocol):
    protocol_root = Path(output_root) / protocol
    metrics = []
    if not protocol_root.is_dir():
        return metrics
    for run_dir in sorted(protocol_root.glob("*/seed_*")):
        row = _metric_row(protocol, run_dir)
        if row is not None:
            metrics.append(row)
    if protocol == "IA5" and not any(row["run_id"] == "Structure" for row in metrics):
        row = _metric_row(
            protocol,
            IA5_STRUCTURE_DIR,
            run_id="Structure",
            method="Structure",
        )
        if row is not None:
            metrics.append(row)
    metrics.sort(key=lambda row: (row["run_id"] != "Structure", row["run_id"], row["seed"]))
    baseline = {
        int(row["seed"]): row for row in metrics if row["run_id"] == "Structure"
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
