"""Post-run schema exports required by the similarity-aware analysis plan."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fieldnames=None):
    path = Path(path)
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _metric_row(run_dir):
    summary = _read_json(run_dir / "summary.json", {})
    task_rows = _read_csv(run_dir / "task_summary.csv")
    if summary.get("status") != "complete" or not task_rows:
        return None
    final = max(task_rows, key=lambda row: int(row["task_id"]))
    return {
        "method": summary["experiment_id"],
        "seed": int(summary["seed"]),
        "Inc-Acc": summary["Inc-Acc"],
        "Last-Acc": summary["Last-Acc"],
        "Forgetting": summary["Forgetting"],
        "Old": final.get("old_accuracy", ""),
        "New": final.get("new_accuracy", ""),
        "completed_tasks": summary["completed_tasks"],
        "status": summary["status"],
    }


def export_run(run_dir, dataset_root):
    run_dir = Path(run_dir)
    config = _read_json(run_dir / "resolved_config.json", {})
    summary = _read_json(run_dir / "summary.json", {})
    if summary.get("status") != "complete":
        return None
    method = str(summary["experiment_id"])
    seed = int(summary["seed"])
    exports = run_dir / "exports"

    task_rows = []
    for row in _read_csv(run_dir / "task_accuracy.csv"):
        if row.get("eval_stage") != "task_final":
            continue
        task_rows.append(
            {
                "method": method,
                "seed": seed,
                "train_task": int(row["trained_task"]),
                "eval_task": int(row["evaluated_task"]),
                "correct": int(row["num_correct"]),
                "total": int(row["num_samples"]),
                "accuracy": float(row["accuracy"]),
            }
        )
    _write_csv(exports / "task_accuracy_matrix.csv", task_rows)

    baseline_path = Path(dataset_root) / "B0" / f"seed_{seed}" / "class_accuracy.csv"
    baseline = {
        (int(row["trained_task"]), int(row["mapped_class_id"])): float(row["accuracy"])
        for row in _read_csv(baseline_path)
        if row.get("eval_stage") == "task_final"
    }
    increment = int(config.get("increment", 40))
    initial = int(config.get("init_cls", increment))

    def birth_task(class_id):
        if class_id < initial:
            return 1
        return 2 + (class_id - initial) // increment

    class_rows = []
    for row in _read_csv(run_dir / "class_accuracy.csv"):
        if row.get("eval_stage") != "task_final":
            continue
        class_id = int(row["mapped_class_id"])
        trained_task = int(row["trained_task"])
        accuracy = float(row["accuracy"])
        baseline_accuracy = baseline.get((trained_task, class_id))
        birth = birth_task(class_id)
        class_rows.append(
            {
                "method": method,
                "seed": seed,
                "train_task": trained_task,
                "class_id": class_id,
                "original_class_id": row.get("original_class_id", ""),
                "class_name": row.get("class_name", ""),
                "birth_task": birth,
                "old_or_new": "new" if birth == trained_task else "old",
                "correct": int(row["num_correct"]),
                "total": int(row["num_samples"]),
                "accuracy": accuracy,
                "delta_vs_B0": "" if baseline_accuracy is None else accuracy - baseline_accuracy,
            }
        )
    _write_csv(exports / "class_accuracy.csv", class_rows)
    metric = _metric_row(run_dir)
    _write_csv(exports / "metrics_per_seed.csv", [metric] if metric else [])
    return metric


def export_output_tree(dataset_root):
    """Refresh all completed exports so later methods can receive B0 deltas."""
    dataset_root = Path(dataset_root)
    metrics = []
    if not dataset_root.is_dir():
        return metrics
    for run_dir in sorted(dataset_root.glob("*/seed_*")):
        metric = export_run(run_dir, dataset_root)
        if metric is not None:
            metrics.append(metric)
    metrics.sort(key=lambda row: (row["method"], row["seed"]))
    _write_csv(
        dataset_root / "metrics_per_seed.csv",
        metrics,
        fieldnames=[
            "method",
            "seed",
            "Inc-Acc",
            "Last-Acc",
            "Forgetting",
            "Old",
            "New",
            "completed_tasks",
            "status",
        ],
    )
    return metrics

