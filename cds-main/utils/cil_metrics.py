"""Structured metrics and result serialization for the CDSP diagnostic."""

import csv
import json
import math
import os

import numpy as np


def class_accuracy_rows(y_pred, y_true, task_index, classifier):
    rows = []
    for class_id in sorted(np.unique(y_true).tolist()):
        mask = y_true == class_id
        rows.append({
            "task": int(task_index),
            "classifier": classifier,
            "class_id": int(class_id),
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(y_pred[mask] == y_true[mask]) * 100.0),
        })
    return rows


def task_accuracy_row(y_pred, y_true, task_boundaries):
    row = []
    start = 0
    for end in task_boundaries:
        mask = np.logical_and(y_true >= start, y_true < end)
        row.append(float(np.mean(y_pred[mask] == y_true[mask]) * 100.0) if mask.any() else None)
        start = end
    return row


def forgetting_from_matrix(matrix):
    if len(matrix) <= 1:
        return 0.0
    current_task = len(matrix) - 1
    forgetting = []
    for old_task in range(current_task):
        history = [row[old_task] for row in matrix[:current_task] if len(row) > old_task]
        if history and len(matrix[-1]) > old_task:
            forgetting.append(max(history) - matrix[-1][old_task])
    return float(np.mean(forgetting)) if forgetting else 0.0


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


class StructuredResultWriter:
    def __init__(self, root, dataset, method, seed):
        self.directory = os.path.join(root, "cdsp", str(dataset), str(method), "seed_{}".format(seed))
        os.makedirs(self.directory, exist_ok=True)
        self.task_metrics = []
        self.task_accuracy_matrix = []
        self.class_accuracy = []
        self.subspace_stats = []
        self.prototype_drift = []

    @property
    def checkpoint_directory(self):
        path = os.path.join(self.directory, "checkpoints")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _write_csv(path, rows, fieldnames=None):
        if fieldnames is None:
            fieldnames = list(rows[0].keys()) if rows else []
        with open(path, "w", newline="", encoding="utf-8") as handle:
            if not fieldnames:
                return
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def flush(self, config, primary_classifier):
        matrix_width = max((len(row) for row in self.task_accuracy_matrix), default=0)
        matrix_rows = []
        for task_index, values in enumerate(self.task_accuracy_matrix):
            row = {"trained_through_task": task_index}
            for column in range(matrix_width):
                row["eval_task_{}".format(column)] = values[column] if column < len(values) else ""
            matrix_rows.append(row)

        self._write_csv(os.path.join(self.directory, "task_metrics.csv"), self.task_metrics)
        self._write_csv(os.path.join(self.directory, "task_accuracy_matrix.csv"), matrix_rows)
        self._write_csv(os.path.join(self.directory, "class_accuracy.csv"), self.class_accuracy)
        self._write_csv(os.path.join(self.directory, "subspace_stats.csv"), self.subspace_stats)
        self._write_csv(os.path.join(self.directory, "prototype_drift.csv"), self.prototype_drift)

        summary = {
            "method": config.get("method", config.get("subspace_mode", "none")),
            "dataset": config.get("dataset"),
            "seed": config.get("seed"),
            "primary_classifier": primary_classifier,
            "config": config,
            "task_metrics": self.task_metrics,
            "task_accuracy_matrix": self.task_accuracy_matrix,
            "final_accuracy": self.task_metrics[-1]["primary_top1"] if self.task_metrics else None,
            "average_incremental_accuracy": (
                float(np.mean([row["primary_top1"] for row in self.task_metrics]))
                if self.task_metrics else None
            ),
            "forgetting": forgetting_from_matrix(self.task_accuracy_matrix),
        }
        with open(os.path.join(self.directory, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
