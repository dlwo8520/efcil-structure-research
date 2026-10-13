"""Structured result serialization for PTM-guided CIL experiments."""

import csv
import json
import math
import os

import numpy as np

from utils.cil_metrics import forgetting_from_matrix


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


class PTMGuidedResultWriter:
    def __init__(self, root, dataset, task_count, method, seed):
        self.directory = os.path.join(
            root,
            "ptm_guided",
            str(dataset),
            "{}_tasks".format(int(task_count)),
            str(method),
            "seed_{}".format(seed),
        )
        os.makedirs(self.directory, exist_ok=True)
        self.task_metrics = []
        self.task_accuracy_matrix = []
        self.class_accuracy = []
        self.ptm_geometry_drift = []
        self.anchor_drift = []
        self.loss_components = []

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

    def flush(self, config):
        width = max((len(row) for row in self.task_accuracy_matrix), default=0)
        matrix_rows = []
        for task_index, values in enumerate(self.task_accuracy_matrix):
            row = {"trained_through_task": task_index}
            for column in range(width):
                row["eval_task_{}".format(column)] = (
                    values[column] if column < len(values) else ""
                )
            matrix_rows.append(row)

        self._write_csv(os.path.join(self.directory, "task_metrics.csv"), self.task_metrics)
        self._write_csv(os.path.join(self.directory, "task_accuracy_matrix.csv"), matrix_rows)
        self._write_csv(os.path.join(self.directory, "class_accuracy.csv"), self.class_accuracy)
        self._write_csv(
            os.path.join(self.directory, "ptm_geometry_drift.csv"), self.ptm_geometry_drift
        )
        self._write_csv(os.path.join(self.directory, "anchor_drift.csv"), self.anchor_drift)
        self._write_csv(os.path.join(self.directory, "loss_components.csv"), self.loss_components)

        last_accuracy = self.task_metrics[-1]["last_accuracy"] if self.task_metrics else None
        incremental_accuracy = (
            float(np.mean([row["last_accuracy"] for row in self.task_metrics]))
            if self.task_metrics else None
        )
        if config.get("smoke_test"):
            status = "SMOKE_TEST"
        elif len(self.task_metrics) == int(config.get("task_count", 0)):
            status = "COMPLETE"
        else:
            status = "IN_PROGRESS"
        summary = {
            "method": config.get("method"),
            "dataset": config.get("dataset"),
            "task_count": config.get("task_count"),
            "seed": config.get("seed"),
            "config": config,
            "task_metrics": self.task_metrics,
            "task_accuracy_matrix": self.task_accuracy_matrix,
            "last_accuracy": last_accuracy,
            "incremental_accuracy": incremental_accuracy,
            "forgetting": forgetting_from_matrix(self.task_accuracy_matrix),
            "status": status,
        }
        with open(os.path.join(self.directory, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
