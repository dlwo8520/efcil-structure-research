"""Result recording for the official PCDS / PCDS+SGDS comparison."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


class StructureRecorder(object):
    def __init__(self, args):
        self.args = dict(args)
        method = str(args.get("method_label", "pcds_structure"))
        self.output_dir = (
            Path(args["output_root"])
            / str(args.get("experiment_name", "cars196_10task"))
            / "seed_{}".format(args["seed"])
            / method
        )
        if (self.output_dir / "COMPLETED.json").exists():
            raise RuntimeError("Completed output already exists: {}".format(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.task_rows = []
        self.class_rows = []
        self.matrix = []
        self.ca_rows = []
        self.rank_rows = []
        self.semantic_rows = []
        self.semantic_task_rows = []
        self.activation_rows = []
        self.class_activation_rows = []
        self.increments = []
        self.class_order = []
        self.initial_trainable_parameters = 0
        self.total_parameters = 0
        write_json(self.output_dir / "config.json", self.args)

    def initialize(self, data_manager, model):
        self.increments = [int(value) for value in data_manager._increments]
        self.class_order = [int(value) for value in data_manager._class_order]
        trainable = sum(
            parameter.numel() for parameter in model._network.parameters()
            if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model._network.parameters())
        self.initial_trainable_parameters = int(trainable)
        self.total_parameters = int(total)
        protocol = {
            "dataset": self.args["dataset"],
            "data_root": self.args.get("data_root"),
            "num_classes": len(self.class_order),
            "task_sizes": self.increments,
            "num_tasks": len(self.increments),
            "class_order": self.class_order,
            "class_order_seed": self.args.get("class_order_seed", self.args["seed"]),
            "run_seed": self.args["seed"],
            "single_head_all_seen_classes": True,
            "raw_replay": False,
            "use_sgds": bool(self.args.get("use_sgds", False)),
            "sgds_target_layer": self.args.get("sgds_target_layer"),
            "trainable_parameters_registered": trainable,
            "total_parameters_before_classifier": total,
            "official_learner": "models.CDS_adapter.Learner",
            "upstream_commit": "a0db2168663982a4cad24981231721a3518ddf29",
        }
        write_json(self.output_dir / "protocol.json", protocol)

    @staticmethod
    def accuracy_parts(y_pred, y_true, new_begin):
        predicted = y_pred[:, 0]
        correct = predicted == y_true
        old_mask = y_true < new_begin
        new_mask = ~old_mask
        return {
            "seen_accuracy": float(correct.mean() * 100.0),
            "old_accuracy": (
                float(correct[old_mask].mean() * 100.0)
                if old_mask.any() else float("nan")
            ),
            "new_accuracy": float(correct[new_mask].mean() * 100.0),
        }

    def add_ca(self, row):
        self.ca_rows.append(dict(row))

    def add_rank(self, rows):
        self.rank_rows.extend(dict(row) for row in rows)

    def add_sgds(self, semantic_rows, semantic_task_row, activation_row, class_rows):
        self.semantic_rows.extend(dict(row) for row in semantic_rows)
        self.semantic_task_rows.append(dict(semantic_task_row))
        self.activation_rows.append(dict(activation_row))
        self.class_activation_rows.extend(dict(row) for row in class_rows)

    def record_evaluation(self, task_id, y_pred, y_true):
        task_id = int(task_id)
        trained_sizes = self.increments[:task_id + 1]
        per_task = []
        offset = 0
        predicted = y_pred[:, 0]
        for size in trained_sizes:
            mask = (y_true >= offset) & (y_true < offset + size)
            per_task.append(float((predicted[mask] == y_true[mask]).mean() * 100.0))
            offset += size
        self.matrix.append(per_task)
        seen_accuracy = float((predicted == y_true).mean() * 100.0)
        new_begin = sum(self.increments[:task_id])
        parts = self.accuracy_parts(y_pred, y_true, new_begin)
        forgetting_values = []
        for old_task in range(task_id):
            best = max(
                [self.matrix[row][old_task] for row in range(old_task, task_id)]
                + [per_task[old_task]]
            )
            forgetting_values.append(best - per_task[old_task])
        average_forgetting = (
            float(np.mean(forgetting_values)) if forgetting_values else 0.0
        )
        self.task_rows.append({
            "task_id": task_id,
            "seen_classes": sum(trained_sizes),
            "seen_accuracy": seen_accuracy,
            "old_accuracy": parts["old_accuracy"],
            "new_accuracy": parts["new_accuracy"],
            "average_forgetting": average_forgetting,
            "average_incremental_accuracy": float(np.mean([
                row["seen_accuracy"] for row in self.task_rows
            ] + [seen_accuracy])),
        })
        for class_id in range(sum(trained_sizes)):
            mask = y_true == class_id
            self.class_rows.append({
                "trained_through_task": task_id,
                "class_id": class_id,
                "accuracy": float((predicted[mask] == y_true[mask]).mean() * 100.0),
                "samples": int(mask.sum()),
            })
        self.flush()

    def flush(self):
        write_csv(self.output_dir / "task_metrics.csv", self.task_rows)
        write_csv(self.output_dir / "class_accuracy.csv", self.class_rows)
        matrix_rows = []
        for task_id, values in enumerate(self.matrix):
            row = {"trained_through_task": task_id}
            row.update({"task_{}".format(index): value for index, value in enumerate(values)})
            row["seen_average"] = self.task_rows[task_id]["seen_accuracy"]
            matrix_rows.append(row)
        write_csv(self.output_dir / "task_accuracy_matrix.csv", matrix_rows)
        write_csv(self.output_dir / "classifier_adaptation.csv", self.ca_rows)
        write_csv(self.output_dir / "effective_rank.csv", self.rank_rows)
        write_csv(self.output_dir / "sgds_semantic_strategy.csv", self.semantic_rows)
        write_csv(self.output_dir / "sgds_semantic_tasks.csv", self.semantic_task_rows)
        write_csv(self.output_dir / "sgds_activation.csv", self.activation_rows)
        write_csv(self.output_dir / "sgds_class_activation.csv", self.class_activation_rows)

    def finalize(self, model=None):
        if len(self.task_rows) != len(self.increments):
            raise RuntimeError(
                "Cannot complete: recorded {} of {} tasks".format(
                    len(self.task_rows), len(self.increments)
                )
            )
        final = self.task_rows[-1]
        peak_allocated = 0
        peak_reserved = 0
        if torch.cuda.is_available():
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
        final_trainable = self.initial_trainable_parameters
        final_total = self.total_parameters
        if model is not None:
            final_trainable = sum(
                parameter.numel() for parameter in model._network.parameters()
                if parameter.requires_grad
            )
            final_total = sum(parameter.numel() for parameter in model._network.parameters())
        summary = {
            "method": self.args.get("method_label", "pcds_structure"),
            "tasks_completed": len(self.task_rows),
            "average_incremental_accuracy": final["average_incremental_accuracy"],
            "final_accuracy": final["seen_accuracy"],
            "final_old_accuracy": final["old_accuracy"],
            "final_new_accuracy": final["new_accuracy"],
            "average_forgetting": final["average_forgetting"],
            "old_new_gap": final["old_accuracy"] - final["new_accuracy"],
            "peak_cuda_allocated_gb": peak_allocated / (1024 ** 3),
            "peak_cuda_reserved_gb": peak_reserved / (1024 ** 3),
            "elapsed_seconds": time.time() - self.started,
            "initial_trainable_parameters_before_classifier": self.initial_trainable_parameters,
            "final_trainable_parameters": int(final_trainable),
            "final_total_parameters": int(final_total),
            "optimizer_skipped_steps": 0,
            "nonfinite_steps": 0,
        }
        write_json(self.output_dir / "summary.json", summary)
        completed = dict(summary)
        completed.update({"status": "COMPLETE", "pid": os.getpid()})
        write_json(self.output_dir / "COMPLETED.json", completed)
        return summary
