from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / "structure_prototype_oracle_audit"
sys.path.insert(0, str(AUDIT))

from oracle_audit.oracle_head import checkpoint_classifier_weights, cosine_scores, evaluate_scores  # noqa: E402
from oracle_audit.runtime import torch_load  # noqa: E402
from ten_task_revalidation.pipeline import _checkpoint_dir  # noqa: E402


def inputs(protocol: str, seed: int = 1993):
    if protocol not in {"IA10", "CAR10"}:
        raise ValueError(protocol)
    checkpoint_path = _checkpoint_dir(protocol, seed) / "task_10.pt"
    cache_path = AUDIT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / "model_10_on_task_10.npz"
    checkpoint = torch_load(checkpoint_path)
    with np.load(cache_path, allow_pickle=False) as values:
        cache = {name: values[name].copy() for name in ("train_features", "train_labels", "test_features", "test_labels", "class_ids")}
    classes = int(checkpoint["total_classes"])
    assert classes == len(cache["class_ids"])
    assert np.array_equal(cache["class_ids"], np.arange(classes))
    known = classes - int(checkpoint["task_sizes"][-1])
    return checkpoint_path, checkpoint, cache, known


def metrics(scores: np.ndarray, labels: np.ndarray, classes: int, known: int) -> dict:
    result = evaluate_scores(scores, labels, np.arange(classes), known)
    predictions = scores.argmax(axis=1)
    per_class = {}
    for class_id in range(classes):
        mask = labels == class_id
        per_class[str(class_id)] = float(np.mean(predictions[mask] == labels[mask]) * 100)
    result["class_macro_accuracy"] = float(np.mean(list(per_class.values())))
    result["per_class_accuracy"] = per_class
    return result


def fixed_scores(features: np.ndarray, checkpoint: dict) -> np.ndarray:
    classes = int(checkpoint["total_classes"])
    weights = checkpoint_classifier_weights(checkpoint).numpy()[:classes]
    return cosine_scores(features, weights, float(checkpoint["config"].get("scale", 1.0)))


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
