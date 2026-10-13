from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

import numpy as np


EPS = 1e-12


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    return array[np.isfinite(array)]


def summary(values: Iterable[float], prefix: str = "") -> dict[str, float | int]:
    array = finite(values)
    if not array.size:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "median", "q10", "q25", "q75", "q90")} | {f"{prefix}count": 0}
    return {
        f"{prefix}count": int(array.size),
        f"{prefix}mean": float(array.mean()),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}q10": float(np.quantile(array, 0.10)),
        f"{prefix}q25": float(np.quantile(array, 0.25)),
        f"{prefix}q75": float(np.quantile(array, 0.75)),
        f"{prefix}q90": float(np.quantile(array, 0.90)),
    }


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...], values: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    output: list[dict[str, Any]] = []
    for identity, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, identity))
        result["class_count"] = len(members)
        for value in values:
            result.update(summary((float(row[value]) for row in members), f"{value}_"))
        output.append(result)
    return output


def prototype_rows_from_snapshots(
    protocol: str,
    snapshots: list[dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tracking: list[dict[str, Any]] = []
    temporal: list[dict[str, Any]] = []
    interclass: list[dict[str, Any]] = []
    previous_oracle: dict[int, np.ndarray] = {}
    previous_stored: dict[int, np.ndarray] = {}
    for task, snapshot in enumerate(snapshots, start=1):
        class_ids = np.asarray(snapshot["class_ids"], dtype=np.int64)
        oracle = np.asarray(snapshot["oracle_means"], dtype=np.float64)
        stored = np.asarray(snapshot["structure_means"], dtype=np.float64)
        births = np.asarray(snapshot["birth_tasks"], dtype=np.int64)
        oracle_unit, stored_unit = normalize(oracle), normalize(stored)
        known_before = int(np.sum(births < task))
        for position, class_id in enumerate(class_ids.tolist()):
            birth, age = int(births[position]), task - int(births[position])
            delta = stored[position] - oracle[position]
            tracking.append({
                "protocol": protocol,
                "task": task,
                "class_id": class_id,
                "birth_task": birth,
                "class_age": age,
                "is_old": birth < task,
                "tracking_similarity": float(stored_unit[position] @ oracle_unit[position]),
                "tracking_relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(oracle[position]), EPS)),
                "stored_norm": float(np.linalg.norm(stored[position])),
                "oracle_norm": float(np.linalg.norm(oracle[position])),
                "definition": "checkpoint.class_means versus full-train oracle mean",
            })
            if class_id in previous_oracle:
                temporal.append({
                    "protocol": protocol,
                    "previous_task": task - 1,
                    "task": task,
                    "class_id": class_id,
                    "birth_task": birth,
                    "class_age": age,
                    "oracle_temporal_similarity": cosine(previous_oracle[class_id], oracle[position]),
                    "stored_temporal_similarity": cosine(previous_stored[class_id], stored[position]),
                    "oracle_drift_norm": float(np.linalg.norm(oracle[position] - previous_oracle[class_id])),
                    "stored_drift_norm": float(np.linalg.norm(stored[position] - previous_stored[class_id])),
                })
        for definition, means in (("oracle", oracle_unit), ("stored", stored_unit)):
            matrix = means @ means.T
            np.fill_diagonal(matrix, -np.inf)
            for position, class_id in enumerate(class_ids.tolist()):
                rivals = matrix[position]
                order = np.argsort(-rivals)
                top5 = rivals[order[: min(5, rivals.size - 1)]]
                top10 = rivals[order[: min(10, rivals.size - 1)]]
                rival = int(class_ids[order[0]]) if order.size else -1
                birth, age = int(births[position]), task - int(births[position])
                interclass.append({
                    "protocol": protocol,
                    "task": task,
                    "class_id": class_id,
                    "birth_task": birth,
                    "class_age": age,
                    "is_old": birth < task,
                    "definition": definition,
                    "nearest_rival_class": rival,
                    "nearest_rival_group": (
                        "old" if rival >= 0 and rival < known_before else "new"
                    ),
                    "nearest_rival_similarity": float(rivals[order[0]]) if order.size else float("nan"),
                    "top5_rival_mean_similarity": float(np.mean(top5)) if top5.size else float("nan"),
                    "top10_rival_mean_similarity": float(np.mean(top10)) if top10.size else float("nan"),
                })
        previous_oracle = {int(c): oracle[p].copy() for p, c in enumerate(class_ids)}
        previous_stored = {int(c): stored[p].copy() for p, c in enumerate(class_ids)}
    return tracking, temporal, interclass


def directional_rows(
    protocol: str,
    task: int,
    labels: np.ndarray,
    predictions: np.ndarray,
    known_before: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classes = int(max(labels.max(initial=0), predictions.max(initial=0)) + 1)
    counts = Counter(zip(labels.tolist(), predictions.tolist()))
    errors: list[dict[str, Any]] = []
    for (true, pred), count in counts.items():
        if true == pred:
            continue
        support = int(np.sum(labels == true))
        errors.append({
            "protocol": protocol,
            "task": task,
            "true_class": int(true),
            "predicted_class": int(pred),
            "pair_type": ("old" if true < known_before else "new") + "-" + ("old" if pred < known_before else "new"),
            "error_count": int(count),
            "conditional_confusion": float(count / max(support, 1)),
            "true_class_support": support,
        })
    errors.sort(key=lambda row: (-row["error_count"], row["true_class"], row["predicted_class"]))
    total = sum(int(row["error_count"]) for row in errors)
    values = np.asarray([row["error_count"] for row in errors], dtype=np.float64)
    probabilities = values / max(values.sum(), EPS)
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, EPS))))
    normalized_entropy = entropy / math.log(max(values.size, 2))
    sorted_values = np.sort(values)
    gini = (
        float((2 * np.arange(1, values.size + 1) - values.size - 1) @ sorted_values / (values.size * values.sum()))
        if values.size and values.sum() else 0.0
    )
    concentration = {
        "protocol": protocol,
        "task": task,
        "total_errors": total,
        "nonzero_directional_pairs": len(errors),
        "all_possible_directional_pairs": classes * max(classes - 1, 0),
        "top10_error_share": float(values[:10].sum() / max(total, 1)),
        "top20_error_share": float(values[:20].sum() / max(total, 1)),
        "top50_error_share": float(values[:50].sum() / max(total, 1)),
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "gini_like": gini,
    }
    for pair_type in ("old-old", "old-new", "new-old", "new-new"):
        count = sum(int(row["error_count"]) for row in errors if row["pair_type"] == pair_type)
        concentration[f"{pair_type.replace('-', '_')}_error_count"] = count
        concentration[f"{pair_type.replace('-', '_')}_error_share"] = count / max(total, 1)
    return errors, concentration
