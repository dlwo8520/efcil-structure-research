from __future__ import annotations

import numpy as np

from oracle_audit.metrics import EPS, gaussian_shape_metrics, mahalanobis_squared_diagonal


def competition_row(
    base: dict,
    condition: str,
    metrics: dict,
) -> dict:
    return {
        **base,
        "condition": condition,
        "single_head_new_acc": metrics["new_accuracy"],
        "new_only_acc": metrics["new_only_accuracy"],
        "new_only_minus_single_head": metrics["new_only_accuracy"] - metrics["new_accuracy"],
        "old_single_head_acc": metrics["old_accuracy"],
        "old_only_acc": metrics["old_only_accuracy"],
        "old_only_minus_single_head": metrics["old_only_accuracy"] - metrics["old_accuracy"],
        "new_to_old_count": metrics["new_to_old_count"],
        "new_to_wrong_new_count": metrics["new_to_new_wrong_count"],
        "new_to_correct_new_count": metrics["new_correct_count"],
        "new_to_old_rate": metrics["new_to_old_rate"],
        "new_to_wrong_new_rate": metrics["new_to_new_wrong_rate"],
        "old_to_new_count": metrics["old_to_new_count"],
        "old_to_wrong_old_count": metrics["old_to_old_wrong_count"],
        "old_to_correct_old_count": metrics["old_correct_count"],
        "old_to_new_rate": metrics["old_to_new_rate"],
        "old_to_wrong_old_rate": metrics["old_to_old_wrong_rate"],
    }


def margin_rows(
    base: dict,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    known_before: int,
) -> list[dict]:
    mapping = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
    old_columns = np.flatnonzero(class_ids < known_before)
    if not old_columns.size:
        return []
    rows = []
    for sample_index, label in enumerate(labels.tolist()):
        if int(label) < known_before:
            continue
        true_column = mapping[int(label)]
        old_offset = int(np.argmax(scores[sample_index, old_columns]))
        old_column = int(old_columns[old_offset])
        true_score = float(scores[sample_index, true_column])
        old_score = float(scores[sample_index, old_column])
        predicted_column = int(np.argmax(scores[sample_index]))
        predicted_class = int(class_ids[predicted_column])
        if predicted_class == int(label):
            outcome = "new_to_correct_new"
        elif predicted_class < known_before:
            outcome = "new_to_old"
        else:
            outcome = "new_to_wrong_new"
        rows.append(
            {
                **base,
                "condition": condition,
                "sample_index": sample_index,
                "true_class_id": int(label),
                "strongest_old_class_id": int(class_ids[old_column]),
                "predicted_class_id": predicted_class,
                "true_new_score": true_score,
                "max_old_score": old_score,
                "new_vs_old_margin": true_score - old_score,
                "outcome": outcome,
            }
        )
    return rows


def boundary_direction_rows(
    base: dict,
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    known_before: int,
    weights: np.ndarray,
    structure_means: np.ndarray,
    oracle_means: np.ndarray,
) -> list[dict]:
    normalized_weights = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), EPS)
    mapping = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
    rows = []
    for class_id in class_ids[class_ids < known_before]:
        class_id = int(class_id)
        column = mapping[class_id]
        mask = labels == class_id
        if not mask.any():
            continue
        mean_scores = scores[mask].mean(axis=0)
        mean_scores[column] = -np.inf
        competitor_column = int(np.argmax(mean_scores))
        competitor_id = int(class_ids[competitor_column])
        delta = structure_means[column] - oracle_means[column]
        signed = float(
            np.dot(normalized_weights[column] - normalized_weights[competitor_column], delta)
        )
        relative_l2 = float(np.linalg.norm(delta) / (np.linalg.norm(oracle_means[column]) + EPS))
        predictions = np.argmax(scores[mask], axis=1)
        rows.append(
            {
                **base,
                "class_id": class_id,
                "competitor_class_id": competitor_id,
                "competitor_is_new": competitor_id >= known_before,
                "competitor_selection": "highest_mean_logit_on_true_class_test_samples",
                "weight_definition": "row-normalized cosine-classifier weights; scale omitted",
                "prototype_relative_l2": relative_l2,
                "boundary_direction_error": signed,
                "absolute_boundary_direction_error": abs(signed),
                "class_accuracy": float(np.mean(predictions == column) * 100.0),
                "confusion_to_competitor_rate": float(np.mean(predictions == competitor_column)),
            }
        )
    return rows


def gaussian_rows(
    base: dict,
    class_stats: dict,
    class_ids: np.ndarray,
) -> list[dict]:
    rows = []
    dimension = next(iter(class_stats.values())).features.shape[1] if class_stats else 0
    for class_id in class_ids:
        stats = class_stats[int(class_id)]
        d2 = mahalanobis_squared_diagonal(
            stats.features, stats.mean, stats.covariance
        )
        rows.append(
            {
                **base,
                "class_id": int(class_id),
                "sample_count": int(stats.features.shape[0]),
                **gaussian_shape_metrics(stats.features),
                "oracle_diag_d2_mean": float(np.mean(d2)),
                "oracle_diag_d2_variance": float(np.var(d2)),
                "gaussian_expected_d2_mean": int(dimension),
                "d2_mean_over_dimension": float(np.mean(d2) / max(dimension, 1)),
            }
        )
    return rows
