from __future__ import annotations

import gc
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from scipy.stats import kendalltau, spearmanr

from classifier_diagnostic.pipeline import _edge_fingerprint, write_csv, write_json
from oracle_audit.oracle_head import checkpoint_classifier_weights
from oracle_audit.runtime import birth_task, torch_load


@dataclass
class BoundaryRiskOptions:
    protocol: str
    t4_checkpoint: Path
    t5_checkpoint: Path
    t4_feature_cache: Path
    t5_feature_cache: Path
    output_dir: Path
    device: torch.device
    seed: int = 1993
    ranking_ks: tuple[int, ...] = (10, 20, 50, 100)
    random_trials: int = 1000
    risk_epsilon: float = 1e-12
    region_shrinkage: float = 0.1
    region_epsilon: float = 1e-5


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _distribution(values: np.ndarray, prefix: str = "") -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            f"{prefix}{key}": float("nan")
            for key in ("mean", "median", "q05", "q10", "q25", "negative_fraction")
        }
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}q05": float(np.quantile(values, 0.05)),
        f"{prefix}q10": float(np.quantile(values, 0.10)),
        f"{prefix}q25": float(np.quantile(values, 0.25)),
        f"{prefix}negative_fraction": float(np.mean(values < 0.0)),
    }


def _pair_type(i: int, j: int, known_before: int) -> str:
    return ("old" if i < known_before else "new") + "-" + (
        "old" if j < known_before else "new"
    )


def _transition_type(true: int, pred_t4: int, pred_t5: int) -> str:
    if pred_t4 == true and pred_t5 == true:
        return "correct_to_correct"
    if pred_t4 == true and pred_t5 != true:
        return "correct_to_wrong_B"
    if pred_t4 != true and pred_t5 == true:
        return "wrong_B_to_correct"
    if pred_t4 == pred_t5:
        return "wrong_B_to_wrong_B"
    return "wrong_C_to_wrong_B"


def _fixed_scores(features: np.ndarray, normalized_weights: np.ndarray) -> np.ndarray:
    return 20.0 * (_normalize(features) @ normalized_weights.T)


def boundary_sign_equivalence(
    raw_features: np.ndarray,
    normalized_weights: np.ndarray,
    labels: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    raw_projection = np.asarray(raw_features, dtype=np.float32) @ normalized_weights.T
    norms = np.maximum(np.linalg.norm(raw_features, axis=1), 1e-12)
    rows = np.arange(labels.size)
    raw_margin = raw_projection[rows, labels, None] - raw_projection
    cosine_margin = 20.0 * raw_margin / norms[:, None]
    valid = np.ones(raw_margin.shape, dtype=bool)
    valid[rows, labels] = False
    away_from_zero = valid & (np.abs(raw_margin) > tolerance) & (
        np.abs(cosine_margin) > tolerance
    )
    mismatch = int(np.sum(np.signbit(raw_margin[away_from_zero]) != np.signbit(cosine_margin[away_from_zero])))
    expected = 20.0 * raw_margin / norms[:, None]
    max_difference = float(np.max(np.abs(cosine_margin[valid] - expected[valid])))
    return {
        "status": "PASS" if mismatch == 0 else "FAIL",
        "definition": "a_ij = normalize(w_i) - normalize(w_j)",
        "checked_sample_pair_count": int(valid.sum()),
        "away_from_zero_count": int(away_from_zero.sum()),
        "pairwise_sign_mismatch_count": mismatch,
        "zero_tolerance": tolerance,
        "max_scaled_margin_identity_difference": max_difference,
    }


def _sample_transition_rows(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    pred_t4: np.ndarray,
    pred_t5: np.ndarray,
    partition: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for index, label in enumerate(labels.tolist()):
        transition = _transition_type(
            int(label), int(pred_t4[index]), int(pred_t5[index])
        )
        if transition == "wrong_C_to_wrong_B":
            source_view = "wrong_B_to_wrong_C"
        else:
            source_view = transition
        rows.append({
            "sample_id": str(sample_ids[index]),
            "true_class": int(label),
            "birth_task_true": birth_task(int(label), partition),
            "pred_t4": int(pred_t4[index]),
            "pred_t5": int(pred_t5[index]),
            "t4_correct": bool(pred_t4[index] == label),
            "t5_correct": bool(pred_t5[index] == label),
            "prediction_transition_type": transition,
            "prediction_transition_type_from_t4_source_view": source_view,
            "fixed_readout": "T5 Structure cosine head applied to T4 and T5 features",
        })
    return rows


def _pair_actual_rows(
    labels: np.ndarray,
    scores_t4: np.ndarray,
    scores_t5: np.ndarray,
    pred_t4: np.ndarray,
    pred_t5: np.ndarray,
    train_labels: np.ndarray,
    partition: list[int],
    known_before: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    classes = scores_t5.shape[1]
    total_t5_errors = max(int(np.sum(pred_t5 != labels)), 1)
    metrics: list[dict[str, Any]] = []
    margins: list[dict[str, Any]] = []
    for i in range(classes):
        mask = labels == i
        n = int(mask.sum())
        p4 = pred_t4[mask]
        p5 = pred_t5[mask]
        s4 = scores_t4[mask]
        s5 = scores_t5[mask]
        h4 = s4[:, i, None] - s4
        h5 = s5[:, i, None] - s5
        delta = h5 - h4
        c4 = np.bincount(p4, minlength=classes)
        c5 = np.bincount(p5, minlength=classes)
        boundary4 = np.sum(h4 < 0.0, axis=0)
        boundary5 = np.sum(h5 < 0.0, axis=0)
        correct4 = p4 == i
        correct5 = p5 == i
        for j in range(classes):
            if i == j:
                continue
            new_correct_to_j = int(np.sum(correct4 & (p5 == j)))
            old_wrong_to_j = int(np.sum((~correct4) & (p4 != j) & (p5 == j)))
            recovered_from_j = int(np.sum((p4 == j) & correct5))
            wrong_j_to_other = int(np.sum((p4 == j) & (~correct5) & (p5 != j)))
            wrong_j_persistent = int(np.sum((p4 == j) & (p5 == j)))
            c_t4, c_t5 = c4[j] / n, c5[j] / n
            b_t4, b_t5 = boundary4[j] / n, boundary5[j] / n
            metrics.append(
                {
                    "true_class": i,
                    "competitor_class": j,
                    "pair_type": _pair_type(i, j, known_before),
                    "class_i_task": birth_task(i, partition),
                    "class_j_task": birth_task(j, partition),
                    "n_true_samples": n,
                    "train_sample_count_i": int(np.sum(train_labels == i)),
                    "train_sample_count_j": int(np.sum(train_labels == j)),
                    "C_t4": float(c_t4),
                    "C_t5": float(c_t5),
                    "delta_C": float(c_t5 - c_t4),
                    "B_t4": float(b_t4),
                    "B_t5": float(b_t5),
                    "delta_B": float(b_t5 - b_t4),
                    "boundary_violation_count_t4": int(boundary4[j]),
                    "boundary_violation_count_t5": int(boundary5[j]),
                    "positive_boundary_degradation_count": int(max(boundary5[j] - boundary4[j], 0)),
                    "error_count_t4": int(c4[j]),
                    "error_count_t5": int(c5[j]),
                    "t5_total_error_contribution": float(c5[j] / total_t5_errors),
                    "new_correct_to_j_count": new_correct_to_j,
                    "new_correct_to_j_rate": float(new_correct_to_j / n),
                    "old_wrong_to_j_count": old_wrong_to_j,
                    "recovered_from_j_count": recovered_from_j,
                    "wrong_j_to_other_wrong_count": wrong_j_to_other,
                    "wrong_j_persistent_count": wrong_j_persistent,
                    "score_definition": "fixed T5 Structure cosine head",
                }
            )
            stats4 = _distribution(h4[:, j], "t4_")
            stats5 = _distribution(h5[:, j], "t5_")
            delta_stats = _distribution(delta[:, j], "delta_")
            margins.append(
                {
                    "true_class": i,
                    "competitor_class": j,
                    "pair_type": _pair_type(i, j, known_before),
                    "n_true_samples": n,
                    **stats4,
                    **stats5,
                    **delta_stats,
                    "delta_margin_mean": stats5["t5_mean"] - stats4["t4_mean"],
                    "delta_margin_median": stats5["t5_median"] - stats4["t4_median"],
                    "delta_margin_q05": stats5["t5_q05"] - stats4["t4_q05"],
                    "delta_margin_q10": stats5["t5_q10"] - stats4["t4_q10"],
                    "delta_margin_q25": stats5["t5_q25"] - stats4["t4_q25"],
                    "delta_negative_fraction": (
                        stats5["t5_negative_fraction"] - stats4["t4_negative_fraction"]
                    ),
                }
            )
    margin_map = {
        (row["true_class"], row["competitor_class"]): row for row in margins
    }
    current_candidates = [
        row for row in metrics if row["n_true_samples"] >= 5 and row["error_count_t5"] >= 2
    ]
    current_candidates.sort(
        key=lambda row: (
            -row["error_count_t5"], -row["C_t5"], -row["t5_total_error_contribution"],
            row["true_class"], row["competitor_class"],
        )
    )
    for rank, row in enumerate(current_candidates, start=1):
        row["current_hard_rank"] = rank
    degraded_candidates = []
    for row in metrics:
        margin = margin_map[(row["true_class"], row["competitor_class"])]
        if (
            row["n_true_samples"] >= 5
            and row["delta_C"] > 0
            and row["delta_B"] > 0
            and margin["delta_margin_q05"] < 0
            and row["new_correct_to_j_count"] >= 1
        ):
            degraded_candidates.append(row)
    degraded_candidates.sort(
        key=lambda row: (
            -row["new_correct_to_j_count"], -row["delta_C"], -row["delta_B"],
            margin_map[(row["true_class"], row["competitor_class"])]["delta_margin_q05"],
            row["true_class"], row["competitor_class"],
        )
    )
    for rank, row in enumerate(degraded_candidates, start=1):
        row["newly_degraded_rank"] = rank
    return metrics, margins


def _quadratic_variance(
    directions: np.ndarray,
    covariance: torch.Tensor | np.ndarray,
    device: torch.device,
) -> np.ndarray:
    direction_tensor = torch.as_tensor(directions, dtype=torch.float32, device=device)
    covariance_tensor = torch.as_tensor(covariance, dtype=torch.float32, device=device)
    values = torch.sum((direction_tensor @ covariance_tensor) * direction_tensor, dim=1)
    return values.clamp_min(0.0).detach().cpu().numpy().astype(np.float64)


def _risk_rows(
    options: BoundaryRiskOptions,
    raw_train: np.ndarray,
    train_labels: np.ndarray,
    normalized_weights: np.ndarray,
    stored_means: np.ndarray,
    stored_covariances: list[torch.Tensor],
    jitter: float,
    metrics: list[dict[str, Any]],
    margins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    classes = normalized_weights.shape[0]
    oracle_means = np.stack(
        [raw_train[train_labels == class_id].mean(axis=0) for class_id in range(classes)]
    ).astype(np.float64)
    stored_means = np.asarray(stored_means[:classes], dtype=np.float64)
    if stored_means.shape != oracle_means.shape:
        raise RuntimeError(
            f"stored/oracle mean coordinate mismatch: {stored_means.shape} vs {oracle_means.shape}"
        )
    oracle_mean_unit = _normalize(oracle_means)
    stored_mean_unit = _normalize(stored_means)
    metric_map = {(row["true_class"], row["competitor_class"]): row for row in metrics}
    margin_map = {(row["true_class"], row["competitor_class"]): row for row in margins}
    rows: list[dict[str, Any]] = []
    for i in range(classes):
        directions = normalized_weights[i][None, :] - normalized_weights
        norm2 = np.sum(np.square(directions.astype(np.float64)), axis=1)
        actual = np.asarray(raw_train[train_labels == i], dtype=np.float64)
        projected = actual @ directions.astype(np.float64).T
        if actual.shape[0] >= 2:
            oracle_variance = np.var(projected, axis=0, ddof=1) + jitter * norm2
        else:
            oracle_variance = jitter * norm2
        oracle_margin = directions.astype(np.float64) @ oracle_means[i]
        stored_margin = directions.astype(np.float64) @ stored_means[i]
        stored_variance = _quadratic_variance(
            directions, stored_covariances[i], options.device
        )
        gamma_oracle = oracle_margin / np.sqrt(oracle_variance + options.risk_epsilon)
        gamma_stored = stored_margin / np.sqrt(stored_variance + options.risk_epsilon)
        for j in range(classes):
            if i == j:
                continue
            actual_row = metric_map[(i, j)]
            margin_row = margin_map[(i, j)]
            rows.append(
                {
                    "true_class": i,
                    "competitor_class": j,
                    "pair_type": actual_row["pair_type"],
                    "n_true_samples": actual_row["n_true_samples"],
                    "gamma_oracle": float(gamma_oracle[j]),
                    "gamma_stored": float(gamma_stored[j]),
                    "q_oracle": float(ndtr(-gamma_oracle[j])),
                    "q_stored": float(ndtr(-gamma_stored[j])),
                    "boundary_mean_oracle": float(oracle_margin[j]),
                    "boundary_mean_stored": float(stored_margin[j]),
                    "boundary_variance_oracle": float(oracle_variance[j]),
                    "boundary_variance_stored": float(stored_variance[j]),
                    "C_t4": actual_row["C_t4"],
                    "C_t5": actual_row["C_t5"],
                    "delta_C": actual_row["delta_C"],
                    "B_t4": actual_row["B_t4"],
                    "B_t5": actual_row["B_t5"],
                    "delta_B": actual_row["delta_B"],
                    "boundary_violation_count_t4": actual_row["boundary_violation_count_t4"],
                    "boundary_violation_count_t5": actual_row["boundary_violation_count_t5"],
                    "positive_boundary_degradation_count": actual_row["positive_boundary_degradation_count"],
                    "correct_to_wrong_j": actual_row["new_correct_to_j_count"],
                    "correct_to_wrong_j_rate": actual_row["new_correct_to_j_rate"],
                    "old_wrong_to_j": actual_row["old_wrong_to_j_count"],
                    "error_count_t5": actual_row["error_count_t5"],
                    "margin_q05_t4": margin_row["t4_q05"],
                    "margin_q05_t5": margin_row["t5_q05"],
                    "delta_margin_q05": margin_row["delta_margin_q05"],
                    "prototype_similarity": float(stored_mean_unit[i] @ stored_mean_unit[j]),
                    "oracle_prototype_similarity_audit": float(
                        oracle_mean_unit[i] @ oracle_mean_unit[j]
                    ),
                    "class_i_stored_oracle_mean_cosine": float(
                        stored_mean_unit[i] @ oracle_mean_unit[i]
                    ),
                    "current_hard_rank": actual_row.get("current_hard_rank"),
                    "newly_degraded_rank": actual_row.get("newly_degraded_rank"),
                    "gamma_interpretation": "smaller_is_higher_Gaussian_boundary_risk_not_multiclass_probability",
                }
            )
    return rows


TARGETS = {
    "A_B_T5": ("B_t5", "boundary_violation_count_t5"),
    "B_POSITIVE_DELTA_B": ("positive_delta_B", "positive_boundary_degradation_count"),
    "C_NEW_ACTUAL_ERROR": ("correct_to_wrong_j_rate", "correct_to_wrong_j"),
}


def _ranking_analysis(
    rows: list[dict[str, Any]],
    ks: tuple[int, ...],
    seed: int,
    random_trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair_count = len(rows)
    effective_ks = sorted({min(k, pair_count) for k in ks if min(k, pair_count) > 0})
    target_values = {
        "A_B_T5": np.asarray([row["B_t5"] for row in rows], dtype=np.float64),
        "B_POSITIVE_DELTA_B": np.maximum(
            np.asarray([row["delta_B"] for row in rows], dtype=np.float64), 0.0
        ),
        "C_NEW_ACTUAL_ERROR": np.asarray(
            [row["correct_to_wrong_j_rate"] for row in rows], dtype=np.float64
        ),
    }
    masses = {
        "A_B_T5": np.asarray(
            [row["boundary_violation_count_t5"] for row in rows], dtype=np.float64
        ),
        "B_POSITIVE_DELTA_B": np.asarray(
            [row["positive_boundary_degradation_count"] for row in rows], dtype=np.float64
        ),
        "C_NEW_ACTUAL_ERROR": np.asarray(
            [row["correct_to_wrong_j"] for row in rows], dtype=np.float64
        ),
    }
    ranking_scores = {
        "PROTOTYPE_SIMILARITY": np.asarray(
            [row["prototype_similarity"] for row in rows], dtype=np.float64
        ),
        "ORACLE_GAMMA": -np.asarray([row["gamma_oracle"] for row in rows], dtype=np.float64),
        "STORED_GAMMA": -np.asarray([row["gamma_stored"] for row in rows], dtype=np.float64),
    }
    indices = np.arange(pair_count)

    def descending(values: np.ndarray) -> np.ndarray:
        return np.lexsort((indices, -values))

    ranking_orders = {name: descending(values) for name, values in ranking_scores.items()}
    actual_orders = {name: descending(values) for name, values in target_values.items()}
    rng = np.random.RandomState(seed)
    random_orders = [rng.permutation(pair_count) for _ in range(random_trials)]
    recall_rows: list[dict[str, Any]] = []
    captured_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        total_mass = float(masses[target].sum())
        for k in effective_ks:
            actual_set = set(actual_orders[target][:k].tolist())
            random_recall = np.mean(
                [len(actual_set.intersection(order[:k].tolist())) / k for order in random_orders]
            )
            random_capture = np.mean(
                [masses[target][order[:k]].sum() / total_mass for order in random_orders]
            ) if total_mass > 0 else float("nan")
            recall_rows.append(
                {
                    "target": target,
                    "ranking": "RANDOM_EXPECTED",
                    "K": k,
                    "recall_at_k": float(random_recall),
                    "pair_count": pair_count,
                    "random_trials": random_trials,
                }
            )
            captured_rows.append(
                {
                    "target": target,
                    "ranking": "RANDOM_EXPECTED",
                    "K": k,
                    "captured_mass_fraction": float(random_capture),
                    "selected_mass": float(random_capture * total_mass) if total_mass > 0 else float("nan"),
                    "total_positive_mass": total_mass,
                    "random_trials": random_trials,
                }
            )
            for ranking, order in ranking_orders.items():
                selected = order[:k]
                recall_rows.append(
                    {
                        "target": target,
                        "ranking": ranking,
                        "K": k,
                        "recall_at_k": len(actual_set.intersection(selected.tolist())) / k,
                        "pair_count": pair_count,
                        "random_trials": None,
                    }
                )
                selected_mass = float(masses[target][selected].sum())
                captured_rows.append(
                    {
                        "target": target,
                        "ranking": ranking,
                        "K": k,
                        "captured_mass_fraction": selected_mass / total_mass if total_mass > 0 else float("nan"),
                        "selected_mass": selected_mass,
                        "total_positive_mass": total_mass,
                        "random_trials": None,
                    }
                )
    correlation_rows: list[dict[str, Any]] = []
    random_scores = rng.uniform(size=pair_count)
    for target, values in target_values.items():
        for ranking, scores in {"RANDOM_SEEDED": random_scores, **ranking_scores}.items():
            if np.unique(scores).size < 2 or np.unique(values).size < 2:
                spearman_stat = spearman_pvalue = float("nan")
                kendall_stat = kendall_pvalue = float("nan")
            else:
                spearman = spearmanr(scores, values, nan_policy="omit")
                kendall = kendalltau(scores, values, nan_policy="omit")
                spearman_stat, spearman_pvalue = float(spearman.statistic), float(spearman.pvalue)
                kendall_stat, kendall_pvalue = float(kendall.statistic), float(kendall.pvalue)
            correlation_rows.append(
                {
                    "target": target,
                    "ranking": ranking,
                    "spearman": spearman_stat,
                    "spearman_pvalue": spearman_pvalue,
                    "kendall": kendall_stat,
                    "kendall_pvalue": kendall_pvalue,
                    "pair_count": pair_count,
                    "unique_target_values": int(np.unique(values).size),
                    "tie_heavy_warning": bool(np.unique(values).size < max(10, pair_count // 100)),
                }
            )
    summary_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        for ranking in ("RANDOM_EXPECTED", "PROTOTYPE_SIMILARITY", "ORACLE_GAMMA", "STORED_GAMMA"):
            recall_selected = [
                row for row in recall_rows if row["target"] == target and row["ranking"] == ranking
            ]
            mass_selected = [
                row for row in captured_rows if row["target"] == target and row["ranking"] == ranking
            ]
            captured_values = np.asarray(
                [row["captured_mass_fraction"] for row in mass_selected], dtype=np.float64
            )
            finite_captured = captured_values[np.isfinite(captured_values)]
            corr = next(
                (
                    row for row in correlation_rows
                    if row["target"] == target
                    and row["ranking"] == ("RANDOM_SEEDED" if ranking == "RANDOM_EXPECTED" else ranking)
                ),
                {},
            )
            summary_rows.append(
                {
                    "target": target,
                    "ranking": ranking,
                    "mean_recall_across_k": float(np.mean([row["recall_at_k"] for row in recall_selected])),
                    "mean_captured_mass_across_k": (
                        float(finite_captured.mean()) if finite_captured.size else float("nan")
                    ),
                    "recall_at_max_k": recall_selected[-1]["recall_at_k"],
                    "captured_mass_at_max_k": mass_selected[-1]["captured_mass_fraction"],
                    "spearman": corr.get("spearman"),
                    "kendall": corr.get("kendall"),
                }
            )
    return summary_rows, recall_rows, captured_rows, correlation_rows


def _region_support(
    options: BoundaryRiskOptions,
    raw_train: np.ndarray,
    train_labels: np.ndarray,
    raw_test: np.ndarray,
    test_labels: np.ndarray,
    risk_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = sorted(
        [row for row in risk_rows if row.get("newly_degraded_rank")],
        key=lambda row: row["newly_degraded_rank"],
    )[:10]
    selected_keys = {(row["true_class"], row["competitor_class"]) for row in selected}
    for row in sorted(
        [item for item in risk_rows if item.get("current_hard_rank")],
        key=lambda item: item["current_hard_rank"],
    ):
        selected_keys.add((row["true_class"], row["competitor_class"]))
        if len(selected_keys) >= 20:
            break
    train = _normalize(raw_train).astype(np.float64)
    test = _normalize(raw_test).astype(np.float64)
    classes = np.unique(train_labels)
    means = np.stack([train[train_labels == class_id].mean(axis=0) for class_id in classes])
    centered = np.concatenate(
        [train[train_labels == class_id] - means[int(class_id)] for class_id in classes]
    )
    covariance = centered.T @ centered / max(centered.shape[0] - len(classes), 1)
    scale = float(np.trace(covariance) / covariance.shape[0])
    shrunk = (
        (1.0 - options.region_shrinkage) * covariance
        + options.region_shrinkage * scale * np.eye(covariance.shape[0])
        + options.region_epsilon * np.eye(covariance.shape[0])
    )
    factor = np.linalg.cholesky(0.5 * (shrunk + shrunk.T))

    def distances(values: np.ndarray, mean: np.ndarray) -> np.ndarray:
        solved = solve_triangular(factor, (values - mean).T, lower=True, check_finite=False)
        return np.sum(np.square(solved), axis=0)

    rows = []
    for i, j in sorted(selected_keys):
        values = test[test_labels == i]
        d_i = distances(values, means[i])
        d_j = distances(values, means[j])
        gap = d_j - d_i
        rows.append(
            {
                "true_class": i,
                "competitor_class": j,
                "num_samples": int(values.shape[0]),
                "d2_true_mean": float(d_i.mean()),
                "d2_competitor_mean": float(d_j.mean()),
                "support_gap_mean": float(gap.mean()),
                "support_gap_median": float(np.median(gap)),
                "support_gap_q05": float(np.quantile(gap, 0.05)),
                "fraction_closer_to_competitor": float(np.mean(gap < 0.0)),
                "covariance": "shared normalized within-class shrinkage",
                "shrinkage_lambda": options.region_shrinkage,
                "epsilon": options.region_epsilon,
                "role": "optional explanation only; not used for readiness",
            }
        )
    return rows


def _plots(
    output_dir: Path,
    risk_rows: list[dict[str, Any]],
    recall_rows: list[dict[str, Any]],
    captured_rows: list[dict[str, Any]],
    scores_t4: np.ndarray,
    scores_t5: np.ndarray,
    labels: np.ndarray,
) -> None:
    for directory in (
        "same_sample_margin", "risk_vs_deltaB", "recall_at_k",
        "captured_mass_at_k", "stored_vs_oracle",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    selected = sorted(
        [row for row in risk_rows if row.get("newly_degraded_rank")],
        key=lambda row: row["newly_degraded_rank"],
    )[:6]
    if not selected:
        selected = sorted(
            [row for row in risk_rows if row.get("current_hard_rank")],
            key=lambda row: row["current_hard_rank"],
        )[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, row in zip(axes.flat, selected):
        i, j = row["true_class"], row["competitor_class"]
        mask = labels == i
        h4 = scores_t4[mask, i] - scores_t4[mask, j]
        h5 = scores_t5[mask, i] - scores_t5[mask, j]
        ax.scatter(h4, h5, s=12, alpha=0.65)
        low = min(float(h4.min()), float(h5.min()))
        high = max(float(h4.max()), float(h5.max()))
        ax.plot([low, high], [low, high], color="black", linewidth=0.8)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.axvline(0, color="grey", linewidth=0.6)
        ax.set_title(f"{i}→{j}")
        ax.set_xlabel("H(T4)")
        ax.set_ylabel("H(T5)")
    for ax in axes.flat[len(selected):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "same_sample_margin" / "top_pair_trajectory.png", dpi=160)
    plt.close(fig)

    delta_b = np.asarray([row["delta_B"] for row in risk_rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, field, label in zip(
        axes,
        ("gamma_oracle", "gamma_stored", "prototype_similarity"),
        ("oracle gamma", "stored gamma", "stored prototype cosine"),
    ):
        ax.scatter([row[field] for row in risk_rows], delta_b, s=5, alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set(xlabel=label, ylabel="delta B")
    fig.tight_layout()
    fig.savefig(output_dir / "risk_vs_deltaB" / "risk_scores_vs_deltaB.png", dpi=160)
    plt.close(fig)

    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 5))
        for ranking in ("RANDOM_EXPECTED", "PROTOTYPE_SIMILARITY", "ORACLE_GAMMA", "STORED_GAMMA"):
            values = [row for row in recall_rows if row["target"] == target and row["ranking"] == ranking]
            ax.plot([row["K"] for row in values], [row["recall_at_k"] for row in values], marker="o", label=ranking)
        ax.set(xlabel="K", ylabel="Recall@K", title=target)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "recall_at_k" / f"{target}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for ranking in ("RANDOM_EXPECTED", "PROTOTYPE_SIMILARITY", "ORACLE_GAMMA", "STORED_GAMMA"):
            values = [row for row in captured_rows if row["target"] == target and row["ranking"] == ranking]
            ax.plot([row["K"] for row in values], [row["captured_mass_fraction"] for row in values], marker="o", label=ranking)
        ax.set(xlabel="K", ylabel="captured positive mass", title=target)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "captured_mass_at_k" / f"{target}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    oracle = np.asarray([row["gamma_oracle"] for row in risk_rows])
    stored = np.asarray([row["gamma_stored"] for row in risk_rows])
    ax.scatter(oracle, stored, s=5, alpha=0.25)
    ax.set(xlabel="oracle gamma", ylabel="stored gamma", title="Stored vs Oracle boundary risk")
    fig.tight_layout()
    fig.savefig(output_dir / "stored_vs_oracle" / "gamma_scatter.png", dpi=160)
    plt.close(fig)


def run_protocol(options: BoundaryRiskOptions) -> dict[str, Any]:
    started = time.time()

    def progress(message: str) -> None:
        print(f"[{options.protocol} +{time.time() - started:.1f}s] {message}", flush=True)

    options.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints_before = {
        "t4": _edge_fingerprint(options.t4_checkpoint),
        "t5": _edge_fingerprint(options.t5_checkpoint),
    }
    progress("loading fixed checkpoints and aligned T4/T5 feature caches")
    t4_checkpoint = torch_load(options.t4_checkpoint)
    t5_checkpoint = torch_load(options.t5_checkpoint)
    if int(t4_checkpoint["cur_task"]) + 1 != 4 or int(t5_checkpoint["cur_task"]) + 1 != 5:
        raise RuntimeError("exact T4/T5 task-end checkpoints are required")
    partition = list(map(int, t5_checkpoint["task_sizes"]))
    classes = int(t5_checkpoint["total_classes"])
    known_before = classes - partition[-1]
    if float(t5_checkpoint["config"].get("scale", 20.0)) != 20.0:
        raise RuntimeError("boundary equivalence requires the recorded scale-20 Structure head")
    t4 = np.load(options.t4_feature_cache, allow_pickle=False)
    t5 = np.load(options.t5_feature_cache, allow_pickle=False)
    train_t4 = np.asarray(t4["train_features"], dtype=np.float32)
    test_t4 = np.asarray(t4["test_features"], dtype=np.float32)
    train_t5 = np.asarray(t5["train_features"], dtype=np.float32)
    test_t5 = np.asarray(t5["test_features"], dtype=np.float32)
    train_labels = np.asarray(t5["train_labels"], dtype=np.int64)
    test_labels = np.asarray(t5["test_labels"], dtype=np.int64)
    if not np.array_equal(train_labels, t4["train_labels"]) or not np.array_equal(test_labels, t4["test_labels"]):
        raise RuntimeError("T4/T5 label mapping mismatch")
    if train_t4.shape != train_t5.shape or test_t4.shape != test_t5.shape:
        raise RuntimeError("T4/T5 feature coordinate or sample count mismatch")
    train_ids = np.asarray(t4["train_sample_ids"])
    test_ids = np.asarray(t4["test_sample_ids"])
    if np.unique(train_ids).size != train_ids.size or np.unique(test_ids).size != test_ids.size:
        raise RuntimeError("canonical original sample IDs must be unique")
    if set(np.unique(train_labels)) != set(range(classes)) or set(np.unique(test_labels)) != set(range(classes)):
        raise RuntimeError("T4/T5 common class set is incomplete")
    normalized_weights = _normalize(
        checkpoint_classifier_weights(t5_checkpoint).numpy()[:classes]
    )
    stored_means = np.asarray(t5_checkpoint["class_means"][:classes], dtype=np.float64)
    stored_covariances = list(t5_checkpoint["class_covs"][:classes])
    jitter = float(t5_checkpoint["config"].get("stats_cov_jitter", 1e-3))
    del t4_checkpoint
    gc.collect()

    progress("computing same-sample transitions and directional C/B/margin metrics")
    scores_t4 = _fixed_scores(test_t4, normalized_weights)
    scores_t5 = _fixed_scores(test_t5, normalized_weights)
    pred_t4 = np.argmax(scores_t4, axis=1)
    pred_t5 = np.argmax(scores_t5, axis=1)
    equivalence = boundary_sign_equivalence(test_t5, normalized_weights, test_labels)
    equivalence.update(
        {
            "protocol": options.protocol,
            "feature_space": "raw T5 Structure sfeatures with positive norm normalization",
            "classifier": "frozen T5 Structure cosine head",
        }
    )
    write_json(options.output_dir / "boundary_direction_equivalence_test.json", equivalence)
    if equivalence["status"] != "PASS":
        raise RuntimeError(f"boundary sign equivalence failed: {equivalence}")
    sample_rows = _sample_transition_rows(test_ids, test_labels, pred_t4, pred_t5, partition)
    metrics, margins = _pair_actual_rows(
        test_labels, scores_t4, scores_t5, pred_t4, pred_t5,
        train_labels, partition, known_before,
    )
    write_csv(options.output_dir / "same_sample_prediction_transition.csv", sample_rows)
    write_csv(options.output_dir / "pair_confusion_boundary_metrics.csv", metrics)
    write_csv(options.output_dir / "same_sample_pair_margin_transition.csv", margins)

    progress("computing oracle/stored gamma in the current raw feature coordinate")
    risk_rows = _risk_rows(
        options, train_t5, train_labels, normalized_weights,
        stored_means, stored_covariances, jitter, metrics, margins,
    )
    write_csv(options.output_dir / "stored_oracle_boundary_risk.csv", risk_rows)
    del t5_checkpoint, stored_covariances
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress("evaluating random/prototype/oracle/stored rankings")
    summary, recall, captured, correlations = _ranking_analysis(
        risk_rows, options.ranking_ks, options.seed + 7100, options.random_trials
    )
    for row in summary + recall + captured + correlations:
        row["protocol"] = options.protocol
    write_csv(options.output_dir / "risk_ranking_summary.csv", summary)
    write_csv(options.output_dir / "risk_recall_at_k.csv", recall)
    write_csv(options.output_dir / "captured_degradation_mass.csv", captured)
    write_csv(options.output_dir / "rank_correlation_summary.csv", correlations)

    progress("running optional shared-covariance region explanation and plots")
    region_rows = _region_support(
        options, train_t5, train_labels, test_t5, test_labels, risk_rows
    )
    write_csv(options.output_dir / "optional_region_support_diagnostic.csv", region_rows)
    _plots(options.output_dir, risk_rows, recall, captured, scores_t4, scores_t5, test_labels)

    fingerprints_after = {
        "t4": _edge_fingerprint(options.t4_checkpoint),
        "t5": _edge_fingerprint(options.t5_checkpoint),
    }
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("checkpoint fingerprint changed during diagnostic")
    transition_counts = Counter(row["prediction_transition_type"] for row in sample_rows)
    newly_degraded = sum(row.get("newly_degraded_rank") is not None for row in metrics)
    manifest = {
        "status": "complete",
        "protocol": options.protocol,
        "seed": options.seed,
        "transition": "T4_to_T5",
        "fixed_readout": "T5 Structure cosine head on both T4/T5 frozen features",
        "t4_checkpoint": str(options.t4_checkpoint),
        "t5_checkpoint": str(options.t5_checkpoint),
        "t4_feature_cache": str(options.t4_feature_cache),
        "t5_feature_cache": str(options.t5_feature_cache),
        "checkpoint_fingerprints_before": fingerprints_before,
        "checkpoint_fingerprints_after": fingerprints_after,
        "checkpoints_unchanged": True,
        "sample_ids_unique": True,
        "t4_t5_label_mapping_identical": True,
        "common_class_count": classes,
        "train_sample_count": int(train_labels.size),
        "test_sample_count": int(test_labels.size),
        "pair_count": len(risk_rows),
        "newly_degraded_pair_count": newly_degraded,
        "prediction_transition_counts": dict(transition_counts),
        "ranking_ks": list(options.ranking_ks),
        "random_trials": options.random_trials,
        "boundary_equivalence": equivalence,
        "diagnostic_only": True,
        "model_parameters_updated": False,
        "test_used_for_method_training": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_dir / "run_manifest.json", manifest)
    progress("protocol complete")
    return manifest
