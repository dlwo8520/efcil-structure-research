from __future__ import annotations

import gc
import hashlib
import json
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

from classifier_diagnostic.pipeline import _edge_fingerprint, read_csv, write_csv, write_json
from classifier_procedure.pipeline import (
    _make_epoch_permutations,
    _replacement_balanced_bank,
    _train_cosine,
)
from oracle_audit.oracle_head import checkpoint_classifier_weights
from oracle_audit.runtime import (
    birth_task,
    build_data_manager,
    extract_split,
    load_frozen_learner,
    release_learner,
    torch_load,
)
from representation_relation.knn import KNNResult, _vote, cosine_knn, macro_accuracy, purity
from representation_relation.linear import (
    LinearResult,
    class_balanced_objective,
    train_linear_lbfgs,
)


@dataclass
class RelationOptions:
    protocol: str
    t4_checkpoint: Path
    t5_checkpoint: Path
    t5_feature_cache: Path
    output_dir: Path
    data_root: Path
    gpu_device: torch.device
    seed: int = 1993
    batch_size: int = 64
    num_workers: int = 4
    top_global_pairs: int = 20
    top_conditional_pairs: int = 20
    knn_candidates: tuple[int, ...] = (1, 3, 5, 9)
    linear_l2_values: tuple[float, ...] = (5e-4, 5e-5, 5e-6)


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def _hash_sample(value: Any) -> str:
    digest = hashlib.sha256()
    if isinstance(value, (str, Path, np.str_)):
        digest.update(str(value).encode("utf-8"))
    else:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()[:20]


def _dataset_identity(data_manager: Any, class_ids: np.ndarray, source: str):
    data, labels, _ = data_manager.get_dataset(
        class_ids, source=source, mode="test", ret_data=True
    )
    sample_ids = np.asarray(
        [f"{source}:{index:06d}:{_hash_sample(value)}" for index, value in enumerate(data)]
    )
    return sample_ids, np.asarray(labels, dtype=np.int64)


def _validation_indices(
    labels: np.ndarray,
    class_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[int]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    rows: list[dict[str, Any]] = []
    no_validation: list[int] = []
    for class_id in class_ids.tolist():
        indices = np.flatnonzero(labels == class_id)
        shuffled = indices[
            np.random.RandomState(seed + class_id * 997).permutation(indices.size)
        ]
        if indices.size >= 5:
            validation_count = min(
                max(1, int(math.floor(indices.size * 0.2))), indices.size - 2
            )
        else:
            validation_count = 0
            no_validation.append(class_id)
        validation_indices.extend(shuffled[:validation_count].tolist())
        train_indices.extend(shuffled[validation_count:].tolist())
        rows.append(
            {
                "row_type": "class",
                "class_id": class_id,
                "original_train_samples": int(indices.size),
                "diagnostic_train_samples": int(indices.size - validation_count),
                "validation_samples": int(validation_count),
                "split_before_replacement_sampling": True,
                "validation_excluded_from_diagnostic_training": True,
                "used_in_historical_backbone_training": True,
                "representation_unseen": False,
            }
        )
    return (
        np.asarray(sorted(train_indices), dtype=np.int64),
        np.asarray(sorted(validation_indices), dtype=np.int64),
        rows,
        no_validation,
    )


def _micro_macro(predictions: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    return (
        float(np.mean(predictions == labels) * 100.0),
        macro_accuracy(predictions, labels),
    )


def _accuracy_summary(
    method: str,
    predictions: np.ndarray,
    labels: np.ndarray,
    known_before: int,
    split: str,
    **extra: Any,
) -> dict[str, Any]:
    micro, macro = _micro_macro(predictions, labels)
    old = labels < known_before
    new = ~old
    return {
        "row_type": "summary",
        "split": split,
        "method": method,
        "num_samples": int(labels.size),
        "micro_accuracy": micro,
        "macro_accuracy": macro,
        "old_accuracy": float(np.mean(predictions[old] == labels[old]) * 100.0),
        "new_accuracy": float(np.mean(predictions[new] == labels[new]) * 100.0),
        **extra,
    }


def _class_macro_from_scores(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    predictions = np.argmax(scores, axis=1)
    return _micro_macro(predictions, labels)


def _converted_weights(weights: np.ndarray, scale: float) -> np.ndarray:
    return scale * weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)


def _linear_equivalence(
    normalized_test: np.ndarray,
    cosine_weights: np.ndarray,
    scale: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    converted = _converted_weights(cosine_weights, scale)
    cosine_logits = scale * (
        normalized_test
        @ (cosine_weights / np.maximum(np.linalg.norm(cosine_weights, axis=1, keepdims=True), 1e-12)).T
    )
    linear_logits = normalized_test @ converted.T
    difference = np.abs(cosine_logits - linear_logits)
    mismatch = int(
        np.sum(np.argmax(cosine_logits, axis=1) != np.argmax(linear_logits, axis=1))
    )
    record = {
        "status": "PASS" if mismatch == 0 and float(difference.max()) <= 1e-5 else "FAIL",
        "scale": scale,
        "max_absolute_logit_difference": float(difference.max()),
        "mean_absolute_logit_difference": float(difference.mean()),
        "prediction_mismatch_count": mismatch,
        "num_samples": int(normalized_test.shape[0]),
        "num_classes": int(cosine_weights.shape[0]),
        "tolerance": 1e-5,
        "conversion": "v_k = scale * w_k / ||w_k||",
    }
    return record, converted, cosine_logits


def _linear_sanity(
    options: RelationOptions,
    converted_weights: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[list[dict[str, Any]], LinearResult, float]:
    rows: list[dict[str, Any]] = []
    results: dict[float, LinearResult] = {}
    converted_train_scores = train_features @ converted_weights.T
    for l2 in options.linear_l2_values:
        converted_objective, converted_balanced_ce = class_balanced_objective(
            converted_train_scores, train_labels, converted_weights, l2
        )
        result = train_linear_lbfgs(
            initial_weights=converted_weights,
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            test_features=test_features,
            test_labels=test_labels,
            l2=l2,
            device=options.gpu_device,
            max_iter=200,
            tolerance_grad=1e-7,
        )
        results[l2] = result
        rows.append(
            {
                "protocol": options.protocol,
                "condition": "LINEAR_REGULARIZATION_SANITY",
                "l2": l2,
                "train_micro_accuracy": result.train_micro,
                "train_macro_accuracy": result.train_macro,
                "validation_micro_accuracy": result.validation_micro,
                "validation_macro_accuracy": result.validation_macro,
                "test_micro_accuracy": result.test_micro,
                "test_macro_accuracy": result.test_macro,
                "train_ce": result.train_ce,
                "class_balanced_train_ce": result.class_balanced_train_ce,
                "validation_ce": result.validation_ce,
                "test_ce": result.test_ce,
                "objective": result.objective,
                "converted_cosine_objective": converted_objective,
                "converted_cosine_class_balanced_ce": converted_balanced_ce,
                "weight_norm": result.weight_norm,
                "solver_closures": result.closure_count,
                "gradient_norm": result.gradient_norm,
                "converged_by_gradient": result.converged,
                "initialization": "converted Structure cosine solution",
                "selected_by_test": False,
            }
        )
    selected_l2 = max(
        options.linear_l2_values,
        key=lambda value: (
            results[value].validation_macro,
            results[value].validation_micro,
            value,
        ),
    )
    for row in rows:
        row["selected_by_validation"] = float(row["l2"]) == selected_l2
        row["selection_rule"] = "validation macro, then micro, then stronger L2"
    return rows, results[selected_l2], selected_l2


def _write_linear_audit(path: Path, selected_l2: float) -> None:
    lines = [
        "# Linear Probe Training Audit",
        "",
        "## 기존 procedure diagnostic 설정",
        "",
        "- 구현: `classifier_procedure/pipeline.py::_train_linear()`",
        "- objective: inverse-class-frequency weighted mean CE + `0.5 * lambda * sum(W^2)`",
        "- class weighting: sample weight `N / (C * n_y)`",
        "- regularization: L2, lambda `5e-4`",
        "- solver: PyTorch full-batch L-BFGS, strong-Wolfe line search",
        "- max_iter/max_eval: 100/200",
        "- tolerance_grad/tolerance_change: `1e-5` / `1e-9`",
        "- initialization: raw pre-CA cosine weight",
        "- bias: condition에 따라 없음/있음",
        "- feature: L2-normalized frozen `sfeatures`",
        "- split: replacement 이전 original image 단위 deterministic stratified split",
        "- sklearn `C`, `n_iter_`, convergence warning: 해당 없음(PyTorch LBFGS 사용)",
        "",
        "## 이번 sanity 설정",
        "",
        "- lambda: `5e-4`, `5e-5`, `5e-6` 세 개만 사용",
        "- max_iter/max_eval: 200/400",
        "- tolerance_grad/tolerance_change: `1e-7` / `1e-10`",
        "- initialization: exact converted Structure cosine solution",
        "- 선택: validation macro accuracy, micro accuracy, stronger L2 순",
        f"- 선택된 lambda: `{selected_l2:g}`",
        "- test는 regularization 선택에 사용하지 않음",
        "",
        "기존 lambda의 penalty는 scale-20 converted cosine weight에 대해 클 수 있으므로, 이전 linear 저성능을 representation 근거로 사용하기 전에 이번 sanity 결과를 확인한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_type(i: int, j: int, known_before: int) -> str:
    return ("old" if i < known_before else "new") + "-" + (
        "old" if j < known_before else "new"
    )


def _directional_confusions(
    predictions: np.ndarray,
    labels: np.ndarray,
    train_labels: np.ndarray,
    partition: list[int],
    known_before: int,
) -> list[dict[str, Any]]:
    error_mask = predictions != labels
    total_errors = int(error_mask.sum())
    pairs = Counter(zip(labels[error_mask].tolist(), predictions[error_mask].tolist()))
    rows = []
    for (true_class, predicted_class), count in pairs.items():
        class_total = int(np.sum(labels == true_class))
        class_errors = int(np.sum((labels == true_class) & error_mask))
        rows.append(
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confusion_count": count,
                "conditional_confusion": count / class_total,
                "total_error_of_class_i": class_errors,
                "pair_share_of_class_i_error": count / class_errors if class_errors else float("nan"),
                "pair_share_of_all_errors": count / total_errors if total_errors else float("nan"),
                "class_i_task": birth_task(true_class, partition),
                "class_j_task": birth_task(predicted_class, partition),
                "pair_type": _pair_type(true_class, predicted_class, known_before),
                "train_sample_count_i": int(np.sum(train_labels == true_class)),
                "train_sample_count_j": int(np.sum(train_labels == predicted_class)),
                "test_sample_count_i": class_total,
            }
        )
    return sorted(rows, key=lambda row: (-row["confusion_count"], row["true_class"], row["predicted_class"]))


def _focused_pairs(
    rows: list[dict[str, Any]],
    global_k: int,
    conditional_k: int,
) -> list[dict[str, Any]]:
    global_rows = sorted(
        rows,
        key=lambda row: (-row["pair_share_of_all_errors"], -row["confusion_count"], row["true_class"], row["predicted_class"]),
    )[:global_k]
    conditional_rows = sorted(
        rows,
        key=lambda row: (-row["conditional_confusion"], -row["confusion_count"], row["true_class"], row["predicted_class"]),
    )[:conditional_k]
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for source, values in (("global_top20", global_rows), ("conditional_top20", conditional_rows)):
        for rank, row in enumerate(values, start=1):
            key = (int(row["true_class"]), int(row["predicted_class"]))
            if key not in selected:
                selected[key] = {**row, "global_rank": None, "conditional_rank": None, "selection_sources": []}
            selected[key]["selection_sources"].append(source)
            selected[key]["global_rank" if source.startswith("global") else "conditional_rank"] = rank
    result = []
    for row in selected.values():
        row["selection_sources"] = ";".join(row["selection_sources"])
        result.append(row)
    return sorted(result, key=lambda row: (-row["confusion_count"], row["true_class"], row["predicted_class"]))


def _distribution(values: np.ndarray) -> dict[str, float]:
    if not values.size:
        return {key: float("nan") for key in (
            "mean", "median", "std", "negative_fraction", "q01", "q05", "q10", "q25", "q50"
        )}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "negative_fraction": float(np.mean(values < 0.0)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "q10": float(np.quantile(values, 0.10)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
    }


def _pair_margin_rows(
    focused: list[dict[str, Any]],
    split_scores: dict[str, np.ndarray],
    split_labels: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = []
    for pair in focused:
        i, j = int(pair["true_class"]), int(pair["predicted_class"])
        for split, scores in split_scores.items():
            mask = split_labels[split] == i
            margins = scores[mask, i] - scores[mask, j]
            rows.append(
                {
                    "true_class": i,
                    "competitor_class": j,
                    "pair_type": pair["pair_type"],
                    "split": split,
                    "num_samples": int(mask.sum()),
                    **_distribution(margins),
                }
            )
    return rows


def _sample_count_groups(train_labels: np.ndarray, class_ids: np.ndarray):
    counts = np.asarray([np.sum(train_labels == class_id) for class_id in class_ids])
    q25, q50, q75 = np.quantile(counts, [0.25, 0.50, 0.75])

    def group(class_id: int) -> str:
        value = counts[class_id]
        if value <= q25:
            return "very_low"
        if value <= q50:
            return "low"
        if value <= q75:
            return "medium"
        return "high"

    return group, {"q25": float(q25), "q50": float(q50), "q75": float(q75)}


def _knn_select(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    candidates: tuple[int, ...],
    device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    maximum = max(candidates)
    neighbors = cosine_knn(
        reference_features, reference_labels, validation_features, maximum, device
    )
    rows = []
    for k in candidates:
        predictions = _vote(
            neighbors.neighbor_labels[:, :k],
            1.0 - neighbors.neighbor_distances[:, :k],
        )
        micro, macro = _micro_macro(predictions, validation_labels)
        rows.append(
            {
                "row_type": "k_selection",
                "split": "validation",
                "method": f"KNN_K{k}",
                "k": k,
                "micro_accuracy": micro,
                "macro_accuracy": macro,
                "selected_by_test": False,
            }
        )
    selected = max(
        candidates,
        key=lambda k: (
            next(row["macro_accuracy"] for row in rows if row["k"] == k),
            next(row["micro_accuracy"] for row in rows if row["k"] == k),
            -k,
        ),
    )
    for row in rows:
        row["selected_by_validation"] = row["k"] == selected
    return selected, rows


def _knn_sample_rows(
    *,
    split: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    cosine_scores: np.ndarray,
    linear_scores: np.ndarray,
    knn: KNNResult,
    train_purity_by_class: dict[int, float],
) -> list[dict[str, Any]]:
    cosine_prediction = np.argmax(cosine_scores, axis=1)
    linear_prediction = np.argmax(linear_scores, axis=1)
    values = purity(knn.neighbor_labels, labels)
    rows = []
    for index, label in enumerate(labels.tolist()):
        competitor_scores = cosine_scores[index].copy()
        competitor_scores[label] = -np.inf
        strongest = int(np.argmax(competitor_scores))
        if not (cosine_prediction[index] == label) and knn.predictions[index] == label and values[index] >= 0.8:
            case = "B1_HEAD_MISSES_LOCAL_SUPPORT"
        elif (
            cosine_prediction[index] != label
            and knn.predictions[index] == cosine_prediction[index]
            and values[index] <= 0.4
        ):
            case = "B2_SHARED_WRONG_LOW_PURITY"
        elif train_purity_by_class.get(label, float("nan")) >= 0.8 and values[index] <= 0.4:
            case = "B3_TRAIN_HIGH_HELDOUT_LOW"
        else:
            case = "OTHER"
        rows.append(
            {
                "split": split,
                "sample_id": str(sample_ids[index]),
                "true_class": label,
                "cosine_prediction": int(cosine_prediction[index]),
                "linear_prediction": int(linear_prediction[index]),
                "knn_prediction": int(knn.predictions[index]),
                "cosine_correct": bool(cosine_prediction[index] == label),
                "linear_correct": bool(linear_prediction[index] == label),
                "knn_correct": bool(knn.predictions[index] == label),
                "neighborhood_purity": float(values[index]),
                "nearest_neighbor_labels": json.dumps(knn.neighbor_labels[index].tolist()),
                "nearest_neighbor_distances": json.dumps([float(value) for value in knn.neighbor_distances[index]]),
                "strongest_confusion_class": strongest,
                "train_leave_one_out_class_purity": train_purity_by_class.get(label),
                "diagnostic_case": case,
            }
        )
    return rows


def _pair_knn_rows(
    focused: list[dict[str, Any]],
    labels: np.ndarray,
    structure_predictions: np.ndarray,
    knn_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for pair in focused:
        i, j = int(pair["true_class"]), int(pair["predicted_class"])
        mask = (labels == i) & (structure_predictions == j)
        count = int(mask.sum())
        corrected = int(np.sum(mask & (knn_predictions == i)))
        same_wrong = int(np.sum(mask & (knn_predictions == j)))
        rows.append(
            {
                "true_class": i,
                "predicted_class": j,
                "pair_type": pair["pair_type"],
                "structure_pair_error_count": count,
                "knn_corrected_count": corrected,
                "knn_same_wrong_count": same_wrong,
                "knn_correction_rate": corrected / count if count else float("nan"),
                "knn_same_wrong_rate": same_wrong / count if count else float("nan"),
            }
        )
    return rows


def _class_mean_initialization(
    features: np.ndarray, labels: np.ndarray, class_ids: np.ndarray
) -> torch.Tensor:
    means = np.stack([features[labels == class_id].mean(axis=0) for class_id in class_ids])
    return torch.as_tensor(means, dtype=torch.float32)


def _diagnostic_cosine(
    protocol: str,
    tag: str,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    class_ids: np.ndarray,
    partition: list[int],
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, Any]:
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    initial = _class_mean_initialization(train_features, train_labels, class_ids)
    balanced_features, balanced_labels, _ = _replacement_balanced_bank(
        train_features, train_labels, class_ids, 256, seed + 1
    )
    permutations = _make_epoch_permutations(balanced_labels, 5, seed + 2)
    result = _train_cosine(
        condition=f"{tag}_DIAGNOSTIC_COSINE",
        protocol=protocol,
        initial_weights=initial,
        task_sizes=partition,
        train_features=balanced_features,
        train_labels=balanced_labels,
        validation_features=features[validation_indices],
        validation_labels=labels[validation_indices],
        scale=20.0,
        init_lr=0.01,
        weight_decay=0.0005,
        max_epochs=5,
        batch_size=256,
        device=device,
        permutations=permutations,
        convergence=False,
        min_epochs=5,
        patience=5,
        relative_loss_threshold=1e-4,
    )
    return result.weights, result


def _transition_rows(
    focused: list[dict[str, Any]],
    test_labels: np.ndarray,
    t4_scores: np.ndarray,
    t5_scores: np.ndarray,
    t4_knn: KNNResult,
    t5_knn: KNNResult,
) -> list[dict[str, Any]]:
    t4_predictions = np.argmax(t4_scores, axis=1)
    t5_predictions = np.argmax(t5_scores, axis=1)
    t4_purity = purity(t4_knn.neighbor_labels, test_labels)
    t5_purity = purity(t5_knn.neighbor_labels, test_labels)
    rows = []
    for pair in focused:
        i, j = int(pair["true_class"]), int(pair["predicted_class"])
        mask = test_labels == i
        count_i = int(mask.sum())
        t4_confusion = int(np.sum(mask & (t4_predictions == j)))
        t5_confusion = int(np.sum(mask & (t5_predictions == j)))
        t4_all_errors = int(np.sum(mask & (t4_predictions != test_labels)))
        t5_all_errors = int(np.sum(mask & (t5_predictions != test_labels)))
        t4_margin = t4_scores[mask, i] - t4_scores[mask, j]
        t5_margin = t5_scores[mask, i] - t5_scores[mask, j]
        t4_knn_error = int(np.sum(mask & (t4_knn.predictions == j)))
        t5_knn_error = int(np.sum(mask & (t5_knn.predictions == j)))
        t4_q05, t5_q05 = float(np.quantile(t4_margin, 0.05)), float(np.quantile(t5_margin, 0.05))
        t4_purity_mean, t5_purity_mean = float(t4_purity[mask].mean()), float(t5_purity[mask].mean())
        confusion_delta = t5_confusion - t4_confusion
        purity_delta = t5_purity_mean - t4_purity_mean
        q05_delta = t5_q05 - t4_q05
        if confusion_delta > 0 and purity_delta < -0.05 and q05_delta < 0:
            interpretation = "C1_TASK_INDUCED_RELATION_DEGRADATION_CANDIDATE"
        elif t4_confusion > 0 and t5_confusion > 0 and abs(confusion_delta) <= 1:
            interpretation = "C3_PRE_EXISTING_HARD_RELATION_CANDIDATE"
        elif confusion_delta > 0 and purity_delta >= -0.05:
            interpretation = "PURITY_MAINTAINED_BUT_DIAGNOSTIC_PROBE_WORSE"
        else:
            interpretation = "MIXED_OR_IMPROVED"
        rows.append(
            {
                "true_class": i,
                "competitor_class": j,
                "pair_type": pair["pair_type"],
                "test_samples_true_i": count_i,
                "diagnostic_probe_confusion_t4": t4_confusion,
                "diagnostic_probe_confusion_t5": t5_confusion,
                "diagnostic_probe_all_error_true_i_t4": t4_all_errors,
                "diagnostic_probe_all_error_true_i_t5": t5_all_errors,
                "confusion_delta_t5_minus_t4": confusion_delta,
                "conditional_confusion_t4": t4_confusion / count_i,
                "conditional_confusion_t5": t5_confusion / count_i,
                "margin_mean_t4": float(t4_margin.mean()),
                "margin_mean_t5": float(t5_margin.mean()),
                "margin_q05_t4": t4_q05,
                "margin_q05_t5": t5_q05,
                "margin_q05_delta_t5_minus_t4": q05_delta,
                "negative_margin_fraction_t4": float(np.mean(t4_margin < 0)),
                "negative_margin_fraction_t5": float(np.mean(t5_margin < 0)),
                "neighborhood_purity_t4": t4_purity_mean,
                "neighborhood_purity_t5": t5_purity_mean,
                "purity_delta_t5_minus_t4": purity_delta,
                "knn_pair_error_t4": t4_knn_error,
                "knn_pair_error_t5": t5_knn_error,
                "knn_pair_error_delta": t5_knn_error - t4_knn_error,
                "interpretation": interpretation,
            }
        )
    return rows


def _extract_t4_full(
    options: RelationOptions,
    checkpoint: dict[str, Any],
    data_manager: Any,
    class_ids: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = options.output_dir / ".cache" / "t4_full_t5_class_features.npz"
    if cache.is_file():
        values = np.load(cache, allow_pickle=False)
        if not np.array_equal(values["train_sample_ids"], train_ids) or not np.array_equal(
            values["test_sample_ids"], test_ids
        ):
            raise RuntimeError("T4 feature cache sample IDs do not match current dataset")
        return (
            values["train_features"], values["train_labels"],
            values["test_features"], values["test_labels"],
        )
    learner = load_frozen_learner(
        checkpoint, options.gpu_device, options.output_dir / ".scratch_t4"
    )
    train_features, train_labels, _ = extract_split(
        learner, data_manager, class_ids, "train", options.batch_size,
        options.num_workers, None, options.seed + 404,
    )
    test_features, test_labels, _ = extract_split(
        learner, data_manager, class_ids, "test", options.batch_size,
        options.num_workers, None, options.seed + 405,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        train_features=train_features,
        train_labels=train_labels,
        train_sample_ids=train_ids,
        test_features=test_features,
        test_labels=test_labels,
        test_sample_ids=test_ids,
    )
    release_learner(learner)
    return train_features, train_labels, test_features, test_labels


def _plots(
    output_dir: Path,
    directional: list[dict[str, Any]],
    focused: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
    split_scores: dict[str, np.ndarray],
    split_labels: dict[str, np.ndarray],
    knn_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
) -> None:
    for name in (
        "pair_error_contribution", "pair_margin", "neighborhood_purity",
        "knn_vs_cosine", "task_transition",
    ):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    top = directional[:20]
    labels = [f"{row['true_class']}→{row['predicted_class']}" for row in top][::-1]
    values = [100.0 * row["pair_share_of_all_errors"] for row in top][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, values)
    ax.set(xlabel="share of all errors (%)", title="Top directional error pairs")
    fig.tight_layout()
    fig.savefig(output_dir / "pair_error_contribution" / "top_pairs.png", dpi=160)
    plt.close(fig)

    first_pairs = [
        (int(row["true_class"]), int(row["predicted_class"])) for row in focused[:10]
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    positions, data, tick_labels = [], [], []
    position = 1
    for pair in first_pairs:
        for split in ("train", "validation", "test"):
            mask = split_labels[split] == pair[0]
            values = (
                split_scores[split][mask, pair[0]]
                - split_scores[split][mask, pair[1]]
            )
            if values.size == 0:
                continue
            data.append(values)
            positions.append(position)
            tick_labels.append(f"{pair[0]}→{pair[1]}\n{split[:1]}")
            position += 1
        position += 1
    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions, tick_labels, rotation=75, fontsize=7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Focused-pair margin distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "pair_margin" / "train_validation_test_margin.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for split, marker in (("train", "o"), ("validation", "s"), ("test", "^")):
        rows = [row for row in margin_rows if row["split"] == split][:20]
        ax.plot(
            range(len(rows)), [100.0 * row["negative_fraction"] for row in rows],
            marker=marker, label=split,
        )
    ax.set(xlabel="focused pair rank", ylabel="negative margin (%)", title="Pair negative-margin fraction")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pair_margin" / "negative_fraction.png", dpi=160)
    plt.close(fig)

    test_rows = [row for row in knn_rows if row["split"] == "test"]
    correct = [row["neighborhood_purity"] for row in test_rows if row["cosine_correct"]]
    wrong = [row["neighborhood_purity"] for row in test_rows if not row["cosine_correct"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(correct, bins=np.linspace(0, 1, 21), alpha=0.6, label="cosine correct", density=True)
    ax.hist(wrong, bins=np.linspace(0, 1, 21), alpha=0.6, label="cosine wrong", density=True)
    ax.set(xlabel="neighborhood purity", ylabel="density", title="Held-out neighborhood purity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "neighborhood_purity" / "purity_distribution.png", dpi=160)
    plt.close(fig)

    matrix = np.zeros((2, 2), dtype=int)
    for row in test_rows:
        matrix[int(bool(row["cosine_correct"])), int(bool(row["knn_correct"]))] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["kNN wrong", "kNN correct"])
    ax.set_yticks([0, 1], ["cosine wrong", "cosine correct"])
    ax.set_title("Cosine × k-NN correctness")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "knn_vs_cosine" / "correctness_matrix.png", dpi=160)
    plt.close(fig)

    names = [f"{row['true_class']}→{row['competitor_class']}" for row in transition_rows[:20]]
    for field, title, filename in (
        ("confusion_delta_t5_minus_t4", "T5−T4 diagnostic confusion", "confusion_change.png"),
        ("purity_delta_t5_minus_t4", "T5−T4 purity", "purity_change.png"),
        ("margin_q05_delta_t5_minus_t4", "T5−T4 margin q05", "margin_q05_change.png"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(names)), [row[field] for row in transition_rows[:20]])
        ax.set_xticks(range(len(names)), names, rotation=70, ha="right", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / "task_transition" / filename, dpi=160)
        plt.close(fig)


def run_protocol(options: RelationOptions) -> dict[str, Any]:
    started = time.time()

    def progress(message: str) -> None:
        print(
            f"[{options.protocol} +{time.time() - started:.1f}s] {message}",
            flush=True,
        )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    progress("loading frozen T4/T5 metadata and cached T5 features")
    fingerprints_before = {
        "t4": _edge_fingerprint(options.t4_checkpoint),
        "t5": _edge_fingerprint(options.t5_checkpoint),
    }
    t4_checkpoint = torch_load(options.t4_checkpoint)
    t5_checkpoint = torch_load(options.t5_checkpoint)
    if int(t4_checkpoint["cur_task"]) + 1 != 4 or int(t5_checkpoint["cur_task"]) + 1 != 5:
        raise ValueError("expected exact T4/T5 task-end checkpoints")
    partition = list(map(int, t5_checkpoint["task_sizes"]))
    total_classes = int(t5_checkpoint["total_classes"])
    known_before = total_classes - partition[-1]
    class_ids = np.arange(total_classes, dtype=np.int64)
    data_manager, _ = build_data_manager(options.protocol, options.data_root, options.seed)
    train_ids, data_train_labels = _dataset_identity(data_manager, class_ids, "train")
    test_ids, data_test_labels = _dataset_identity(data_manager, class_ids, "test")
    cached = np.load(options.t5_feature_cache, allow_pickle=False)
    t5_train = np.asarray(cached["train_features"], dtype=np.float32)
    train_labels = np.asarray(cached["train_labels"], dtype=np.int64)
    t5_test = np.asarray(cached["test_features"], dtype=np.float32)
    test_labels = np.asarray(cached["test_labels"], dtype=np.int64)
    if not np.array_equal(train_labels, data_train_labels) or not np.array_equal(
        test_labels, data_test_labels
    ):
        raise RuntimeError("T5 feature cache label order differs from current dataset")
    train_indices, validation_indices, validation_rows, no_validation = _validation_indices(
        train_labels, class_ids, options.seed + 5300
    )
    normalized_train = _normalize(t5_train)
    normalized_test = _normalize(t5_test)
    scale = float(t5_checkpoint["config"].get("scale", 20.0))
    if not math.isclose(scale, 20.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"expected Structure cosine scale 20, checkpoint records {scale}")
    structure_weights = checkpoint_classifier_weights(t5_checkpoint).numpy()[:total_classes]
    equivalence, converted_weights, structure_test_scores = _linear_equivalence(
        normalized_test, structure_weights, scale
    )
    reconstructed_accuracy = float(
        np.mean(np.argmax(structure_test_scores, axis=1) == test_labels) * 100.0
    )
    historical_path = (
        options.t5_feature_cache.parent.parent
        / "gaussian_vs_actual_classifier_accuracy.csv"
    )
    historical_rows = read_csv(historical_path)
    historical_row = next(
        (
            row
            for row in historical_rows
            if row.get("task") == "5"
            and row.get("row_type") == "summary"
            and row.get("condition") == "STRUCTURE_CA"
        ),
        None,
    )
    if historical_row is not None:
        historical_accuracy = float(historical_row["seen_class_accuracy"])
        accuracy_difference = abs(reconstructed_accuracy - historical_accuracy)
        equivalence["historical_structure_seen_accuracy"] = historical_accuracy
        equivalence["reconstructed_structure_seen_accuracy"] = reconstructed_accuracy
        equivalence["historical_accuracy_absolute_difference"] = accuracy_difference
        equivalence["historical_prediction_path_regression"] = (
            "PASS" if accuracy_difference <= 1e-8 else "FAIL"
        )
        if accuracy_difference > 1e-8:
            equivalence["status"] = "FAIL"
    else:
        equivalence["historical_prediction_path_regression"] = "UNAVAILABLE"
        equivalence["reconstructed_structure_seen_accuracy"] = reconstructed_accuracy
    del t5_checkpoint, structure_weights
    gc.collect()
    write_json(options.output_dir / "cosine_linear_equivalence.json", equivalence)
    if equivalence["status"] != "PASS":
        raise RuntimeError(f"cosine-linear equivalence failed: {equivalence}")
    progress("cosine→linear equivalence PASS; running three-setting linear sanity")

    probe_train = normalized_train[train_indices]
    probe_train_labels = train_labels[train_indices]
    validation = normalized_train[validation_indices]
    validation_labels = train_labels[validation_indices]
    regularization_rows, selected_linear, selected_l2 = _linear_sanity(
        options,
        converted_weights,
        probe_train,
        probe_train_labels,
        validation,
        validation_labels,
        normalized_test,
        test_labels,
    )
    write_csv(options.output_dir / "linear_regularization_check.csv", regularization_rows)
    _write_linear_audit(options.output_dir / "linear_probe_training_audit.md", selected_l2)
    structure_validation_scores = validation @ converted_weights.T
    structure_validation_predictions = np.argmax(structure_validation_scores, axis=1)
    validation_micro, validation_macro = _micro_macro(
        structure_validation_predictions, validation_labels
    )
    summary_validation = {
        "row_type": "summary",
        "class_id": None,
        "original_train_samples": int(train_labels.size),
        "diagnostic_train_samples": int(train_indices.size),
        "validation_samples": int(validation_indices.size),
        "validation_classes": total_classes - len(no_validation),
        "classes_without_validation": len(no_validation),
        "structure_validation_micro_accuracy": validation_micro,
        "structure_validation_macro_accuracy": validation_macro,
        "selected_linear_validation_micro_accuracy": selected_linear.validation_micro,
        "selected_linear_validation_macro_accuracy": selected_linear.validation_macro,
        "split_before_replacement_sampling": True,
        "validation_excluded_from_diagnostic_training": True,
        "used_in_historical_backbone_training": True,
        "representation_unseen": False,
    }
    write_csv(
        options.output_dir / "validation_split_audit.csv",
        [summary_validation] + validation_rows,
    )
    progress("linear/validation audit complete; computing directional pairs and margins")

    structure_predictions = np.argmax(structure_test_scores, axis=1)
    directional = _directional_confusions(
        structure_predictions, test_labels, train_labels, partition, known_before
    )
    write_csv(options.output_dir / "directional_pair_confusion.csv", directional)
    focused = _focused_pairs(
        directional, options.top_global_pairs, options.top_conditional_pairs
    )
    structure_train_scores = normalized_train[train_indices] @ converted_weights.T
    pair_split_scores = {
        "train": structure_train_scores,
        "validation": structure_validation_scores,
        "test": structure_test_scores,
    }
    pair_split_labels = {
        "train": probe_train_labels,
        "validation": validation_labels,
        "test": test_labels,
    }
    pair_margin_rows = _pair_margin_rows(
        focused, pair_split_scores, pair_split_labels
    )
    write_csv(options.output_dir / "pair_margin_distribution.csv", pair_margin_rows)
    margin_lookup = {
        (row["true_class"], row["competitor_class"], row["split"]): row
        for row in pair_margin_rows
    }
    top_rows = []
    for pair in focused:
        key = (pair["true_class"], pair["predicted_class"])
        train_row = margin_lookup[(*key, "train")]
        test_row = margin_lookup[(*key, "test")]
        top_rows.append(
            {
                **pair,
                "error_share": pair["pair_share_of_all_errors"],
                "train_negative_fraction": train_row["negative_fraction"],
                "test_negative_fraction": test_row["negative_fraction"],
                "train_q05": train_row["q05"],
                "test_q05": test_row["q05"],
            }
        )
    write_csv(options.output_dir / "top_pair_analysis.csv", top_rows)

    progress("pair analysis complete; selecting k and running neighborhood diagnostics")
    selected_k, knn_selection_rows = _knn_select(
        probe_train,
        probe_train_labels,
        validation,
        validation_labels,
        options.knn_candidates,
        options.gpu_device,
    )
    train_knn = cosine_knn(
        probe_train,
        probe_train_labels,
        probe_train,
        selected_k,
        options.gpu_device,
        reference_ids=train_ids[train_indices],
        query_ids=train_ids[train_indices],
    )
    train_purity_values = purity(train_knn.neighbor_labels, probe_train_labels)
    train_purity_by_class = {
        class_id: float(train_purity_values[probe_train_labels == class_id].mean())
        for class_id in class_ids.tolist()
    }
    validation_knn = cosine_knn(
        probe_train, probe_train_labels, validation, selected_k, options.gpu_device
    )
    test_knn = cosine_knn(
        probe_train, probe_train_labels, normalized_test, selected_k, options.gpu_device
    )
    selected_linear_validation_scores = validation @ selected_linear.weights.T
    selected_linear_test_scores = normalized_test @ selected_linear.weights.T
    knn_sample_rows = _knn_sample_rows(
        split="validation",
        sample_ids=train_ids[validation_indices],
        labels=validation_labels,
        cosine_scores=structure_validation_scores,
        linear_scores=selected_linear_validation_scores,
        knn=validation_knn,
        train_purity_by_class=train_purity_by_class,
    )
    knn_sample_rows.extend(
        _knn_sample_rows(
            split="test",
            sample_ids=test_ids,
            labels=test_labels,
            cosine_scores=structure_test_scores,
            linear_scores=selected_linear_test_scores,
            knn=test_knn,
            train_purity_by_class=train_purity_by_class,
        )
    )
    write_csv(options.output_dir / "knn_neighborhood_diagnostic.csv", knn_sample_rows)

    knn_summary = list(knn_selection_rows)
    knn_summary.extend(
        [
            _accuracy_summary(
                "Structure cosine", structure_predictions, test_labels,
                known_before, "test",
            ),
            _accuracy_summary(
                "Validated Linear", np.argmax(selected_linear_test_scores, axis=1),
                test_labels, known_before, "test", selected_l2=selected_l2,
            ),
            _accuracy_summary(
                f"k-NN (k={selected_k})", test_knn.predictions,
                test_labels, known_before, "test", selected_k=selected_k,
            ),
        ]
    )
    count_group, thresholds = _sample_count_groups(probe_train_labels, class_ids)
    for group in ("very_low", "low", "medium", "high"):
        mask = np.asarray([count_group(int(label)) == group for label in test_labels])
        for method, predictions in (
            ("Structure cosine", structure_predictions),
            ("Validated Linear", np.argmax(selected_linear_test_scores, axis=1)),
            (f"k-NN (k={selected_k})", test_knn.predictions),
        ):
            knn_summary.append(
                {
                    "row_type": "sample_count_group",
                    "split": "test",
                    "method": method,
                    "sample_count_group": group,
                    "num_samples": int(mask.sum()),
                    "accuracy": float(np.mean(predictions[mask] == test_labels[mask]) * 100.0),
                    **thresholds,
                }
            )
    write_csv(options.output_dir / "knn_summary.csv", knn_summary)
    pair_knn_rows = _pair_knn_rows(
        focused, test_labels, structure_predictions, test_knn.predictions
    )
    write_csv(options.output_dir / "pair_knn_recovery.csv", pair_knn_rows)

    progress("k-NN diagnostic complete; extracting T4 features for the full T5 class set")
    t4_train, t4_train_labels, t4_test, t4_test_labels = _extract_t4_full(
        options,
        t4_checkpoint,
        data_manager,
        class_ids,
        train_ids,
        test_ids,
    )
    del t4_checkpoint
    gc.collect()
    if not np.array_equal(t4_train_labels, train_labels) or not np.array_equal(
        t4_test_labels, test_labels
    ):
        raise RuntimeError("T4/T5 common sample labels differ")
    t4_normalized_train = _normalize(t4_train)
    t4_normalized_test = _normalize(t4_test)
    progress("T4 feature extraction/cache ready; fitting matched T4/T5 diagnostic probes")
    t4_probe_weights, t4_probe_result = _diagnostic_cosine(
        options.protocol, "T4", t4_normalized_train, train_labels,
        train_indices, validation_indices, class_ids, partition,
        options.gpu_device, options.seed + 6100,
    )
    t5_probe_weights, t5_probe_result = _diagnostic_cosine(
        options.protocol, "T5", normalized_train, train_labels,
        train_indices, validation_indices, class_ids, partition,
        options.gpu_device, options.seed + 6100,
    )
    t4_probe_scores = 20.0 * (
        t4_normalized_test
        @ _normalize(t4_probe_weights).T
    )
    t5_probe_scores = 20.0 * (
        normalized_test
        @ _normalize(t5_probe_weights).T
    )
    t4_probe_micro, t4_probe_macro = _class_macro_from_scores(t4_probe_scores, test_labels)
    t5_probe_micro, t5_probe_macro = _class_macro_from_scores(t5_probe_scores, test_labels)
    t4_test_knn = cosine_knn(
        t4_normalized_train[train_indices], probe_train_labels,
        t4_normalized_test, selected_k, options.gpu_device,
    )
    t5_test_knn = test_knn
    transition_rows = _transition_rows(
        focused,
        test_labels,
        t4_probe_scores,
        t5_probe_scores,
        t4_test_knn,
        t5_test_knn,
    )
    write_csv(
        options.output_dir / "pair_task_transition_analysis.csv", transition_rows
    )
    np.savez_compressed(
        options.output_dir / "diagnostic_weights.npz",
        converted_structure_linear=converted_weights,
        selected_linear=selected_linear.weights,
        selected_linear_l2=np.asarray([selected_l2]),
        t4_diagnostic_cosine=t4_probe_weights,
        t5_diagnostic_cosine=t5_probe_weights,
    )
    _plots(
        options.output_dir,
        directional,
        focused,
        pair_margin_rows,
        pair_split_scores,
        pair_split_labels,
        knn_sample_rows,
        transition_rows,
    )
    progress("transition tables and plots complete; validating immutable checkpoints")

    fingerprints_after = {
        "t4": _edge_fingerprint(options.t4_checkpoint),
        "t5": _edge_fingerprint(options.t5_checkpoint),
    }
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("T4/T5 main checkpoint fingerprint changed")
    manifest = {
        "status": "complete",
        "protocol": options.protocol,
        "task": 5,
        "seed": options.seed,
        "t4_checkpoint": str(options.t4_checkpoint),
        "t5_checkpoint": str(options.t5_checkpoint),
        "checkpoint_fingerprints_before": fingerprints_before,
        "checkpoint_fingerprints_after": fingerprints_after,
        "checkpoints_unchanged": True,
        "diagnostic_only": True,
        "backbone_adapter_updated": False,
        "train_samples": int(train_labels.size),
        "diagnostic_train_samples": int(train_indices.size),
        "validation_samples": int(validation_indices.size),
        "test_samples": int(test_labels.size),
        "classes_without_validation": no_validation,
        "cosine_linear_equivalence": equivalence,
        "selected_linear_l2": selected_l2,
        "selected_linear_converged": selected_linear.converged,
        "selected_linear_gradient_norm": selected_linear.gradient_norm,
        "focused_pair_count": len(focused),
        "top_global_pairs": options.top_global_pairs,
        "top_conditional_pairs": options.top_conditional_pairs,
        "knn_candidates": list(options.knn_candidates),
        "selected_k": selected_k,
        "sample_count_quantiles": thresholds,
        "t4_t5_sample_ids_identical": True,
        "t4_t5_common_class_set": total_classes,
        "t4_diagnostic_probe_steps": t4_probe_result.optimizer_steps,
        "t5_diagnostic_probe_steps": t5_probe_result.optimizer_steps,
        "t4_diagnostic_probe_test_micro_accuracy": t4_probe_micro,
        "t4_diagnostic_probe_test_macro_accuracy": t4_probe_macro,
        "t5_diagnostic_probe_test_micro_accuracy": t5_probe_micro,
        "t5_diagnostic_probe_test_macro_accuracy": t5_probe_macro,
        "variant_pair_analysis": "skipped_no_verified_sample_aligned_existing_logits",
        "t2_t3_executed": False,
        "ia10_car10_executed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_dir / "run_manifest.json", manifest)
    progress("protocol complete")
    del t4_train, t4_test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest
