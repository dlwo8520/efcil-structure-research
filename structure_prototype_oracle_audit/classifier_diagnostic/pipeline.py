from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT
from oracle_audit.metrics import EPS, safe_cosine
from oracle_audit.oracle_head import (
    DiagnosticCosineHead,
    checkpoint_classifier_weights,
    cosine_scores,
    evaluate_scores,
    load_classifier_initialization,
    train_diagnostic_head,
)
from oracle_audit.runtime import (
    birth_task,
    build_data_manager,
    discover_checkpoints,
    extract_split,
    load_frozen_learner,
    release_learner,
    torch_load,
)


METHODS_WITH_SCORES = ("STRUCTURE_CA", "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED")


@dataclass
class DiagnosticOptions:
    protocol: str
    gpu_device: torch.device
    output_dir: Path
    scientific_dir: Path
    training_dir: Path
    data_root: Path
    seed: int = 1993
    tasks: set[int] | None = None
    batch_size: int = 64
    num_workers: int = 4
    bias_points: int = 101
    force: bool = False


def _finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=_finite) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite(row.get(key)) for key in fields})
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _edge_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(0, stat.st_size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    return {"path": str(path), "size": stat.st_size, "edge_sha256": digest.hexdigest()}


def _checksum(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _normalize(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)


def _scores(features: np.ndarray, weights: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return scale * (_normalize(features) @ _normalize(weights).T)


def _stats(features: np.ndarray, labels: np.ndarray, class_ids: np.ndarray, jitter: float):
    means: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    banks: dict[int, np.ndarray] = {}
    identity = np.eye(features.shape[1], dtype=np.float64)
    for class_id in class_ids:
        values = np.asarray(features[labels == int(class_id)], dtype=np.float64)
        if values.shape[0] == 0:
            raise RuntimeError(f"missing train features for class {class_id}")
        means.append(values.mean(axis=0))
        covariance = (
            np.cov(values, rowvar=False) + jitter * identity
            if values.shape[0] >= 2
            else jitter * identity
        )
        covariances.append(covariance)
        banks[int(class_id)] = values.astype(np.float32)
    return np.stack(means), np.stack(covariances), banks


def _accuracy_row(
    protocol: str,
    task: int,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    known_before: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "task": task,
        "row_type": "summary",
        "condition": condition,
        **extra,
        **evaluate_scores(scores, labels, class_ids, known_before),
    }


def _reference_accuracy_row(
    protocol: str,
    task: int,
    condition: str,
    reference: dict[str, str],
    **extra: Any,
) -> dict[str, Any]:
    numeric = (
        "num_samples", "seen_class_accuracy", "old_accuracy", "new_accuracy",
        "old_only_accuracy", "new_only_accuracy", "old_to_new_count",
        "old_to_old_wrong_count", "old_correct_count", "new_to_old_count",
        "new_to_new_wrong_count", "new_correct_count", "old_to_new_rate",
        "old_to_old_wrong_rate", "new_to_old_rate", "new_to_new_wrong_rate",
    )
    row: dict[str, Any] = {
        "protocol": protocol,
        "task": task,
        "row_type": "summary",
        "condition": condition,
        **extra,
    }
    for key in numeric:
        raw = reference.get(key, "")
        if raw != "":
            row[key] = int(float(raw)) if key.endswith("count") or key == "num_samples" else float(raw)
    return row


def _class_and_task_accuracy_rows(
    base: dict[str, Any],
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    partition: list[int],
) -> list[dict[str, Any]]:
    predictions = class_ids[np.argmax(scores, axis=1)]
    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        mask = labels == int(class_id)
        rows.append(
            {
                **base,
                "row_type": "class",
                "class_id": int(class_id),
                "class_birth_task": birth_task(int(class_id), partition),
                "num_samples": int(mask.sum()),
                "accuracy": float(np.mean(predictions[mask] == labels[mask]) * 100.0),
            }
        )
    boundary = 0
    for source_task, size in enumerate(partition[: int(base["task"])], start=1):
        start, boundary = boundary, boundary + int(size)
        mask = (labels >= start) & (labels < boundary)
        rows.append(
            {
                **base,
                "row_type": "source_task",
                "source_task": source_task,
                "num_samples": int(mask.sum()),
                "accuracy": float(np.mean(predictions[mask] == labels[mask]) * 100.0),
            }
        )
    return rows


def _calibration(scores: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict[str, float]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    chosen = probabilities[np.arange(labels.size), labels]
    nll = float(-np.mean(np.log(np.maximum(chosen, 1e-12))))
    one_hot = np.eye(scores.shape[1], dtype=np.float64)[labels]
    brier = float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = prediction == labels
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"nll": nll, "brier_score": brier, "ece_15bin": ece}


def _quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q50": float(np.quantile(values, 0.50)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _score_components(
    scores: np.ndarray,
    labels: np.ndarray,
    known_before: int,
) -> dict[str, np.ndarray]:
    n, classes = scores.shape
    indices = np.arange(n)
    true = scores[indices, labels]
    old_columns = np.arange(known_before)
    new_columns = np.arange(known_before, classes)
    same = np.empty(n, dtype=np.float64)
    opposite = np.empty(n, dtype=np.float64)
    for i, label in enumerate(labels.tolist()):
        if label < known_before:
            candidates = old_columns[old_columns != label]
            same[i] = np.max(scores[i, candidates]) if candidates.size else float("-inf")
            opposite[i] = np.max(scores[i, new_columns])
        else:
            candidates = new_columns[new_columns != label]
            same[i] = np.max(scores[i, candidates]) if candidates.size else float("-inf")
            opposite[i] = np.max(scores[i, old_columns])
    return {
        "true": true,
        "same": same,
        "opposite": opposite,
        "internal_margin": true - same,
        "opposite_margin": true - opposite,
        "group_margin": scores[:, new_columns].max(axis=1) - scores[:, old_columns].max(axis=1),
    }


def _rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fixed_gaussian_bank(
    means: np.ndarray,
    covariances: np.ndarray,
    counts: dict[int, int],
    task_index: int,
    task_size_argument: int,
    device: torch.device,
    seed: int,
) -> dict[int, np.ndarray]:
    saved_cpu = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        result: dict[int, np.ndarray] = {}
        for class_id in sorted(counts):
            task_id = int(class_id) // int(task_size_argument)
            mean_scale = 0.9 + (task_id + 1) / (task_index + 1) * 0.1
            mean = torch.as_tensor(means[class_id] * mean_scale, dtype=torch.float32, device=device)
            covariance = torch.as_tensor(covariances[class_id], dtype=torch.float32, device=device)
            distribution = torch.distributions.MultivariateNormal(mean, covariance)
            result[class_id] = distribution.sample((int(counts[class_id]),)).cpu().numpy()
        return result
    finally:
        torch.set_rng_state(saved_cpu)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _assemble_bank(banks: dict[int, np.ndarray], class_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for class_id in class_ids:
        values = np.asarray(banks[int(class_id)], dtype=np.float32)
        features.append(values)
        labels.append(np.full(values.shape[0], int(class_id), dtype=np.int64))
    return np.concatenate(features), np.concatenate(labels)


def _actual_old_bank(
    actual: dict[int, np.ndarray],
    known_before: int,
    cap: int,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    rng = np.random.RandomState(seed)
    result: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for class_id in range(known_before):
        values = actual[class_id]
        count = min(int(values.shape[0]), cap)
        indices = np.arange(values.shape[0])
        if values.shape[0] > count:
            indices = np.sort(rng.choice(indices, count, replace=False))
        result[class_id] = values[indices].copy()
        counts[class_id] = count
    return result, counts


@dataclass
class FixedHeadResult:
    weights: np.ndarray
    trace: list[dict[str, Any]]


def _train_fixed_head(
    initial_weights: torch.Tensor,
    task_sizes: list[int],
    bank_features: np.ndarray,
    bank_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    known_before: int,
    epochs: int,
    batch_size: int,
    init_lr: float,
    weight_decay: float,
    scale: float,
    device: torch.device,
    shuffle_seed: int,
    condition: str,
    protocol: str,
    task: int,
) -> FixedHeadResult:
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        head = DiagnosticCosineHead(initial_weights, task_sizes=task_sizes).to(device)
        parameters = list(head.parameters())
        optimizer = torch.optim.SGD(
            parameters, lr=init_lr, momentum=0.9, weight_decay=weight_decay
        )
        if {id(p) for p in parameters} != {
            id(p) for group in optimizer.param_groups for p in group["params"]
        }:
            raise AssertionError("fixed-bank optimizer is not head-only")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        features = torch.as_tensor(bank_features, dtype=torch.float32)
        labels = torch.as_tensor(bank_labels, dtype=torch.long)
        trace: list[dict[str, Any]] = []
        for epoch in range(epochs):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(shuffle_seed + epoch)
            permutation = torch.randperm(labels.numel(), generator=generator)
            before = head.weight.detach().cpu().numpy().copy()
            total_loss = total_correct = total_samples = 0.0
            grad_norms: list[float] = []
            for start in range(0, labels.numel(), batch_size):
                indices = permutation[start : start + batch_size]
                inputs = features[indices].to(device, non_blocking=True)
                targets = labels[indices].to(device, non_blocking=True)
                logits = scale * head(inputs)
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_sq = sum(
                    float(torch.sum(parameter.grad.detach() ** 2).cpu())
                    for parameter in parameters
                    if parameter.grad is not None
                )
                grad_norms.append(math.sqrt(grad_sq))
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * targets.numel()
                total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
                total_samples += targets.numel()
            scheduler.step()
            weights = head.weight.detach().cpu().numpy().copy()
            test_scores = cosine_scores(test_features, weights, scale)
            test_metrics = evaluate_scores(
                test_scores, test_labels, np.arange(weights.shape[0]), known_before
            )
            trace.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "condition": condition,
                    "epoch": epoch + 1,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "train_loss": total_loss / total_samples,
                    "train_accuracy": 100.0 * total_correct / total_samples,
                    "gradient_norm_mean": float(np.mean(grad_norms)),
                    "weight_update_l2": float(np.linalg.norm(weights - before)),
                    "test_seen_accuracy_record_only": test_metrics["seen_class_accuracy"],
                    "test_not_used_for_selection": True,
                }
            )
        return FixedHeadResult(head.weight.detach().cpu().numpy().copy(), trace)
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _margin_rows(
    protocol: str,
    task: int,
    condition: str,
    labels: np.ndarray,
    pre_scores: np.ndarray,
    post_scores: np.ndarray,
    known_before: int,
) -> list[dict[str, Any]]:
    pre = _score_components(pre_scores, labels, known_before)
    post = _score_components(post_scores, labels, known_before)
    rows = []
    for index, label in enumerate(labels.tolist()):
        rows.append(
            {
                "protocol": protocol,
                "task": task,
                "condition": condition,
                "sample_id": f"{protocol}_T{task:02d}_test_{index:06d}",
                "true_class": int(label),
                "sample_group": "old" if label < known_before else "new",
                "pre_true_score": pre["true"][index],
                "pre_best_same_group_competitor": pre["same"][index],
                "pre_best_opposite_group_competitor": pre["opposite"][index],
                "pre_internal_margin": pre["internal_margin"][index],
                "pre_opposite_group_margin": pre["opposite_margin"][index],
                "post_true_score": post["true"][index],
                "post_best_same_group_competitor": post["same"][index],
                "post_best_opposite_group_competitor": post["opposite"][index],
                "post_internal_margin": post["internal_margin"][index],
                "post_opposite_group_margin": post["opposite_margin"][index],
                "internal_margin_change": post["internal_margin"][index] - pre["internal_margin"][index],
                "opposite_group_margin_change": post["opposite_margin"][index] - pre["opposite_margin"][index],
            }
        )
    return rows


def _distribution_rows(
    protocol: str,
    task: int,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    known_before: int,
) -> list[dict[str, Any]]:
    values = _score_components(scores, labels, known_before)
    rows = []
    for group, mask in (("old", labels < known_before), ("new", labels >= known_before)):
        for name in ("true", "same", "opposite", "internal_margin", "opposite_margin", "group_margin"):
            rows.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "condition": condition,
                    "sample_group": group,
                    "score_type": name,
                    "num_samples": int(mask.sum()),
                    **_quantiles(values[name][mask]),
                }
            )
    return rows


def _alignment_rows(
    protocol: str,
    task: int,
    condition: str,
    weights: np.ndarray,
    means: np.ndarray,
    known_before: int,
    partition: list[int],
) -> list[dict[str, Any]]:
    return [
        {
            "protocol": protocol,
            "task": task,
            "condition": condition,
            "class_id": class_id,
            "class_group": "old" if class_id < known_before else "new",
            "class_birth_task": birth_task(class_id, partition),
            "class_age": task - birth_task(class_id, partition),
            "weight_mean_cosine": safe_cosine(weights[class_id], means[class_id]),
            "weight_norm": float(np.linalg.norm(weights[class_id])),
            "mean_norm": float(np.linalg.norm(means[class_id])),
        }
        for class_id in range(weights.shape[0])
    ]


def _bias_rows(
    protocol: str,
    task: int,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    known_before: int,
    betas: np.ndarray,
) -> list[dict[str, Any]]:
    class_ids = np.arange(scores.shape[1])
    rows = []
    for beta in betas:
        adjusted = scores.copy()
        adjusted[:, known_before:] += float(beta)
        metrics = evaluate_scores(adjusted, labels, class_ids, known_before)
        rows.append(
            {
                "protocol": protocol,
                "task": task,
                "condition": condition,
                "beta": float(beta),
                "selection_policy": "TEST_SWEEP_DIAGNOSTIC_NOT_SELECTED",
                "balanced_group_accuracy": 0.5 * (metrics["old_accuracy"] + metrics["new_accuracy"]),
                **metrics,
            }
        )
    return rows


def _group_margin_rows(
    protocol: str,
    task: int,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    known_before: int,
) -> list[dict[str, Any]]:
    margin = scores[:, known_before:].max(axis=1) - scores[:, :known_before].max(axis=1)
    positive = labels >= known_before
    auc = _rank_auc(margin, positive)
    rows = [
        {
            "protocol": protocol,
            "task": task,
            "condition": condition,
            "row_type": "summary",
            "sample_group": "all",
            "num_samples": int(labels.size),
            "old_vs_new_group_auroc": auc,
            **_quantiles(margin),
        }
    ]
    rows.extend(
        {
            "protocol": protocol,
            "task": task,
            "condition": condition,
            "row_type": "sample",
            "sample_id": f"{protocol}_T{task:02d}_test_{index:06d}",
            "true_class": int(label),
            "sample_group": "new" if label >= known_before else "old",
            "group_margin_new_minus_old": float(margin[index]),
            "group_prediction_correct": bool((margin[index] >= 0) == (label >= known_before)),
        }
        for index, label in enumerate(labels.tolist())
    )
    return rows


def _plot_task(
    output_dir: Path,
    task: int,
    bias_rows: list[dict[str, Any]],
    score_map: dict[str, np.ndarray],
    labels: np.ndarray,
    known_before: int,
    alignment_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bias_dir = output_dir / "bias_tradeoff"
    score_dir = output_dir / "score_distributions"
    margin_dir = output_dir / "margin_distributions"
    alignment_dir = output_dir / "weight_alignment"
    for directory in (bias_dir, score_dir, margin_dir, alignment_dir):
        directory.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for condition in sorted({row["condition"] for row in bias_rows}):
        selected = [row for row in bias_rows if row["condition"] == condition]
        axes[0].plot([r["old_accuracy"] for r in selected], [r["new_accuracy"] for r in selected], label=condition)
        axes[1].plot([r["beta"] for r in selected], [r["seen_class_accuracy"] for r in selected], label=condition)
        axes[2].plot([r["beta"] for r in selected], [r["balanced_group_accuracy"] for r in selected], label=condition)
    axes[0].set(xlabel="Old accuracy", ylabel="New accuracy", title="Old-New trade-off")
    axes[1].set(xlabel="beta", ylabel="Seen accuracy", title="Seen vs bias")
    axes[2].set(xlabel="beta", ylabel="Balanced group accuracy", title="Balanced vs bias")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(bias_dir / f"task_{task:02d}_bias_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for condition, scores in score_map.items():
        components = _score_components(scores, labels, known_before)
        axes[0].hist(components["group_margin"][labels < known_before], bins=50, density=True, histtype="step", label=condition)
        axes[1].hist(components["group_margin"][labels >= known_before], bins=50, density=True, histtype="step", label=condition)
    axes[0].set_title("Old test: max(New)-max(Old)")
    axes[1].set_title("New test: max(New)-max(Old)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(score_dir / f"task_{task:02d}_group_score_hist.png", dpi=160)
    fig.savefig(margin_dir / f"task_{task:02d}_group_margin_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for condition in sorted({row["condition"] for row in alignment_rows}):
        values = [row["weight_mean_cosine"] for row in alignment_rows if row["condition"] == condition]
        ax.hist(values, bins=40, density=True, histtype="step", label=condition)
    ax.set(xlabel="cos(weight, class mean)", ylabel="density", title=f"Task {task} weight-mean alignment")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(alignment_dir / f"task_{task:02d}_alignment.png", dpi=160)
    plt.close(fig)


def _aggregate_task_files(output_dir: Path) -> None:
    filenames = (
        "ncm_absolute_comparison.csv",
        "ncm_prediction_change_analysis.csv",
        "gaussian_vs_actual_classifier_accuracy.csv",
        "ca_margin_shift.csv",
        "score_distribution_summary.csv",
        "weight_mean_alignment.csv",
        "classifier_weight_diagnostics.csv",
        "ca_optimization_trace.csv",
        "calibration_metrics.csv",
        "bias_sweep.csv",
        "old_new_group_margin.csv",
    )
    for filename in filenames:
        rows: list[dict[str, Any]] = []
        for path in sorted((output_dir / "task_results").glob(f"task_*/{filename}")):
            rows.extend(read_csv(path))
        write_csv(output_dir / filename, rows)


def run_task(
    options: DiagnosticOptions,
    checkpoint_path: Path,
    data_manager: Any,
    protocol_config: dict[str, Any],
    reference_rows: list[dict[str, str]],
) -> dict[str, Any]:
    checkpoint = torch_load(checkpoint_path)
    task = int(checkpoint["cur_task"]) + 1
    task_dir = options.output_dir / "task_results" / f"task_{task:02d}"
    marker = task_dir / "task_manifest.json"
    if marker.is_file() and not options.force:
        return read_json(marker)
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    total_classes = int(checkpoint["total_classes"])
    partition = list(map(int, protocol_config["task_partition"]))
    known_before = total_classes - partition[task - 1]
    if task <= 1:
        raise ValueError("classifier bottleneck diagnostic requires T2 or later")
    class_ids = np.arange(total_classes, dtype=np.int64)
    feature_cache = options.output_dir / ".cache" / f"task_{task:02d}_features.npz"
    if feature_cache.is_file() and not options.force:
        cached = np.load(feature_cache, allow_pickle=False)
        train_features, train_labels = cached["train_features"], cached["train_labels"]
        test_features, test_labels = cached["test_features"], cached["test_labels"]
    else:
        learner = load_frozen_learner(checkpoint, options.gpu_device, task_dir / ".scratch")
        train_features, train_labels, _ = extract_split(
            learner, data_manager, class_ids, "train", options.batch_size,
            options.num_workers, None, options.seed + task,
        )
        test_features, test_labels, _ = extract_split(
            learner, data_manager, class_ids, "test", options.batch_size,
            options.num_workers, None, options.seed + task,
        )
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            feature_cache,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
        )
        release_learner(learner)

    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle_means, oracle_covariances, actual_banks = _stats(
        train_features, train_labels, class_ids, jitter
    )
    structure_means = np.asarray(checkpoint["class_means"][:total_classes], dtype=np.float64)
    structure_covariances = np.stack(
        [value.detach().cpu().numpy() for value in checkpoint["class_covs"][:total_classes]]
    )
    scale = float(checkpoint["config"].get("scale", 20.0))

    # Requested NCM: current-class prototype is identical in both conditions.
    apr_prototypes = oracle_means.copy()
    apr_prototypes[:known_before] = structure_means[:known_before]
    oracle_prototypes = oracle_means.copy()
    apr_ncm_scores = _scores(test_features, apr_prototypes, 1.0)
    oracle_ncm_scores = _scores(test_features, oracle_prototypes, 1.0)

    initial_weights, sidecar, initialization_source = load_classifier_initialization(
        checkpoint, checkpoint_path, class_ids, task
    )
    if sidecar is None or initialization_source != "exact_pre_ca_sidecar":
        raise RuntimeError(f"{options.protocol} T{task} lacks exact pre-CA sidecar")
    checkpoint_weights = checkpoint_classifier_weights(checkpoint).numpy()[:total_classes]
    pre_scores = cosine_scores(test_features, initial_weights.numpy(), scale)
    structure_scores = cosine_scores(test_features, checkpoint_weights, scale)

    reference = {
        row["condition"]: row
        for row in reference_rows
        if int(row["task"]) == task
    }
    required_reference = {"A_STRUCTURE", "B_ORACLE_MEAN", "D_FULL_ORACLE"}
    if not required_reference.issubset(reference):
        raise RuntimeError(f"missing existing A/B/D reference for {options.protocol} T{task}")

    # Reconstruct D weight only because the old audit did not persist diagnostic heads.
    d_means = structure_means.copy()
    d_covariances = structure_covariances.copy()
    d_means[:known_before] = oracle_means[:known_before]
    d_covariances[:known_before] = oracle_covariances[:known_before]
    d_result = train_diagnostic_head(
        initial_weights,
        d_means,
        d_covariances,
        class_ids,
        task_index=task - 1,
        task_size=int(sidecar["task_size_argument"]),
        epochs=int(sidecar["ca_epochs"]),
        samples_per_class=int(sidecar["replay_samples_per_class"]),
        init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]),
        scale=float(sidecar["scale"]),
        device=options.gpu_device,
        seed=options.seed + task * 10000,
        rng_state=sidecar["rng_state_before_ca"],
        task_sizes=list(map(int, sidecar["task_sizes"])),
    )
    d_scores = cosine_scores(test_features, d_result.weights, scale)
    d_metrics = evaluate_scores(d_scores, test_labels, class_ids, known_before)
    reference_d = float(reference["D_FULL_ORACLE"]["seen_class_accuracy"])
    if abs(d_metrics["seen_class_accuracy"] - reference_d) > 1e-9:
        raise RuntimeError(
            f"D reconstruction mismatch: {d_metrics['seen_class_accuracy']} vs {reference_d}"
        )

    cap = int(sidecar["replay_samples_per_class"])
    actual_old, old_counts = _actual_old_bank(
        actual_banks, known_before, cap, options.seed + task * 101
    )
    full_equals_matched = all(actual_banks[c].shape[0] <= cap for c in range(known_before))
    new_counts = {class_id: cap for class_id in range(known_before, total_classes)}
    new_gaussian = _fixed_gaussian_bank(
        structure_means,
        structure_covariances,
        new_counts,
        task - 1,
        int(sidecar["task_size_argument"]),
        options.gpu_device,
        options.seed + task * 1000 + 17,
    )
    old_gaussian = _fixed_gaussian_bank(
        oracle_means,
        oracle_covariances,
        old_counts,
        task - 1,
        int(sidecar["task_size_argument"]),
        options.gpu_device,
        options.seed + task * 1000 + 23,
    )
    g_banks = {**old_gaussian, **new_gaussian}
    r_banks = {**actual_old, **new_gaussian}
    g_features, g_labels = _assemble_bank(g_banks, class_ids)
    r_features, r_labels = _assemble_bank(r_banks, class_ids)
    if not np.array_equal(g_labels, r_labels):
        raise AssertionError("G/R bank label layout differs")
    for class_id in range(known_before, total_classes):
        if not np.array_equal(g_banks[class_id], r_banks[class_id]):
            raise AssertionError("G/R current-class bank differs")

    common_training = dict(
        initial_weights=initial_weights,
        task_sizes=list(map(int, sidecar["task_sizes"])),
        test_features=test_features,
        test_labels=test_labels,
        known_before=known_before,
        epochs=int(sidecar["ca_epochs"]),
        batch_size=cap,
        init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]),
        scale=scale,
        device=options.gpu_device,
        shuffle_seed=options.seed + task * 10000 + 91,
        protocol=options.protocol,
        task=task,
    )
    g_result = _train_fixed_head(
        bank_features=g_features, bank_labels=g_labels, condition="G_FIXED", **common_training
    )
    r_result = _train_fixed_head(
        bank_features=r_features, bank_labels=r_labels, condition="R_FIXED", **common_training
    )
    g_scores = cosine_scores(test_features, g_result.weights, scale)
    r_scores = cosine_scores(test_features, r_result.weights, scale)

    ncm_rows = [
        _accuracy_row(options.protocol, task, "APR_NCM", apr_ncm_scores, test_labels, class_ids, known_before, old_source="APR mean", classifier="cosine NCM"),
        _accuracy_row(options.protocol, task, "ORACLE_NCM", oracle_ncm_scores, test_labels, class_ids, known_before, old_source="oracle train mean", classifier="cosine NCM"),
        _accuracy_row(options.protocol, task, "STRUCTURE_CA", structure_scores, test_labels, class_ids, known_before, old_source="Structure Gaussian", classifier="cosine CA head"),
        _reference_accuracy_row(options.protocol, task, "ORACLE_MEAN_CA", reference["B_ORACLE_MEAN"], old_source="oracle mean + Structure covariance", classifier="cosine CA head"),
        _reference_accuracy_row(options.protocol, task, "FULL_ORACLE_GAUSSIAN_CA", reference["D_FULL_ORACLE"], old_source="oracle Gaussian regenerated per epoch", classifier="cosine CA head"),
    ]
    oracle_seen = next(row["seen_class_accuracy"] for row in ncm_rows if row["condition"] == "ORACLE_NCM")
    for row in ncm_rows:
        row["oracle_ncm_minus_method_seen"] = oracle_seen - row["seen_class_accuracy"]
        row["ncm_current_prototype_policy"] = "same current oracle-train mean in APR/Oracle NCM"
    write_csv(task_dir / "ncm_absolute_comparison.csv", ncm_rows)

    apr_prediction = np.argmax(apr_ncm_scores, axis=1)
    oracle_prediction = np.argmax(oracle_ncm_scores, axis=1)
    changed = np.flatnonzero(apr_prediction != oracle_prediction)
    change_rows = []
    for index in changed.tolist():
        label = int(test_labels[index])
        apr_comp = np.max(np.delete(apr_ncm_scores[index], label))
        oracle_comp = np.max(np.delete(oracle_ncm_scores[index], label))
        shift_class = label if label < known_before else int(apr_prediction[index])
        change_rows.append(
            {
                "protocol": options.protocol,
                "task": task,
                "sample_id": f"{options.protocol}_T{task:02d}_test_{index:06d}",
                "true_class": label,
                "APR_prediction": int(apr_prediction[index]),
                "Oracle_prediction": int(oracle_prediction[index]),
                "APR_true_score": apr_ncm_scores[index, label],
                "APR_top_competitor_score": apr_comp,
                "APR_margin": apr_ncm_scores[index, label] - apr_comp,
                "Oracle_true_score": oracle_ncm_scores[index, label],
                "Oracle_top_competitor_score": oracle_comp,
                "Oracle_margin": oracle_ncm_scores[index, label] - oracle_comp,
                "margin_change": (oracle_ncm_scores[index, label] - oracle_comp) - (apr_ncm_scores[index, label] - apr_comp),
                "prototype_shift_class": shift_class,
                "prototype_shift_norm": float(np.linalg.norm(oracle_prototypes[shift_class] - apr_prototypes[shift_class])),
                "prototype_shift_cosine": safe_cosine(oracle_prototypes[shift_class], apr_prototypes[shift_class]),
            }
        )
    write_csv(task_dir / "ncm_prediction_change_analysis.csv", change_rows)

    score_map = {
        "STRUCTURE_CA": structure_scores,
        "FULL_ORACLE_GAUSSIAN_CA": d_scores,
        "G_FIXED": g_scores,
        "R_FIXED": r_scores,
    }
    weight_map = {
        "PRE_CA": initial_weights.numpy(),
        "STRUCTURE_CA": checkpoint_weights,
        "FULL_ORACLE_GAUSSIAN_CA": d_result.weights,
        "G_FIXED": g_result.weights,
        "R_FIXED": r_result.weights,
    }
    accuracy_rows = [
        _accuracy_row(options.protocol, task, "STRUCTURE_CA", structure_scores, test_labels, class_ids, known_before, old_source="Structure Gaussian regenerated per epoch", classifier="cosine CA head"),
        _reference_accuracy_row(options.protocol, task, "ORACLE_MEAN_CA", reference["B_ORACLE_MEAN"], old_source="oracle mean + Structure covariance", classifier="cosine CA head"),
        _reference_accuracy_row(options.protocol, task, "FULL_ORACLE_GAUSSIAN_CA", reference["D_FULL_ORACLE"], old_source="oracle Gaussian regenerated per epoch", classifier="cosine CA head"),
        _accuracy_row(options.protocol, task, "G_FIXED", g_scores, test_labels, class_ids, known_before, old_source="fixed matched-count oracle Gaussian", classifier="cosine CA head"),
        _accuracy_row(options.protocol, task, "R_FIXED", r_scores, test_labels, class_ids, known_before, old_source="fixed matched/full actual old features", classifier="cosine CA head", matched_equals_full_actual=full_equals_matched),
    ]
    for condition in ("STRUCTURE_CA", "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED"):
        accuracy_rows.extend(
            _class_and_task_accuracy_rows(
                {"protocol": options.protocol, "task": task, "condition": condition},
                score_map[condition], test_labels, class_ids, partition,
            )
        )
    write_csv(task_dir / "gaussian_vs_actual_classifier_accuracy.csv", accuracy_rows)

    margin_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for condition, scores in score_map.items():
        margin_rows.extend(_margin_rows(options.protocol, task, condition, test_labels, pre_scores, scores, known_before))
        distribution_rows.extend(_distribution_rows(options.protocol, task, condition, scores, test_labels, known_before))
        group_rows.extend(_group_margin_rows(options.protocol, task, condition, scores, test_labels, known_before))
        calibration_rows.append(
            {
                "protocol": options.protocol,
                "task": task,
                "condition": condition,
                "score_function": "softmax(scale=20 * cosine)",
                "temperature_selection": "none",
                **_calibration(scores, test_labels),
            }
        )
    for condition, scores in (("APR_NCM", apr_ncm_scores * scale), ("ORACLE_NCM", oracle_ncm_scores * scale)):
        calibration_rows.append(
            {
                "protocol": options.protocol,
                "task": task,
                "condition": condition,
                "score_function": "diagnostic softmax(scale=20 * cosine NCM)",
                "temperature_selection": "fixed Structure scale; NCM calibration is auxiliary",
                **_calibration(scores, test_labels),
            }
        )
    write_csv(task_dir / "ca_margin_shift.csv", margin_rows)
    write_csv(task_dir / "score_distribution_summary.csv", distribution_rows)
    write_csv(task_dir / "calibration_metrics.csv", calibration_rows)
    write_csv(task_dir / "old_new_group_margin.csv", group_rows)

    alignment_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    alignment_means = oracle_means.copy()
    for condition, weights in weight_map.items():
        condition_alignment = _alignment_rows(
            options.protocol, task, condition, weights, alignment_means,
            known_before, partition,
        )
        alignment_rows.extend(condition_alignment)
        for row in condition_alignment:
            diagnostic_rows.append(
                {
                    "protocol": options.protocol,
                    "task": task,
                    "condition": condition,
                    "class_id": row["class_id"],
                    "class_group": row["class_group"],
                    "weight_norm": row["weight_norm"],
                    "bias_exists": False,
                    "bias_value": None,
                    "inference_score": "20 * cosine(L2-normalized feature, L2-normalized weight)",
                    "weight_norm_affects_argmax_directly": False,
                }
            )
    write_csv(task_dir / "weight_mean_alignment.csv", alignment_rows)
    write_csv(task_dir / "classifier_weight_diagnostics.csv", diagnostic_rows)
    write_csv(task_dir / "ca_optimization_trace.csv", g_result.trace + r_result.trace)

    betas = np.linspace(-5.0, 5.0, options.bias_points)
    bias_rows: list[dict[str, Any]] = []
    for condition, scores in score_map.items():
        bias_rows.extend(_bias_rows(options.protocol, task, condition, scores, test_labels, known_before, betas))
    write_csv(task_dir / "bias_sweep.csv", bias_rows)

    # Keep tabular task artifacts isolated while collecting figures under the
    # protocol-level directories required by the diagnostic specification.
    _plot_task(options.output_dir, task, bias_rows, score_map, test_labels, known_before, alignment_rows)
    np.savez_compressed(
        task_dir / "diagnostic_head_weights.npz",
        pre_ca=initial_weights.numpy(),
        structure_ca=checkpoint_weights,
        full_oracle_gaussian_ca=d_result.weights,
        g_fixed=g_result.weights,
        r_fixed=r_result.weights,
    )
    manifest = {
        "status": "complete",
        "protocol": options.protocol,
        "task": task,
        "known_before": known_before,
        "total_classes": total_classes,
        "train_samples": int(train_labels.size),
        "test_samples": int(test_labels.size),
        "actual_old_count_min": min(old_counts.values()),
        "actual_old_count_max": max(old_counts.values()),
        "matched_equals_full_actual": full_equals_matched,
        "new_source_identical_G_R": True,
        "new_source": "fixed Structure Gaussian bank",
        "feature_input": "raw frozen sfeatures; L2 normalization inside cosine head",
        "bias_exists": False,
        "existing_D_accuracy_reused": True,
        "D_head_reconstructed_for_logits": True,
        "D_reconstruction_seen_accuracy": d_metrics["seen_class_accuracy"],
        "D_reference_seen_accuracy": reference_d,
        "g_bank_checksum": _checksum(g_features),
        "r_bank_checksum": _checksum(r_features),
        "common_labels_checksum": _checksum(g_labels),
        "elapsed_seconds": time.time() - started,
    }
    write_json(marker, manifest)
    del checkpoint, oracle_covariances, structure_covariances
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run_protocol(options: DiagnosticOptions) -> dict[str, Any]:
    started = time.time()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(options.training_dir)
    selected: list[Path] = []
    for path in checkpoints:
        checkpoint = torch_load(path)
        task = int(checkpoint["cur_task"]) + 1
        del checkpoint
        if task > 1 and (options.tasks is None or task in options.tasks):
            selected.append(path)
    if not selected:
        raise ValueError("no T2+ checkpoint selected")
    before = [_edge_fingerprint(path) for path in selected]
    data_manager, protocol_config = build_data_manager(
        options.protocol, options.data_root, options.seed
    )
    reference_path = options.scientific_dir / "oracle_classifier_replacement.csv"
    reference_rows = read_csv(reference_path)
    if not reference_rows:
        raise FileNotFoundError(reference_path)
    task_manifests = []
    for path in selected:
        print(f"[classifier diagnostic] {options.protocol} {path.name}", flush=True)
        task_manifests.append(
            run_task(options, path, data_manager, protocol_config, reference_rows)
        )
        _aggregate_task_files(options.output_dir)
    after = [_edge_fingerprint(path) for path in selected]
    if before != after:
        raise RuntimeError("main checkpoint changed during classifier diagnostic")
    manifest = {
        "status": "complete",
        "protocol": options.protocol,
        "seed": options.seed,
        "completed_tasks": sorted(int(row["task"]) for row in task_manifests),
        "checkpoint_fingerprints_unchanged": True,
        "scientific_audit_reused": str(options.scientific_dir),
        "training_reused": True,
        "main_checkpoint_modified": False,
        "old_actual_features_used_only_in_diagnostic": True,
        "ia10_car10_executed": False,
        "elapsed_seconds": time.time() - started,
        "task_manifests": task_manifests,
    }
    write_json(options.output_dir / "run_manifest.json", manifest)
    return manifest
