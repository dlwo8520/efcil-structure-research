from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from classifier_diagnostic.pipeline import (
    _accuracy_row,
    _assemble_bank,
    _calibration,
    _checksum,
    _class_and_task_accuracy_rows,
    _edge_fingerprint,
    _fixed_gaussian_bank,
    _score_components,
    _stats,
    read_csv,
    write_csv,
    write_json,
)
from oracle_audit.oracle_head import (
    DiagnosticCosineHead,
    checkpoint_classifier_weights,
    cosine_scores,
    load_classifier_initialization,
)
from oracle_audit.runtime import torch_load


CONDITIONS = (
    "FULL_ORACLE_GAUSSIAN_CA",
    "G_FIXED",
    "R_FIXED",
    "G256_FIXED",
    "R256_FIXED",
)


@dataclass
class ProcedureOptions:
    protocol: str
    checkpoint_path: Path
    bottleneck_dir: Path
    scientific_dir: Path
    output_dir: Path
    gpu_device: torch.device
    seed: int = 1993
    epochs: int = 5
    max_converged_epochs: int = 100
    convergence_min_epochs: int = 15
    convergence_patience: int = 12
    relative_loss_threshold: float = 1e-4
    linear_max_iter: int = 100
    samples_per_class: int = 256
    batch_size: int = 256


@dataclass
class TrainResult:
    weights: np.ndarray
    bias: np.ndarray | None
    trace: list[dict[str, Any]]
    selected_epoch: int
    stop_reason: str
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    gradient_norm: float
    weight_update_norm: float
    optimizer_steps: int


def _normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _softmax_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if labels.size == 0:
        return float("nan"), float("nan")
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_sum = np.log(np.exp(shifted).sum(axis=1))
    loss = float(np.mean(log_sum - shifted[np.arange(labels.size), labels]))
    accuracy = float(np.mean(np.argmax(scores, axis=1) == labels) * 100.0)
    return loss, accuracy


def _head_scores(
    features: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> np.ndarray:
    return cosine_scores(features, weights, scale)


def _linear_scores(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    result = np.asarray(features, dtype=np.float32) @ np.asarray(weights, dtype=np.float32).T
    if bias is not None:
        result = result + bias[None, :]
    return result


def _gradient_norm(parameters: list[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2).cpu())
    return math.sqrt(total)


def _make_epoch_permutations(
    labels: np.ndarray,
    epochs: int,
    seed: int,
) -> list[np.ndarray]:
    return [
        np.random.RandomState(seed + epoch).permutation(labels.size).astype(np.int64)
        for epoch in range(epochs)
    ]


def _label_sequence_hash(labels: np.ndarray, permutations: list[np.ndarray]) -> tuple[str, list[str]]:
    total = hashlib.sha256()
    per_epoch = []
    for permutation in permutations:
        sequence = np.ascontiguousarray(labels[permutation], dtype=np.int64)
        digest = hashlib.sha256(sequence.tobytes()).hexdigest()
        per_epoch.append(digest)
        total.update(sequence.tobytes())
    return total.hexdigest(), per_epoch


def _train_cosine(
    *,
    condition: str,
    protocol: str,
    initial_weights: torch.Tensor,
    task_sizes: list[int],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    scale: float,
    init_lr: float,
    weight_decay: float,
    max_epochs: int,
    batch_size: int,
    device: torch.device,
    permutations: list[np.ndarray],
    convergence: bool,
    min_epochs: int,
    patience: int,
    relative_loss_threshold: float,
) -> TrainResult:
    if len(permutations) < max_epochs:
        raise ValueError("not enough precomputed permutations")
    head = DiagnosticCosineHead(initial_weights, task_sizes=task_sizes).to(device)
    parameters = list(head.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=init_lr, momentum=0.9, weight_decay=weight_decay
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for p in parameters}:
        raise AssertionError("cosine optimizer must contain only diagnostic head parameters")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    cpu_features = torch.as_tensor(train_features, dtype=torch.float32)
    cpu_labels = torch.as_tensor(train_labels, dtype=torch.long)
    validation_tensor = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    validation_target = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    initial_numpy = initial_weights.detach().cpu().numpy()
    best_validation = float("inf")
    best_epoch = 0
    best_state: list[torch.Tensor] | None = None
    stale_epochs = 0
    trace: list[dict[str, Any]] = []
    total_steps = 0
    stop_reason = "max_epoch"

    for epoch in range(max_epochs):
        permutation = torch.as_tensor(permutations[epoch], dtype=torch.long)
        before = head.weight.detach().cpu().numpy().copy()
        loss_sum = 0.0
        correct = 0
        samples = 0
        grad_norms: list[float] = []
        for start in range(0, train_labels.size, batch_size):
            indices = permutation[start : start + batch_size]
            inputs = cpu_features[indices].to(device, non_blocking=True)
            targets = cpu_labels[indices].to(device, non_blocking=True)
            logits = scale * head(inputs)
            loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in {condition}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(_gradient_norm(parameters))
            optimizer.step()
            count = int(targets.numel())
            loss_sum += float(loss.detach().cpu()) * count
            correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
            samples += count
            total_steps += 1
        scheduler.step()
        with torch.no_grad():
            if validation_labels.size:
                validation_logits = scale * head(validation_tensor)
                validation_loss = float(
                    F.cross_entropy(validation_logits, validation_target).detach().cpu()
                )
                validation_accuracy = float(
                    (validation_logits.argmax(dim=1) == validation_target).float().mean().cpu()
                    * 100.0
                )
            else:
                validation_loss = float("nan")
                validation_accuracy = float("nan")
        train_loss = loss_sum / samples
        current_weights = head.weight.detach().cpu().numpy().copy()
        trace.append(
            {
                "protocol": protocol,
                "condition": condition,
                "epoch": epoch + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "train_accuracy": 100.0 * correct / samples,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "gradient_norm_mean": float(np.mean(grad_norms)),
                "gradient_norm_last": grad_norms[-1],
                "epoch_weight_update_norm": float(np.linalg.norm(current_weights - before)),
                "optimizer_steps_epoch": math.ceil(samples / batch_size),
                "optimizer_steps_cumulative": total_steps,
                "test_used_for_selection": False,
            }
        )
        selection_value = validation_loss if np.isfinite(validation_loss) else train_loss
        if selection_value < best_validation - 1e-7:
            best_validation = selection_value
            best_epoch = epoch + 1
            best_state = [parameter.detach().cpu().clone() for parameter in head.weights]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if convergence and epoch + 1 >= min_epochs and stale_epochs >= patience:
            lookback = min(5, len(trace) - 1)
            earlier = float(trace[-1 - lookback]["train_loss"])
            relative = max(0.0, earlier - train_loss) / max(abs(earlier), 1e-12)
            if relative < relative_loss_threshold:
                stop_reason = "validation_patience_and_relative_train_loss"
                break

    if convergence and best_state is not None:
        for parameter, selected in zip(head.weights, best_state):
            parameter.data.copy_(selected.to(device))
        selected_epoch = best_epoch
    else:
        selected_epoch = len(trace)
    weights = head.weight.detach().cpu().numpy().copy()
    train_scores = _head_scores(train_features, weights, scale)
    validation_scores = _head_scores(validation_features, weights, scale)
    final_train_loss, final_train_accuracy = _softmax_metrics(train_scores, train_labels)
    final_validation_loss, final_validation_accuracy = _softmax_metrics(
        validation_scores, validation_labels
    )
    return TrainResult(
        weights=weights,
        bias=None,
        trace=trace,
        selected_epoch=selected_epoch,
        stop_reason=stop_reason,
        train_loss=final_train_loss,
        train_accuracy=final_train_accuracy,
        validation_loss=final_validation_loss,
        validation_accuracy=final_validation_accuracy,
        gradient_norm=float(trace[-1]["gradient_norm_last"]),
        weight_update_norm=float(np.linalg.norm(weights - initial_numpy)),
        optimizer_steps=total_steps,
    )


class _LinearProbe(nn.Module):
    def __init__(self, initial_weights: torch.Tensor, with_bias: bool):
        super().__init__()
        self.weight = nn.Parameter(initial_weights.detach().clone().float())
        self.bias = nn.Parameter(torch.zeros(initial_weights.shape[0])) if with_bias else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight, self.bias)


def _train_linear(
    *,
    condition: str,
    protocol: str,
    initial_weights: torch.Tensor,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    with_bias: bool,
    l2: float,
    max_iter: int,
    device: torch.device,
) -> TrainResult:
    model = _LinearProbe(initial_weights, with_bias).to(device)
    parameters = list(model.parameters())
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-5,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    vx = torch.as_tensor(validation_features, dtype=torch.float32, device=device)
    vy = torch.as_tensor(validation_labels, dtype=torch.long, device=device)
    counts = np.bincount(train_labels, minlength=initial_weights.shape[0]).astype(np.float64)
    sample_weight = train_labels.size / (initial_weights.shape[0] * counts[train_labels])
    sample_weight_tensor = torch.as_tensor(sample_weight, dtype=torch.float32, device=device)
    initial_numpy = initial_weights.detach().cpu().numpy()
    trace: list[dict[str, Any]] = []
    closure_index = 0

    def closure():
        nonlocal closure_index
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        per_sample = F.cross_entropy(logits, y, reduction="none")
        data_loss = torch.mean(per_sample * sample_weight_tensor)
        objective = data_loss + 0.5 * l2 * torch.sum(model.weight ** 2)
        objective.backward()
        closure_index += 1
        with torch.no_grad():
            if validation_labels.size:
                val_logits = model(vx)
                val_loss = float(F.cross_entropy(val_logits, vy).cpu())
                val_accuracy = float((val_logits.argmax(1) == vy).float().mean().cpu() * 100.0)
            else:
                val_loss = float("nan")
                val_accuracy = float("nan")
            trace.append(
                {
                    "protocol": protocol,
                    "condition": condition,
                    "iteration": closure_index,
                    "objective": float(objective.detach().cpu()),
                    "train_loss": float(data_loss.detach().cpu()),
                    "train_accuracy": float((logits.argmax(1) == y).float().mean().cpu() * 100.0),
                    "validation_loss": val_loss,
                    "validation_accuracy": val_accuracy,
                    "gradient_norm": _gradient_norm(parameters),
                    "weight_update_norm": float(
                        np.linalg.norm(model.weight.detach().cpu().numpy() - initial_numpy)
                    ),
                    "test_used_for_selection": False,
                }
            )
        return objective

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_logits = model(x)
    final_data_loss = torch.mean(
        F.cross_entropy(final_logits, y, reduction="none") * sample_weight_tensor
    )
    final_objective = final_data_loss + 0.5 * l2 * torch.sum(model.weight ** 2)
    final_objective.backward()
    final_gradient = _gradient_norm(parameters)
    weights = model.weight.detach().cpu().numpy().copy()
    bias = model.bias.detach().cpu().numpy().copy() if model.bias is not None else None
    train_scores = _linear_scores(train_features, weights, bias)
    validation_scores = _linear_scores(validation_features, weights, bias)
    train_loss, train_accuracy = _softmax_metrics(train_scores, train_labels)
    validation_loss, validation_accuracy = _softmax_metrics(validation_scores, validation_labels)
    stop_reason = "lbfgs_solver_tolerance_or_max_iter"
    return TrainResult(
        weights=weights,
        bias=bias,
        trace=trace,
        selected_epoch=len(trace),
        stop_reason=stop_reason,
        train_loss=train_loss,
        train_accuracy=train_accuracy,
        validation_loss=validation_loss,
        validation_accuracy=validation_accuracy,
        gradient_norm=final_gradient,
        weight_update_norm=float(np.linalg.norm(weights - initial_numpy)),
        optimizer_steps=len(trace),
    )


def _split_train_validation(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    no_validation_classes: list[int] = []
    counts: dict[int, dict[str, int]] = {}
    for class_id in class_ids.tolist():
        indices = np.flatnonzero(labels == class_id)
        rng = np.random.RandomState(seed + class_id * 997)
        shuffled = indices[rng.permutation(indices.size)]
        if indices.size >= 5:
            validation_count = max(1, int(math.floor(indices.size * 0.2)))
            validation_count = min(validation_count, indices.size - 2)
        else:
            validation_count = 0
            no_validation_classes.append(class_id)
        validation_indices.extend(shuffled[:validation_count].tolist())
        train_indices.extend(shuffled[validation_count:].tolist())
        counts[class_id] = {
            "total": int(indices.size),
            "train": int(indices.size - validation_count),
            "validation": int(validation_count),
        }
    train_indices_array = np.asarray(sorted(train_indices), dtype=np.int64)
    validation_indices_array = np.asarray(sorted(validation_indices), dtype=np.int64)
    metadata = {
        "validation_policy": "deterministic per-class 20% if count>=5; keep at least 2 train",
        "validation_classes": len(class_ids) - len(no_validation_classes),
        "no_validation_classes": no_validation_classes,
        "class_counts": counts,
    }
    return (
        features[train_indices_array],
        labels[train_indices_array],
        features[validation_indices_array],
        labels[validation_indices_array],
        metadata,
    )


def _replacement_balanced_bank(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    draws_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    rng = np.random.RandomState(seed)
    banks: dict[int, np.ndarray] = {}
    for class_id in class_ids.tolist():
        values = features[labels == class_id]
        if not values.size:
            raise RuntimeError(f"missing train values for class {class_id}")
        selected = rng.randint(0, values.shape[0], size=draws_per_class)
        banks[class_id] = values[selected].copy()
    bank_features, bank_labels = _assemble_bank(banks, class_ids)
    return bank_features, bank_labels, banks


def _add_group_balance(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("row_type") == "summary":
        row["group_balance"] = 0.5 * (
            float(row["old_accuracy"]) + float(row["new_accuracy"])
        )
    return row


def _margin_rows(
    protocol: str,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    known_before: int,
) -> list[dict[str, Any]]:
    components = _score_components(scores, labels, known_before)
    rows = []
    for index, label in enumerate(labels.tolist()):
        rows.append(
            {
                "protocol": protocol,
                "task": 5,
                "condition": condition,
                "sample_id": f"{protocol}_T05_test_{index:06d}",
                "true_class": label,
                "sample_group": "old" if label < known_before else "new",
                "true_score": components["true"][index],
                "highest_same_group_wrong_score": components["same"][index],
                "highest_opposite_group_score": components["opposite"][index],
                "internal_margin": components["internal_margin"][index],
                "opposite_group_margin": components["opposite_margin"][index],
            }
        )
    return rows


def _summary_accuracy_rows(
    protocol: str,
    condition: str,
    scores: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    known_before: int,
    partition: list[int],
    **extra: Any,
) -> list[dict[str, Any]]:
    base = {"protocol": protocol, "task": 5, "condition": condition, **extra}
    rows = [
        _add_group_balance(
            _accuracy_row(
                protocol,
                5,
                condition,
                scores,
                labels,
                class_ids,
                known_before,
                **extra,
            )
        )
    ]
    rows.extend(_class_and_task_accuracy_rows(base, scores, labels, class_ids, partition))
    return rows


def _sampling_rows(
    protocol: str,
    total_classes: int,
    known_before: int,
    actual_counts: np.ndarray,
    epochs: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        dynamic = condition == "FULL_ORACLE_GAUSSIAN_CA"
        is_256 = condition in (
            "FULL_ORACLE_GAUSSIAN_CA",
            "G256_FIXED",
            "R256_FIXED",
        )
        exposures = np.full(total_classes, 256, dtype=np.int64)
        source_unique = np.full(total_classes, 256, dtype=np.int64)
        if condition in ("G_FIXED", "R_FIXED"):
            exposures[:known_before] = actual_counts[:known_before]
            source_unique[:known_before] = actual_counts[:known_before]
        elif condition == "R256_FIXED":
            source_unique[:known_before] = actual_counts[:known_before]
        per_epoch = int(exposures.sum())
        steps_per_epoch = math.ceil(per_epoch / batch_size)
        old = exposures[:known_before]
        new = exposures[known_before:]
        audit_rows.append(
            {
                "protocol": protocol,
                "task": 5,
                "condition": condition,
                "old_draws_per_class_min_epoch": int(old.min()),
                "old_draws_per_class_mean_epoch": float(old.mean()),
                "old_draws_per_class_max_epoch": int(old.max()),
                "new_draws_per_class_min_epoch": int(new.min()),
                "new_draws_per_class_mean_epoch": float(new.mean()),
                "new_draws_per_class_max_epoch": int(new.max()),
                "old_total_draw_count": int(old.sum() * epochs),
                "new_total_draw_count": int(new.sum() * epochs),
                "total_draws": per_epoch * epochs,
                "draws_per_epoch": per_epoch,
                "batch_size": batch_size,
                "optimizer_steps_per_epoch": steps_per_epoch,
                "optimizer_steps": steps_per_epoch * epochs,
                "epochs": epochs,
                "bank_lifecycle": "dynamic_regenerated_each_epoch" if dynamic else "fixed_repeated",
                "shuffle_rule": "global permutation then contiguous batches",
                "sampler": "none; explicit tensor permutation",
                "replacement": "Gaussian iid" if condition.startswith("G") or dynamic else "actual replacement" if condition == "R256_FIXED" else "none",
                "lr_schedule": f"CosineAnnealingLR(T_max={epochs}), epoch step",
                "balanced_256_per_class": is_256,
            }
        )
        for class_id in range(total_classes):
            class_rows.append(
                {
                    "protocol": protocol,
                    "task": 5,
                    "condition": condition,
                    "class_id": class_id,
                    "class_group": "old" if class_id < known_before else "new",
                    "source_unique_count": int(source_unique[class_id]),
                    "draws_per_epoch": int(exposures[class_id]),
                    "total_draws": int(exposures[class_id] * epochs),
                    "fixed_dynamic": "dynamic" if dynamic else "fixed",
                }
            )
    return audit_rows, class_rows


def _plot_outputs(
    output_dir: Path,
    convergence_rows: list[dict[str, Any]],
    exposure_rows: list[dict[str, Any]],
    probe_summary: list[dict[str, Any]],
    margin_rows: list[dict[str, Any]],
) -> None:
    for directory in (
        "cosine_convergence",
        "old_new_margin",
        "classifier_comparison",
        "class_exposure",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in ("COSINE_5EP", "COSINE_CONVERGED"):
        rows = [row for row in convergence_rows if row["condition"] == condition]
        axes[0].plot([row["epoch"] for row in rows], [row["train_loss"] for row in rows], label=condition)
        axes[1].plot([row["epoch"] for row in rows], [row["validation_accuracy"] for row in rows], label=condition)
    axes[0].set(title="Train CE", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Validation accuracy", xlabel="epoch", ylabel="accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "cosine_convergence" / "cosine_convergence.png", dpi=160)
    plt.close(figure)

    summary = [row for row in probe_summary if row.get("row_type") == "summary"]
    names = [row["condition"] for row in summary]
    x = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.25
    for offset, key, label in ((-width, "seen_class_accuracy", "Seen"), (0, "old_accuracy", "Old"), (width, "new_accuracy", "New")):
        axis.bar(x + offset, [float(row[key]) for row in summary], width, label=label)
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylabel("accuracy (%)")
    axis.set_title("Classifier probe comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "classifier_comparison" / "classifier_accuracy.png", dpi=160)
    plt.close(figure)

    audit_summary = [row for row in exposure_rows if "old_total_draw_count" in row]
    figure, axis = plt.subplots(figsize=(10, 5))
    names = [row["condition"] for row in audit_summary]
    old = [row["old_total_draw_count"] for row in audit_summary]
    new = [row["new_total_draw_count"] for row in audit_summary]
    axis.bar(names, old, label="Old")
    axis.bar(names, new, bottom=old, label="New")
    axis.set_ylabel("total exposure")
    axis.set_title("Class exposure by condition")
    axis.tick_params(axis="x", rotation=20)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "class_exposure" / "class_exposure.png", dpi=160)
    plt.close(figure)

    conditions = [row["condition"] for row in summary]
    old_values = [
        [float(row["opposite_group_margin"]) for row in margin_rows if row["condition"] == condition and row["sample_group"] == "old"]
        for condition in conditions
    ]
    new_values = [
        [float(row["opposite_group_margin"]) for row in margin_rows if row["condition"] == condition and row["sample_group"] == "new"]
        for condition in conditions
    ]
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].boxplot(old_values, labels=conditions, showfliers=False)
    axes[0].set_title("Old-vs-New margin")
    axes[1].boxplot(new_values, labels=conditions, showfliers=False)
    axes[1].set_title("New-vs-Old margin")
    axes[1].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "old_new_margin" / "opposite_group_margin.png", dpi=160)
    plt.close(figure)


def run_protocol(options: ProcedureOptions) -> dict[str, Any]:
    started = time.time()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    before = _edge_fingerprint(options.checkpoint_path)
    checkpoint = torch_load(options.checkpoint_path)
    if int(checkpoint["cur_task"]) + 1 != 5:
        raise ValueError("procedure diagnostic must use T5 checkpoint")
    total_classes = int(checkpoint["total_classes"])
    partition = list(map(int, checkpoint["task_sizes"]))
    known_before = total_classes - partition[-1]
    class_ids = np.arange(total_classes, dtype=np.int64)
    cache_path = options.bottleneck_dir / ".cache" / "task_05_features.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"validated T5 feature cache required: {cache_path}")
    cached = np.load(cache_path, allow_pickle=False)
    train_features = np.asarray(cached["train_features"], dtype=np.float32)
    train_labels = np.asarray(cached["train_labels"], dtype=np.int64)
    test_features = np.asarray(cached["test_features"], dtype=np.float32)
    test_labels = np.asarray(cached["test_labels"], dtype=np.int64)
    actual_counts = np.bincount(train_labels, minlength=total_classes)
    oracle_means, oracle_covariances, actual_banks = _stats(
        train_features,
        train_labels,
        class_ids,
        float(checkpoint["config"].get("stats_cov_jitter", 1e-3)),
    )
    structure_means = np.asarray(checkpoint["class_means"][:total_classes], dtype=np.float64)
    structure_covariances = np.stack(
        [value.detach().cpu().numpy() for value in checkpoint["class_covs"][:total_classes]]
    )
    initial_weights, sidecar, initialization_source = load_classifier_initialization(
        checkpoint, options.checkpoint_path, class_ids, 5
    )
    if sidecar is None or initialization_source != "exact_pre_ca_sidecar":
        raise RuntimeError("exact pre-CA sidecar is required")
    scale = float(sidecar["scale"])
    init_lr = float(sidecar["init_lr"])
    weight_decay = float(sidecar["weight_decay"])
    task_size_argument = int(sidecar["task_size_argument"])
    task_sizes = list(map(int, sidecar["task_sizes"]))

    audit_rows, exposure_rows = _sampling_rows(
        options.protocol,
        total_classes,
        known_before,
        actual_counts,
        options.epochs,
        options.batch_size,
    )
    write_csv(options.output_dir / "ca_sampling_audit.csv", audit_rows)
    write_csv(options.output_dir / "class_exposure_summary.csv", exposure_rows)

    old_counts_256 = {class_id: options.samples_per_class for class_id in range(known_before)}
    new_counts_256 = {
        class_id: options.samples_per_class for class_id in range(known_before, total_classes)
    }
    g_old = _fixed_gaussian_bank(
        oracle_means,
        oracle_covariances,
        old_counts_256,
        4,
        task_size_argument,
        options.gpu_device,
        options.seed + 5101,
    )
    common_new = _fixed_gaussian_bank(
        structure_means,
        structure_covariances,
        new_counts_256,
        4,
        task_size_argument,
        options.gpu_device,
        options.seed + 5102,
    )
    rng = np.random.RandomState(options.seed + 5103)
    r_old: dict[int, np.ndarray] = {}
    for class_id in range(known_before):
        values = actual_banks[class_id]
        selected = rng.randint(0, values.shape[0], size=options.samples_per_class)
        r_old[class_id] = values[selected].copy()
    g_banks = {**g_old, **common_new}
    r_banks = {**r_old, **common_new}
    g_features, g_labels = _assemble_bank(g_banks, class_ids)
    r_features, r_labels = _assemble_bank(r_banks, class_ids)
    if not np.array_equal(g_labels, r_labels):
        raise AssertionError("G256/R256 label layout mismatch")
    for class_id in range(known_before, total_classes):
        if not np.array_equal(g_banks[class_id], r_banks[class_id]):
            raise AssertionError("G256/R256 current-new bank mismatch")
    permutations = _make_epoch_permutations(
        g_labels, options.epochs, options.seed + 5200
    )
    label_hash, epoch_hashes = _label_sequence_hash(g_labels, permutations)
    expected_steps = options.epochs * total_classes
    paired_manifest = {
        "protocol": options.protocol,
        "task": 5,
        "seed": options.seed,
        "permutation_seed_base": options.seed + 5200,
        "label_sequence_hash": label_hash,
        "epoch_label_sequence_hashes": epoch_hashes,
        "total_draws": int(g_labels.size * options.epochs),
        "draws_per_epoch": int(g_labels.size),
        "class_draw_counts": {
            str(class_id): options.samples_per_class * options.epochs
            for class_id in class_ids.tolist()
        },
        "batch_size": options.batch_size,
        "optimizer_steps_per_epoch": total_classes,
        "optimizer_steps": expected_steps,
        "g_labels_checksum": _checksum(g_labels),
        "r_labels_checksum": _checksum(r_labels),
        "g_new_bank_checksum": _checksum(
            np.concatenate([g_banks[class_id] for class_id in range(known_before, total_classes)])
        ),
        "r_new_bank_checksum": _checksum(
            np.concatenate([r_banks[class_id] for class_id in range(known_before, total_classes)])
        ),
        "new_bank_identical": True,
        "feature_source_only_difference": True,
    }
    if paired_manifest["g_new_bank_checksum"] != paired_manifest["r_new_bank_checksum"]:
        raise AssertionError("paired new-bank checksum differs")
    write_json(options.output_dir / "paired_sampling_manifest.json", paired_manifest)

    common_pair = dict(
        protocol=options.protocol,
        initial_weights=initial_weights,
        task_sizes=task_sizes,
        validation_features=np.empty((0, train_features.shape[1]), dtype=np.float32),
        validation_labels=np.empty(0, dtype=np.int64),
        scale=scale,
        init_lr=init_lr,
        weight_decay=weight_decay,
        max_epochs=options.epochs,
        batch_size=options.batch_size,
        device=options.gpu_device,
        permutations=permutations,
        convergence=False,
        min_epochs=options.epochs,
        patience=options.epochs,
        relative_loss_threshold=options.relative_loss_threshold,
    )
    g256 = _train_cosine(
        condition="G256_FIXED", train_features=g_features, train_labels=g_labels, **common_pair
    )
    r256 = _train_cosine(
        condition="R256_FIXED", train_features=r_features, train_labels=r_labels, **common_pair
    )
    if g256.optimizer_steps != expected_steps or r256.optimizer_steps != expected_steps:
        raise AssertionError("paired optimizer-step budget mismatch")
    g256_scores = _head_scores(test_features, g256.weights, scale)
    r256_scores = _head_scores(test_features, r256.weights, scale)

    prior_accuracy_path = options.bottleneck_dir / "gaussian_vs_actual_classifier_accuracy.csv"
    prior_rows = [
        row for row in read_csv(prior_accuracy_path)
        if int(row["task"]) == 5 and row["condition"] in CONDITIONS[:3]
    ]
    if not prior_rows:
        raise FileNotFoundError(prior_accuracy_path)
    accuracy_rows: list[dict[str, Any]] = []
    for row in prior_rows:
        converted: dict[str, Any] = dict(row)
        if row.get("row_type") == "summary":
            converted["group_balance"] = 0.5 * (
                float(row["old_accuracy"]) + float(row["new_accuracy"])
            )
        accuracy_rows.append(converted)
    accuracy_rows.extend(
        _summary_accuracy_rows(
            options.protocol,
            "G256_FIXED",
            g256_scores,
            test_labels,
            class_ids,
            known_before,
            partition,
            old_source="fixed oracle Gaussian, 256/class",
            new_source="fixed Structure Gaussian, 256/class",
        )
    )
    accuracy_rows.extend(
        _summary_accuracy_rows(
            options.protocol,
            "R256_FIXED",
            r256_scores,
            test_labels,
            class_ids,
            known_before,
            partition,
            old_source="actual replacement, 256/class",
            new_source="fixed Structure Gaussian, 256/class",
        )
    )
    write_csv(options.output_dir / "g256_r256_accuracy.csv", accuracy_rows)

    normalized_train = _normalize(train_features)
    normalized_test = _normalize(test_features)
    split = _split_train_validation(
        normalized_train, train_labels, class_ids, options.seed + 5300
    )
    probe_train_features, probe_train_labels, validation_features, validation_labels, split_metadata = split
    balanced_features, balanced_labels, _ = _replacement_balanced_bank(
        probe_train_features,
        probe_train_labels,
        class_ids,
        options.samples_per_class,
        options.seed + 5400,
    )
    probe_permutations_5 = _make_epoch_permutations(
        balanced_labels, options.epochs, options.seed + 5500
    )
    probe_permutations_converged = _make_epoch_permutations(
        balanced_labels, options.max_converged_epochs, options.seed + 5500
    )
    common_probe = dict(
        protocol=options.protocol,
        initial_weights=initial_weights,
        task_sizes=task_sizes,
        train_features=balanced_features,
        train_labels=balanced_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        scale=scale,
        init_lr=init_lr,
        weight_decay=weight_decay,
        batch_size=options.batch_size,
        device=options.gpu_device,
        relative_loss_threshold=options.relative_loss_threshold,
    )
    cosine_5 = _train_cosine(
        condition="COSINE_5EP",
        max_epochs=options.epochs,
        permutations=probe_permutations_5,
        convergence=False,
        min_epochs=options.epochs,
        patience=options.epochs,
        **common_probe,
    )
    cosine_converged = _train_cosine(
        condition="COSINE_CONVERGED",
        max_epochs=options.max_converged_epochs,
        permutations=probe_permutations_converged,
        convergence=True,
        min_epochs=options.convergence_min_epochs,
        patience=options.convergence_patience,
        **common_probe,
    )
    linear_no_bias = _train_linear(
        condition="LINEAR_NO_BIAS",
        protocol=options.protocol,
        initial_weights=initial_weights,
        train_features=probe_train_features,
        train_labels=probe_train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        with_bias=False,
        l2=weight_decay,
        max_iter=options.linear_max_iter,
        device=options.gpu_device,
    )
    linear_with_bias = _train_linear(
        condition="LINEAR_WITH_BIAS",
        protocol=options.protocol,
        initial_weights=initial_weights,
        train_features=probe_train_features,
        train_labels=probe_train_labels,
        validation_features=validation_features,
        validation_labels=validation_labels,
        with_bias=True,
        l2=weight_decay,
        max_iter=options.linear_max_iter,
        device=options.gpu_device,
    )

    checkpoint_weights = checkpoint_classifier_weights(checkpoint).numpy()[:total_classes]
    probe_results = {
        "COSINE_5EP": cosine_5,
        "COSINE_CONVERGED": cosine_converged,
        "LINEAR_NO_BIAS": linear_no_bias,
        "LINEAR_WITH_BIAS": linear_with_bias,
    }
    probe_scores = {
        "STRUCTURE_CA": _head_scores(normalized_test, checkpoint_weights, scale),
        "COSINE_5EP": _head_scores(normalized_test, cosine_5.weights, scale),
        "COSINE_CONVERGED": _head_scores(normalized_test, cosine_converged.weights, scale),
        "LINEAR_NO_BIAS": _linear_scores(normalized_test, linear_no_bias.weights, None),
        "LINEAR_WITH_BIAS": _linear_scores(
            normalized_test, linear_with_bias.weights, linear_with_bias.bias
        ),
    }
    probe_rows: list[dict[str, Any]] = []
    for condition, scores in probe_scores.items():
        result = probe_results.get(condition)
        extra = {
            "classifier": "cosine scale=20" if condition.startswith("COSINE") or condition == "STRUCTURE_CA" else "linear",
            "bias": condition == "LINEAR_WITH_BIAS",
            "train_accuracy": result.train_accuracy if result else None,
            "validation_accuracy": result.validation_accuracy if result else None,
            "final_training_ce": result.train_loss if result else None,
            "final_validation_ce": result.validation_loss if result else None,
            "gradient_norm": result.gradient_norm if result else None,
            "weight_update_norm": result.weight_update_norm if result else None,
            "selected_epoch_or_iterations": result.selected_epoch if result else None,
            "stop_reason": result.stop_reason if result else "existing checkpoint",
        }
        probe_rows.extend(
            _summary_accuracy_rows(
                options.protocol,
                condition,
                scores,
                test_labels,
                class_ids,
                known_before,
                partition,
                **extra,
            )
        )
    write_csv(options.output_dir / "classifier_probe_accuracy.csv", probe_rows)

    convergence_rows = cosine_5.trace + cosine_converged.trace
    write_csv(options.output_dir / "cosine_convergence_trace.csv", convergence_rows)
    write_csv(
        options.output_dir / "classifier_optimization_trace.csv",
        convergence_rows + linear_no_bias.trace + linear_with_bias.trace,
    )
    margin_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for condition, scores in probe_scores.items():
        margin_rows.extend(
            _margin_rows(options.protocol, condition, scores, test_labels, known_before)
        )
        calibration_rows.append(
            {
                "protocol": options.protocol,
                "task": 5,
                "condition": condition,
                "score_function": "scale20_cosine" if condition.startswith("COSINE") or condition == "STRUCTURE_CA" else "linear_logits",
                "test_used_for_selection": False,
                **_calibration(scores, test_labels),
            }
        )
    write_csv(options.output_dir / "classifier_margin_comparison.csv", margin_rows)
    write_csv(options.output_dir / "classifier_calibration.csv", calibration_rows)
    _plot_outputs(options.output_dir, convergence_rows, audit_rows, probe_rows, margin_rows)

    np.savez_compressed(
        options.output_dir / "diagnostic_classifier_weights.npz",
        g256=g256.weights,
        r256=r256.weights,
        cosine_5ep=cosine_5.weights,
        cosine_converged=cosine_converged.weights,
        linear_no_bias=linear_no_bias.weights,
        linear_with_bias=linear_with_bias.weights,
        linear_bias=linear_with_bias.bias,
    )
    after = _edge_fingerprint(options.checkpoint_path)
    if before != after:
        raise RuntimeError("main Structure checkpoint fingerprint changed")
    manifest = {
        "status": "complete",
        "protocol": options.protocol,
        "task": 5,
        "seed": options.seed,
        "checkpoint": str(options.checkpoint_path),
        "checkpoint_fingerprint_before": before,
        "checkpoint_fingerprint_after": after,
        "checkpoint_unchanged": True,
        "training_reused": True,
        "backbone_adapter_updated": False,
        "diagnostic_only": True,
        "train_samples": int(train_labels.size),
        "validation_samples": int(validation_labels.size),
        "test_samples": int(test_labels.size),
        "validation_classes": split_metadata["validation_classes"],
        "no_validation_classes": split_metadata["no_validation_classes"],
        "g256_r256_optimizer_steps": expected_steps,
        "paired_label_sequence_hash": label_hash,
        "new_bank_identical": True,
        "cosine_5ep_epochs": len(cosine_5.trace),
        "cosine_converged_epochs_run": len(cosine_converged.trace),
        "cosine_converged_selected_epoch": cosine_converged.selected_epoch,
        "cosine_converged_stop_reason": cosine_converged.stop_reason,
        "linear_no_bias_closures": len(linear_no_bias.trace),
        "linear_with_bias_closures": len(linear_with_bias.trace),
        "t2_t4_executed": False,
        "ia10_car10_executed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_dir / "run_manifest.json", manifest)
    del checkpoint, oracle_covariances, structure_covariances, g_features, r_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def copy_sampling_audit(source: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_root / "ca_sampling_audit.md")

