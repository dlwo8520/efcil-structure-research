from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right

import torch
from torch.nn import functional as F

from .model import ContinualViT


@dataclass
class EvaluationResult:
    seen_accuracy: float
    class_accuracy: list[float]
    class_correct: list[int]
    class_count: list[int]
    prototypes: torch.Tensor


@torch.no_grad()
def evaluate_seen(
    model: ContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    seen_classes: int,
) -> EvaluationResult:
    model.eval()
    correct = torch.zeros(seen_classes, dtype=torch.long)
    count = torch.zeros(seen_classes, dtype=torch.long)
    feature_sum = torch.zeros(seen_classes, model.feature_dim, dtype=torch.float64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            output = model(images)
        predictions = output["logits"].argmax(dim=1)
        labels_cpu = labels.long().cpu()
        predictions_cpu = predictions.long().cpu()
        features_cpu = output["features"].detach().double().cpu()
        count.index_add_(0, labels_cpu, torch.ones_like(labels_cpu))
        correct.index_add_(0, labels_cpu, predictions_cpu.eq(labels_cpu).long())
        feature_sum.index_add_(0, labels_cpu, features_cpu)
    if int(count.sum()) == 0 or bool((count == 0).any()):
        raise RuntimeError("Evaluation did not observe every seen class")
    class_accuracy = correct.double() / count.double()
    prototypes = F.normalize((feature_sum / count[:, None]).float(), dim=1)
    return EvaluationResult(
        seen_accuracy=float(correct.sum().item() / count.sum().item()),
        class_accuracy=class_accuracy.tolist(),
        class_correct=correct.tolist(),
        class_count=count.tolist(),
        prototypes=prototypes,
    )


def summarize_task_metrics(
    task_id: int,
    classes_per_task: int | list[int] | tuple[int, ...],
    evaluation: EvaluationResult,
    task_accuracy_history: list[list[float]],
) -> tuple[dict, list[float]]:
    task_sizes = _resolved_task_sizes(classes_per_task, task_id + 1)
    offsets = _task_offsets(task_sizes)
    task_accuracies: list[float] = []
    for learned_task in range(task_id + 1):
        begin, end = offsets[learned_task], offsets[learned_task + 1]
        task_accuracies.append(
            sum(evaluation.class_correct[begin:end])
            / sum(evaluation.class_count[begin:end])
        )
    new_begin = offsets[task_id]
    old_accuracy = (
        sum(evaluation.class_correct[:new_begin])
        / sum(evaluation.class_count[:new_begin])
        if new_begin
        else float("nan")
    )
    task_forgetting: list[float] = []
    for learned_task in range(task_id):
        # The accuracy matrix is lower triangular: task j does not exist in
        # evaluation rows earlier than j.
        best = max(
            row[learned_task]
            for row in task_accuracy_history
            if learned_task < len(row)
        )
        task_forgetting.append(best - task_accuracies[learned_task])
    average_forgetting = (
        sum(task_forgetting) / len(task_forgetting)
        if task_forgetting
        else 0.0
    )
    row = {
        "task_id": task_id,
        "seen_classes": offsets[-1],
        "seen_accuracy": evaluation.seen_accuracy,
        "old_accuracy": old_accuracy,
        "new_accuracy": task_accuracies[-1],
        "average_forgetting": average_forgetting,
    }
    return row, task_accuracies


def build_class_rows(
    task_id: int,
    classes_per_task: int | list[int] | tuple[int, ...],
    evaluation: EvaluationResult,
    class_accuracy_history: list[list[float]],
    reference_prototypes: dict[int, torch.Tensor],
) -> list[dict]:
    rows: list[dict] = []
    task_sizes = _resolved_task_sizes(classes_per_task, task_id + 1)
    offsets = _task_offsets(task_sizes)
    new_begin = offsets[task_id]
    for class_id, accuracy in enumerate(evaluation.class_accuracy):
        previous = [row[class_id] for row in class_accuracy_history if class_id < len(row)]
        best = max(previous + [accuracy])
        prototype = evaluation.prototypes[class_id]
        reference = reference_prototypes.get(class_id)
        if reference is None:
            reference = prototype.detach().cpu()
            reference_prototypes[class_id] = reference
        cosine_drift = float(1.0 - torch.dot(reference, prototype.cpu()).item())
        l2_drift = float(torch.linalg.vector_norm(reference - prototype.cpu()).item())
        rows.append({
            "task_id": task_id,
            "class_id": class_id,
            "learned_task": bisect_right(offsets, class_id) - 1,
            "is_new": int(class_id >= new_begin),
            "accuracy": accuracy,
            "best_accuracy": best,
            "forgetting": best - accuracy,
            "prototype_cosine_drift": cosine_drift,
            "prototype_l2_drift": l2_drift,
            "correct": evaluation.class_correct[class_id],
            "count": evaluation.class_count[class_id],
        })
    return rows


def _resolved_task_sizes(
    value: int | list[int] | tuple[int, ...], learned_tasks: int
) -> list[int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("classes_per_task must be positive")
        return [value] * learned_tasks
    sizes = [int(item) for item in value]
    if len(sizes) < learned_tasks or any(item <= 0 for item in sizes):
        raise ValueError("task_sizes must contain a positive size for every task")
    return sizes[:learned_tasks]


def _task_offsets(task_sizes: list[int]) -> list[int]:
    offsets = [0]
    for size in task_sizes:
        offsets.append(offsets[-1] + size)
    return offsets
