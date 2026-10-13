from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from representation_relation.knn import macro_accuracy


@dataclass
class LinearResult:
    weights: np.ndarray
    bias: np.ndarray | None
    trace: list[dict[str, Any]]
    train_micro: float
    train_macro: float
    validation_micro: float
    validation_macro: float
    test_micro: float
    test_macro: float
    train_ce: float
    validation_ce: float
    test_ce: float
    class_balanced_train_ce: float
    objective: float
    weight_norm: float
    gradient_norm: float
    closure_count: int
    converged: bool


class LinearHead(nn.Module):
    def __init__(self, weights: np.ndarray, with_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.as_tensor(weights, dtype=torch.float32).clone())
        self.bias = nn.Parameter(torch.zeros(weights.shape[0])) if with_bias else None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.weight, self.bias)


def _ce(scores: np.ndarray, labels: np.ndarray) -> float:
    shifted = scores - scores.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(labels.size), labels]))


def _metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    predictions = np.argmax(scores, axis=1)
    return (
        float(np.mean(predictions == labels) * 100.0),
        macro_accuracy(predictions, labels),
        _ce(scores, labels),
    )


def class_balanced_objective(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> tuple[float, float]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    per_sample = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(labels.size), labels]
    counts = np.bincount(labels, minlength=scores.shape[1]).astype(np.float64)
    sample_weights = labels.size / (scores.shape[1] * counts[labels])
    balanced_ce = float(np.mean(per_sample * sample_weights))
    return balanced_ce + 0.5 * l2 * float(np.sum(weights ** 2)), balanced_ce


def train_linear_lbfgs(
    *,
    initial_weights: np.ndarray,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    l2: float,
    device: torch.device,
    max_iter: int = 200,
    tolerance_grad: float = 1e-7,
) -> LinearResult:
    model = LinearHead(initial_weights, with_bias=False).to(device)
    optimizer = torch.optim.LBFGS(
        list(model.parameters()),
        lr=1.0,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=tolerance_grad,
        tolerance_change=1e-10,
        history_size=30,
        line_search_fn="strong_wolfe",
    )
    x = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    y = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    counts = np.bincount(train_labels, minlength=initial_weights.shape[0]).astype(np.float64)
    sample_weights = train_labels.size / (initial_weights.shape[0] * counts[train_labels])
    sw = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
    trace: list[dict[str, Any]] = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        data_loss = torch.mean(F.cross_entropy(logits, y, reduction="none") * sw)
        objective = data_loss + 0.5 * l2 * torch.sum(model.weight ** 2)
        objective.backward()
        gradient = float(torch.linalg.vector_norm(model.weight.grad).detach().cpu())
        trace.append(
            {
                "closure": len(trace) + 1,
                "objective": float(objective.detach().cpu()),
                "class_balanced_train_ce": float(data_loss.detach().cpu()),
                "gradient_norm": gradient,
                "weight_norm": float(torch.linalg.vector_norm(model.weight).detach().cpu()),
            }
        )
        return objective

    optimizer.step(closure)
    optimizer.zero_grad(set_to_none=True)
    final_logits = model(x)
    final_data_loss = torch.mean(F.cross_entropy(final_logits, y, reduction="none") * sw)
    final_objective = final_data_loss + 0.5 * l2 * torch.sum(model.weight ** 2)
    final_objective.backward()
    gradient_norm = float(torch.linalg.vector_norm(model.weight.grad).detach().cpu())
    weights = model.weight.detach().cpu().numpy().copy()
    bias = None
    train_scores = train_features @ weights.T
    validation_scores = validation_features @ weights.T
    test_scores = test_features @ weights.T
    train_micro, train_macro, train_ce = _metrics(train_scores, train_labels)
    validation_micro, validation_macro, validation_ce = _metrics(
        validation_scores, validation_labels
    )
    test_micro, test_macro, test_ce = _metrics(test_scores, test_labels)
    objective, balanced_ce = class_balanced_objective(
        train_scores, train_labels, weights, l2
    )
    return LinearResult(
        weights=weights,
        bias=bias,
        trace=trace,
        train_micro=train_micro,
        train_macro=train_macro,
        validation_micro=validation_micro,
        validation_macro=validation_macro,
        test_micro=test_micro,
        test_macro=test_macro,
        train_ce=train_ce,
        validation_ce=validation_ce,
        test_ce=test_ce,
        class_balanced_train_ce=balanced_ce,
        objective=objective,
        weight_norm=float(np.linalg.norm(weights)),
        gradient_norm=gradient_norm,
        closure_count=len(trace),
        converged=gradient_norm <= max(tolerance_grad * 10.0, 1e-6),
    )

