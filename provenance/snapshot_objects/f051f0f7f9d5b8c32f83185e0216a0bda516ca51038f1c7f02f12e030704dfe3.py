"""Gradient and tabular diagnostics shared by follow-up learners."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import torch


def _gradient_list(
    loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]
) -> List[torch.Tensor | None]:
    if not loss.requires_grad:
        return [None for _ in parameters]
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _norm(gradients: Sequence[torch.Tensor | None]) -> torch.Tensor:
    terms = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    if not terms:
        return torch.tensor(0.0)
    return torch.sqrt(torch.stack(terms).sum())


def _cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
    eps: float = 1e-12,
) -> float | None:
    products = []
    left_sq = []
    right_sq = []
    for lhs, rhs in zip(left, right):
        if lhs is None or rhs is None:
            continue
        lhs_f = lhs.detach().float()
        rhs_f = rhs.detach().float()
        products.append((lhs_f * rhs_f).sum())
        left_sq.append(lhs_f.square().sum())
        right_sq.append(rhs_f.square().sum())
    if not products:
        return None
    denominator = torch.sqrt(torch.stack(left_sq).sum() * torch.stack(right_sq).sum())
    if float(denominator) <= eps:
        return None
    return float(torch.stack(products).sum() / denominator)


def component_gradient_metrics(
    losses: Mapping[str, torch.Tensor | None],
    parameters: Iterable[torch.nn.Parameter],
) -> Dict[str, float | None]:
    """Measure component norms and all pairwise cosines without changing .grad."""
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    active = {
        name: loss
        for name, loss in losses.items()
        if loss is not None and isinstance(loss, torch.Tensor)
    }
    gradients = {name: _gradient_list(loss, trainable) for name, loss in active.items()}
    result: Dict[str, float | None] = {}
    for name, values in gradients.items():
        result[f"grad_{name}_norm"] = float(_norm(values))
    names = list(gradients)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            result[f"grad_cosine_{left}_{right}"] = _cosine(
                gradients[left], gradients[right]
            )
    return result


def mean_numeric_rows(rows: Sequence[Mapping[str, float | int | None]]) -> Dict[str, float | None]:
    """Mean numeric non-key columns while ignoring None."""
    keys = sorted({key for row in rows for key in row})
    result: Dict[str, float | None] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        result[key] = sum(values) / len(values) if values else None
    return result
