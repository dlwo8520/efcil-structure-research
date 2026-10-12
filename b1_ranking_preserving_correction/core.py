"""Frozen-logit, task-common correction. No Structure parameter is trainable."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def slices(partition: list[int]) -> list[slice]:
    if not partition or any(int(n) < 1 for n in partition):
        raise ValueError("each task must contain at least one class")
    edges = [0]
    for size in partition:
        edges.append(edges[-1] + int(size))
    return [slice(edges[i], edges[i + 1]) for i in range(len(partition))]


def descriptors(logits: torch.Tensor, partition: list[int]) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != sum(partition):
        raise ValueError("logit shape and task partition disagree")
    maximum, gap, logsum = [], [], []
    for span in slices(partition):
        part = logits[:, span]
        top = torch.topk(part, k=min(2, part.shape[1]), dim=1).values
        maximum.append(top[:, 0])
        gap.append(top[:, 0] - top[:, 1] if part.shape[1] > 1 else torch.zeros_like(top[:, 0]))
        logsum.append(torch.logsumexp(part, dim=1))
    return torch.cat([torch.stack(x, dim=1) for x in (maximum, gap, logsum)], dim=1)


def normalize_fit(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = raw.mean(dim=0)
    std = raw.std(dim=0, unbiased=False).clamp_min(1e-3)
    return ((raw - mean) / std).clamp(-5, 5), mean, std


def normalize_apply(raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return ((raw - mean) / std).clamp(-5, 5)


class Correction(nn.Module):
    def __init__(self, task_n: int, conditional: bool):
        super().__init__()
        self.task_n = task_n
        self.conditional = conditional
        self.b = nn.Parameter(torch.zeros(task_n, dtype=torch.float64))
        if conditional:
            self.A = nn.Parameter(torch.zeros(task_n, 3 * task_n, dtype=torch.float64))

    def offsets(self, phi: torch.Tensor) -> torch.Tensor:
        raw = F.linear(phi, self.A, self.b) if self.conditional else self.b.expand(phi.shape[0], -1)
        return raw - raw.mean(dim=1, keepdim=True)

    def forward(self, logits: torch.Tensor, phi: torch.Tensor, birth_zero: torch.Tensor) -> torch.Tensor:
        return logits + self.offsets(phi)[:, birth_zero]


def objective(model: Correction, logits: torch.Tensor, phi: torch.Tensor,
              labels: torch.Tensor, birth_zero: torch.Tensor,
              lam: float = 1.0, eta: float = 1e-3) -> torch.Tensor:
    offsets = model.offsets(phi)
    adjusted = logits + offsets[:, birth_zero]
    regularization = sum(parameter.square().sum() for parameter in model.parameters())
    return (F.cross_entropy(adjusted, labels) + lam * offsets.square().mean() / 2
            + eta * regularization / 2)


def fit(logits: torch.Tensor, phi: torch.Tensor, labels: torch.Tensor,
        birth_zero: torch.Tensor, conditional: bool) -> tuple[Correction, dict]:
    model = Correction(int(birth_zero.max().item()) + 1, conditional).to(logits.device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=100, max_eval=150,
                                  history_size=20, line_search_fn="strong_wolfe")
    calls = 0
    with torch.no_grad():
        initial = float(objective(model, logits, phi, labels, birth_zero))

    def closure():
        nonlocal calls
        optimizer.zero_grad()
        loss = objective(model, logits, phi, labels, birth_zero)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite L-BFGS objective")
        loss.backward()
        calls += 1
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final = float(objective(model, logits, phi, labels, birth_zero))
    optimizer.zero_grad()
    objective(model, logits, phi, labels, birth_zero).backward()
    gradient = max(float(p.grad.detach().abs().max()) for p in model.parameters())
    model.zero_grad(set_to_none=True)
    state = optimizer.state[next(model.parameters())]
    iterations = int(state.get("n_iter", 0))
    evaluations = int(state.get("func_evals", calls))
    finite = bool(torch.isfinite(torch.tensor([initial, final, gradient])).all())
    if not finite or final > initial + 1e-9:
        raise RuntimeError(f"invalid correction fit: initial={initial}, final={final}, grad={gradient}")
    # PyTorch LBFGS may stop for gradient, step, loss, or budget reasons. A
    # budget hit is inconclusive even if the observed objective improved.
    budget_hit = iterations >= 100 or evaluations >= 150
    return model, {"initial_objective": initial, "final_objective": final,
                   "closure_calls": calls, "iterations": iterations,
                   "function_evaluations": evaluations, "gradient_max_abs": gradient,
                   "budget_hit": budget_hit, "converged": not budget_hit}


def invariant(logits: torch.Tensor, corrected: torch.Tensor, partition: list[int],
              tolerance: float = 1e-8) -> dict:
    if logits.shape != corrected.shape:
        raise AssertionError("correction changed score shape")
    changed = corrected - logits
    disagreements = 0
    maximum = 0.0
    for span in slices(partition):
        disagreements += int((logits[:, span].argmax(1) != corrected[:, span].argmax(1)).sum())
        # Any pairwise score-difference change equals the span of the shift.
        drift = changed[:, span].max(1).values - changed[:, span].min(1).values
        maximum = max(maximum, float(drift.max()))
    if disagreements or maximum > tolerance:
        raise AssertionError(f"within-task invariant failed: {disagreements=} {maximum=}")
    return {"local_argmax_disagreements": disagreements, "within_task_pair_margin_max_abs": maximum}
