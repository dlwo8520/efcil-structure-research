"""Predeclared within-/cross-task decomposition of the existing CA softmax."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def class_birth_tasks(partition: list[int], device: torch.device | None = None) -> torch.Tensor:
    if not partition or any(int(size) <= 0 for size in partition):
        raise ValueError("task partition must have positive sizes")
    return torch.repeat_interleave(
        torch.arange(len(partition), dtype=torch.long, device=device),
        torch.tensor(partition, dtype=torch.long, device=device),
    )


def loss_components(
    logits: torch.Tensor, targets: torch.Tensor, birth_by_class: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-sample within, cross and all-class CE without changing logits.

    ``G(y)`` is precisely the *birth task's class block*, not all classes seen
    by the birth task.  Log-sum-exp keeps the decomposition stable.
    """
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logit/target dimensions disagree")
    if logits.shape[1] != birth_by_class.numel():
        raise ValueError("birth-task map must cover every seen candidate class")
    if targets.numel() == 0 or targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("targets outside seen class set")
    birth_by_class = birth_by_class.to(logits.device)
    selected_group = birth_by_class[targets]
    same_group = selected_group[:, None] == birth_by_class[None, :]
    target_score = logits.gather(1, targets[:, None]).squeeze(1)
    within_partition = torch.logsumexp(logits.masked_fill(~same_group, float("-inf")), dim=1)
    all_partition = torch.logsumexp(logits, dim=1)
    within = within_partition - target_score
    cross = all_partition - within_partition
    ce = all_partition - target_score
    return within, cross, ce


def assert_ce_identity(logits: torch.Tensor, targets: torch.Tensor,
                       birth_by_class: torch.Tensor, atol: float = 2e-5) -> float:
    within, cross, ce = loss_components(logits, targets, birth_by_class)
    reference = F.cross_entropy(logits, targets, reduction="none")
    max_delta = float(torch.max(torch.abs(within + cross - reference)).detach().cpu())
    if not torch.allclose(within + cross, reference, rtol=0.0, atol=atol):
        raise AssertionError(f"CA CE decomposition failed: max absolute delta={max_delta}")
    if not torch.allclose(ce, reference, rtol=0.0, atol=atol):
        raise AssertionError("direct logsumexp CE differs from PyTorch CE")
    return max_delta


def ca_loss(logits: torch.Tensor, targets: torch.Tensor,
            birth_by_class: torch.Tensor, cross_coefficient: int) -> tuple[torch.Tensor, dict]:
    if cross_coefficient not in (1, 2):
        raise ValueError("only the prespecified coefficients 1 and 2 are allowed")
    within, cross, _ = loss_components(logits, targets, birth_by_class)
    # Use the exact original CE kernel for Control; the identity guard is run
    # before training and again on its first real minibatch.
    loss = (F.cross_entropy(logits, targets) if cross_coefficient == 1
            else (within + 2 * cross).mean())
    return loss, {"within": float(within.detach().mean().cpu()),
                  "cross": float(cross.detach().mean().cpu())}
