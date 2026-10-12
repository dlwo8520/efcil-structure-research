"""CEOS boundary sampler isolated from classifier retraining.

ORIGINAL (CEOS, CVPR Findings 2026): choose nearest current-class enemy
features, interpolate from an old prototype with lambda in (tau, 1), tau >
0.5, and L2-normalize the result.

ADAPTED FOR STRUCTURE: the starting old prototype is Structure's APR-corrected
class mean.  The caller controls the Gaussian/CEOS mix while preserving the
same total replay count per class.  ACB is intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F


@dataclass
class CEOSBatch:
    features: torch.Tensor
    enemy_labels: torch.Tensor
    lambdas: torch.Tensor
    distance_from_old: torch.Tensor
    distance_to_enemy: torch.Tensor
    cosine_to_old: torch.Tensor
    cosine_to_enemy: torch.Tensor
    old_dominance: torch.Tensor


def sample_ceos(
    old_prototype: torch.Tensor,
    current_features: torch.Tensor,
    current_labels: torch.Tensor,
    count: int,
    *,
    lambda_min: float = 0.7,
    enemy_top_k: int = 1,
) -> CEOSBatch:
    """Sample CEOS points for one old class using current-task features only."""
    if old_prototype.ndim != 1:
        raise ValueError("old_prototype must be [D]")
    if current_features.ndim != 2 or current_features.shape[1] != old_prototype.shape[0]:
        raise ValueError("current_features must be [N, D] with matching D")
    if current_features.shape[0] != current_labels.numel() or current_features.shape[0] == 0:
        raise ValueError("current feature/label lengths differ or are empty")
    if count < 0 or enemy_top_k <= 0:
        raise ValueError("count must be non-negative and enemy_top_k positive")
    if not 0.5 < lambda_min < 1.0:
        raise ValueError("CEOS lambda_min must satisfy 0.5 < lambda_min < 1")
    if count == 0:
        empty_float = current_features.new_empty((0,))
        empty_long = current_labels.new_empty((0,), dtype=torch.long)
        return CEOSBatch(
            current_features.new_empty((0, old_prototype.numel())),
            empty_long,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            empty_float,
            torch.empty((0,), device=current_features.device, dtype=torch.bool),
        )

    # Structure's classifier is cosine-normalized.  Normalize both endpoints
    # before nearest-enemy search/interpolation so CEOS's dominance constraint
    # is evaluated in the same angular feature space used by that classifier.
    prototype = F.normalize(old_prototype.to(current_features).float(), dim=0)
    enemies = F.normalize(current_features.float(), dim=1)
    distances = torch.linalg.vector_norm(enemies - prototype[None, :], dim=1)
    k = min(int(enemy_top_k), enemies.shape[0])
    nearest_indices = distances.topk(k, largest=False).indices
    choices = nearest_indices[
        torch.randint(0, k, (int(count),), device=current_features.device)
    ]
    selected = enemies[choices]
    selected_labels = current_labels.to(current_features.device).long()[choices]
    lambdas = float(lambda_min) + (1.0 - float(lambda_min)) * torch.rand(
        int(count), device=current_features.device
    )
    mixed_raw = lambdas[:, None] * prototype[None, :] + (1.0 - lambdas[:, None]) * selected
    generated = F.normalize(mixed_raw, dim=1)
    old_unit = prototype[None, :].expand_as(generated)
    enemy_unit = selected
    distance_from_old = torch.linalg.vector_norm(generated - old_unit, dim=1)
    distance_to_enemy = torch.linalg.vector_norm(generated - enemy_unit, dim=1)
    return CEOSBatch(
        features=generated,
        enemy_labels=selected_labels,
        lambdas=lambdas,
        distance_from_old=distance_from_old,
        distance_to_enemy=distance_to_enemy,
        cosine_to_old=(generated * old_unit).sum(dim=1),
        cosine_to_enemy=(generated * enemy_unit).sum(dim=1),
        old_dominance=distance_from_old < distance_to_enemy,
    )
