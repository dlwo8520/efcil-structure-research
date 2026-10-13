"""Pair-selective AM-Drift losses used by C0--C4."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from similarity.losses import RelationLossResult, pairwise_diagonal_distance


def selective_control_relation(features, cache, sample_pair_weights, eps=1e-8):
    """Apply a per-sample/per-old-class weight to AM-Drift.

    Structure's E3 implementation uses the maximum dimensionless intrusion
    ``max_k [1 - D2_k / tau_k]+``.  Weighting every pair by one therefore
    reproduces the inherited AM-Drift implementation exactly.
    """
    if getattr(cache, "covariance_view", None) != "diagonal":
        raise ValueError("control loss requires a diagonal SeparationCache")
    distances = pairwise_diagonal_distance(features, cache, eps=eps)
    taus = cache.taus.to(features.device).float().clamp_min(eps)[None, :]
    taus = taus.expand_as(distances)
    weights = torch.as_tensor(sample_pair_weights, device=features.device).float()
    if weights.shape != distances.shape:
        raise ValueError(
            "sample_pair_weights must have shape [batch, old regions], "
            f"got {tuple(weights.shape)} vs {tuple(distances.shape)}"
        )
    if not torch.isfinite(weights).all() or (weights < 0).any() or (weights > 1).any():
        raise ValueError("pair weights must be finite values in [0,1]")

    rho = distances / taus
    raw_pair_loss = F.relu(1.0 - rho)
    weighted_pair_loss = raw_pair_loss * weights
    sample_loss, active_index = weighted_pair_loss.max(dim=1)
    rows = torch.arange(features.shape[0], device=features.device)
    active_d2 = distances[rows, active_index]
    active_tau = taus[rows, active_index]
    active_rho = rho[rows, active_index]
    active_weight = weights[rows, active_index]
    zero = sample_loss.sum() * 0.0
    return RelationLossResult(
        loss=sample_loss.mean(),
        absolute_component=sample_loss.mean(),
        relative_component=zero,
        diagnostics={
            "absolute_sample_loss": sample_loss.detach(),
            "relative_sample_loss": torch.zeros_like(sample_loss).detach(),
            "nearest_old_index": active_index.detach(),
            "nearest_d2": active_d2.detach(),
            "nearest_tau": active_tau.detach(),
            "nearest_rho": active_rho.detach(),
            "intrusion": (sample_loss > 0).detach(),
            "all_d2": distances.detach(),
            "all_rho": rho.detach(),
            "all_raw_sep_loss": raw_pair_loss.detach(),
            "all_applied_weight": weights.detach(),
            "all_weighted_sep_loss": weighted_pair_loss.detach(),
            "gate_weight": active_weight.detach(),
            "used_relative": torch.zeros_like(sample_loss, dtype=torch.bool).detach(),
        },
    )
