"""Similarity-selective losses layered over Structure AM-Drift.

FROM CLAD CONCEPT
-----------------
Top-P conflict-class selection, class-aware similarity and disentanglement.

ADAPTED FOR EXEMPLAR-FREE STRUCTURE
-----------------------------------
Exemplars are replaced by APR-corrected prototypes. Selective AM-Drift,
similarity-enlarged tau and Proto-CLAD are adaptations, not CLAD equations.

The experiment specification writes raw ``[tau-D2]+`` hinges. The inherited
E3 AM-Drift baseline is actually the dimensionless ``[1-D2/tau]+``. Methods A
and B deliberately retain that repository scale so lambda=0.3 is controlled;
the raw hinge is still returned for diagnostics.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from similarity.losses import RelationLossResult, pairwise_diagonal_distance


def top_p_selection(similarities: torch.Tensor, fraction: float):
    """Per-new-class Top-P mask with deterministic old-class-ID tie breaks."""
    values = torch.as_tensor(similarities).float()
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("similarities must be [new_classes, old_classes]")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("Top-P fraction must be in (0, 1]")
    count = max(1, int(math.ceil(values.shape[1] * float(fraction))))
    # stable=True preserves ascending old-class ID when similarities tie.
    order = torch.argsort(values, dim=1, descending=True, stable=True)
    selected = torch.zeros_like(values, dtype=torch.bool)
    selected.scatter_(1, order[:, :count], True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, values.shape[1] + 1, device=values.device)
    ranks.scatter_(1, order, rank_values[None, :].expand_as(order))
    cutoff = values.gather(1, order[:, count - 1 : count]).squeeze(1)
    return selected, ranks, cutoff, count


def _common(features, targets, cache, similarities, known_classes, top_p, eps=1e-8):
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets must be [B,D] and [B]")
    if getattr(cache, "covariance_view", None) != "diagonal":
        raise ValueError("selective losses require a diagonal SeparationCache")
    similarities = torch.as_tensor(similarities, device=features.device).float()
    offsets = targets.long() - int(known_classes)
    if similarities.ndim != 2 or similarities.shape[1] != cache.region_count:
        raise ValueError("similarity table must be [new classes, old regions]")
    if (offsets < 0).any() or (offsets >= similarities.shape[0]).any():
        raise ValueError("targets do not index the frozen current-task similarity table")
    selected_all, ranks_all, cutoffs_all, count = top_p_selection(similarities, top_p)
    selected = selected_all[offsets]
    ranks = ranks_all[offsets]
    cutoffs = cutoffs_all[offsets]
    sample_similarity = similarities[offsets]
    d2 = pairwise_diagonal_distance(features, cache, eps=eps)
    drift_tau = cache.taus.to(features.device).float().clamp_min(eps)[None, :].expand_as(d2)
    return {
        "offsets": offsets,
        "similarity": sample_similarity,
        "selected": selected,
        "ranks": ranks,
        "cutoff": cutoffs,
        "selected_count": count,
        "d2": d2,
        "drift_tau": drift_tau,
    }


def _result(sample_loss, pair_loss, common, pair_tau, extra=None):
    selected = common["selected"]
    masked_rho = torch.where(selected, common["d2"] / pair_tau, torch.inf)
    nearest_rho, nearest = masked_rho.min(dim=1)
    rows = torch.arange(sample_loss.shape[0], device=sample_loss.device)
    nearest_d2 = common["d2"][rows, nearest]
    nearest_tau = pair_tau[rows, nearest]
    zero = sample_loss.sum() * 0.0
    diagnostics = {
        "absolute_sample_loss": sample_loss.detach(),
        "relative_sample_loss": torch.zeros_like(sample_loss).detach(),
        "nearest_old_index": nearest.detach(),
        "nearest_d2": nearest_d2.detach(),
        "nearest_tau": nearest_tau.detach(),
        "nearest_rho": nearest_rho.detach(),
        "intrusion": nearest_rho.lt(1.0).detach(),
        "all_d2": common["d2"].detach(),
        "all_rho": (common["d2"] / pair_tau).detach(),
        "all_pair_taus": pair_tau.detach(),
        "gate_weight": torch.zeros_like(sample_loss).detach(),
        "used_relative": torch.zeros_like(sample_loss, dtype=torch.bool).detach(),
        "selected_mask": selected.detach(),
        "similarity": common["similarity"].detach(),
        "similarity_rank": common["ranks"].detach(),
        "cutoff_beta": common["cutoff"].detach(),
        "pair_loss": pair_loss.detach(),
        "raw_hinge": F.relu(pair_tau - common["d2"]).detach(),
    }
    diagnostics.update(extra or {})
    return RelationLossResult(
        loss=sample_loss.mean(),
        absolute_component=sample_loss.mean(),
        relative_component=zero,
        diagnostics=diagnostics,
    )


def selective_am_relation(features, targets, cache, similarities, known_classes, top_p):
    common = _common(features, targets, cache, similarities, known_classes, top_p)
    pair_loss = F.relu(1.0 - common["d2"] / common["drift_tau"])
    selected_values = torch.where(common["selected"], pair_loss, 0.0)
    sample_loss = selected_values.sum(dim=1) / common["selected_count"]
    return _result(sample_loss, pair_loss, common, common["drift_tau"])


def enlarged_boundary_relation(
    features, targets, cache, similarities, known_classes, top_p, alpha, eps=1e-8
):
    common = _common(features, targets, cache, similarities, known_classes, top_p, eps=eps)
    cutoff = common["cutoff"][:, None]
    excess = ((common["similarity"] - cutoff) / (1.0 - cutoff + eps)).clamp(0.0, 1.0)
    scale = 1.0 + float(alpha) * excess
    enlarged_tau = common["drift_tau"] * scale
    pair_loss = F.relu(1.0 - common["d2"] / enlarged_tau)
    sample_loss = torch.where(common["selected"], pair_loss, 0.0).sum(dim=1)
    sample_loss = sample_loss / common["selected_count"]
    return _result(
        sample_loss,
        pair_loss,
        common,
        enlarged_tau,
        {"similarity_excess": excess.detach(), "tau_scale": scale.detach()},
    )


def proto_clad_relation(
    features,
    targets,
    cache,
    apr_old_prototypes,
    similarities,
    known_classes,
    top_p,
    eps=1e-8,
):
    common = _common(features, targets, cache, similarities, known_classes, top_p, eps=eps)
    z = F.normalize(features.float(), dim=1, eps=eps)
    old = F.normalize(
        torch.as_tensor(apr_old_prototypes, device=features.device).float(), dim=1, eps=eps
    )
    cosine = z @ old.T
    pair_loss = 1.0 + cosine
    sample_loss = torch.where(common["selected"], pair_loss, 0.0).sum(dim=1)
    sample_loss = sample_loss / common["selected_count"]
    result = _result(sample_loss, pair_loss, common, common["drift_tau"])
    result.diagnostics.update({"pair_cosine": cosine.detach()})
    return result
