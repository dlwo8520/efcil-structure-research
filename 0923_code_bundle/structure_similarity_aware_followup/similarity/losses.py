"""Losses for the similarity-aware Structure study.

The equations here are adaptations for exemplar-free class-incremental
classification.  They are not verbatim implementations of LUCIR, REMINDER,
STAR, or CLAD.  In particular, Structure's existing E3 implementation uses a
dimensionless max-relative-intrusion penalty, so the absolute branch below
keeps that scale for a controlled comparison with AM-Drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F


@dataclass
class RelationLossResult:
    loss: torch.Tensor
    absolute_component: torch.Tensor
    relative_component: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


@dataclass
class SCCLossResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def cosine_similarity_matrix(new_old_prototypes, old_prototypes, eps=1e-8):
    """Return frozen new-by-old cosine similarity in the old-model space."""
    new = torch.as_tensor(new_old_prototypes).detach().float()
    old = torch.as_tensor(old_prototypes).detach().float()
    if new.ndim != 2 or old.ndim != 2 or new.shape[1] != old.shape[1]:
        raise ValueError("prototype matrices must be [classes, dimension] with equal dimension")
    if new.shape[0] == 0 or old.shape[0] == 0:
        raise ValueError("similarity needs at least one new and one old class")
    return F.normalize(new, dim=1, eps=eps) @ F.normalize(old, dim=1, eps=eps).T


def _validate_inputs(features, targets, cache, similarities, known_classes):
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets must be [B,D] and [B]")
    if getattr(cache, "covariance_view", None) != "diagonal":
        raise ValueError("similarity-aware loss requires a diagonal SeparationCache")
    similarities = torch.as_tensor(similarities)
    if similarities.ndim != 2 or similarities.shape[1] != cache.region_count:
        raise ValueError("similarity matrix must be [new classes, old regions]")
    offsets = targets.long() - int(known_classes)
    if (offsets < 0).any() or (offsets >= similarities.shape[0]).any():
        raise ValueError("targets do not index the frozen new-class similarity table")
    return offsets


def pairwise_diagonal_distance(features, cache, eps=1e-8):
    """D² from every sample to every APR-corrected old diagonal region."""
    if features.ndim != 2 or features.shape[1] != cache.means.shape[1]:
        raise ValueError("feature dimension does not match the separation cache")
    means = cache.means.to(features.device, non_blocking=True).float()
    variances = cache.factors.to(features.device, non_blocking=True).float().clamp_min(eps)
    difference = features[:, None, :] - means[None, :, :]
    return (difference.square() / variances.unsqueeze(0)).sum(dim=2)


def _absolute_samples(features, cache, pair_taus=None, eps=1e-8):
    distances = pairwise_diagonal_distance(features, cache, eps=eps)
    if pair_taus is None:
        taus = cache.taus.to(features.device).float().clamp_min(eps)[None, :]
        taus = taus.expand_as(distances)
    else:
        taus = pair_taus.to(features.device).float().clamp_min(eps)
        if taus.shape != distances.shape:
            raise ValueError("pair-specific taus must match [batch, old regions]")
    rho = distances / taus
    nearest_rho, nearest_index = rho.min(dim=1)
    rows = torch.arange(features.shape[0], device=features.device)
    nearest_d2 = distances[rows, nearest_index]
    nearest_tau = taus[rows, nearest_index]
    penalty = F.relu(1.0 - nearest_rho)
    return penalty, distances, rho, nearest_index, nearest_d2, nearest_tau


def legacy_absolute_relation(features, cache):
    """AM-Drift/E3 branch, equal to the existing max-relative-intrusion loss."""
    penalty, distances, rho, nearest, nearest_d2, nearest_tau = _absolute_samples(
        features, cache
    )
    zero = penalty.sum() * 0.0
    return RelationLossResult(
        loss=penalty.mean(),
        absolute_component=penalty.mean(),
        relative_component=zero,
        diagnostics={
            "absolute_sample_loss": penalty.detach(),
            "relative_sample_loss": torch.zeros_like(penalty).detach(),
            "nearest_old_index": nearest.detach(),
            "nearest_d2": nearest_d2.detach(),
            "nearest_tau": nearest_tau.detach(),
            "nearest_rho": (nearest_d2 / nearest_tau.clamp_min(1e-8)).detach(),
            "intrusion": (nearest_d2 < nearest_tau).detach(),
            "all_d2": distances.detach(),
            "all_rho": rho.detach(),
            "gate_weight": torch.zeros_like(penalty).detach(),
            "used_relative": torch.zeros_like(penalty, dtype=torch.bool).detach(),
        },
    )


def similarity_tau_relation(
    features,
    targets,
    cache,
    similarities,
    known_classes,
    alpha=0.5,
    eps=1e-8,
):
    """S1: relax each AM-Drift tau according to frozen old-model similarity."""
    offsets = _validate_inputs(features, targets, cache, similarities, known_classes)
    similarity = torch.as_tensor(similarities, device=features.device).float()[offsets]
    normalized = ((similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    scale = (1.0 - float(alpha) * normalized).clamp_min(eps)
    base_taus = cache.taus.to(features.device).float()[None, :]
    pair_taus = base_taus * scale
    penalty, distances, rho, nearest, nearest_d2, nearest_tau = _absolute_samples(
        features, cache, pair_taus, eps=eps
    )
    rows = torch.arange(features.shape[0], device=features.device)
    zero = penalty.sum() * 0.0
    return RelationLossResult(
        loss=penalty.mean(),
        absolute_component=penalty.mean(),
        relative_component=zero,
        diagnostics={
            "absolute_sample_loss": penalty.detach(),
            "relative_sample_loss": torch.zeros_like(penalty).detach(),
            "nearest_old_index": nearest.detach(),
            "nearest_similarity": similarity[rows, nearest].detach(),
            "similarity_scale": scale[rows, nearest].detach(),
            "nearest_d2": nearest_d2.detach(),
            "nearest_tau": nearest_tau.detach(),
            "nearest_rho": (nearest_d2 / nearest_tau.clamp_min(eps)).detach(),
            "intrusion": (nearest_d2 < nearest_tau).detach(),
            "all_d2": distances.detach(),
            "all_rho": rho.detach(),
            "all_pair_taus": pair_taus.detach(),
            "gate_weight": torch.zeros_like(penalty).detach(),
            "used_relative": torch.zeros_like(penalty, dtype=torch.bool).detach(),
        },
    )


def _relative_samples(
    features,
    targets,
    old_prototypes,
    current_prototypes,
    similarities,
    known_classes,
    margin,
    eps=1e-8,
):
    similarities = torch.as_tensor(similarities, device=features.device).float()
    offsets = targets.long() - int(known_classes)
    if (offsets < 0).any() or (offsets >= similarities.shape[0]).any():
        raise ValueError("targets do not index current prototypes")
    hard_similarity, hard_old = similarities.max(dim=1)
    hard_for_sample = hard_old[offsets]
    similarity_for_sample = hard_similarity[offsets]
    z = F.normalize(features, dim=1, eps=eps)
    old = F.normalize(torch.as_tensor(old_prototypes, device=features.device).float(), dim=1, eps=eps)
    current = F.normalize(
        torch.as_tensor(current_prototypes, device=features.device).float(), dim=1, eps=eps
    )
    s_new = (z * current[offsets]).sum(dim=1)
    s_old = (z * old[hard_for_sample]).sum(dim=1)
    gap = s_new - s_old
    violation = F.relu(float(margin) - gap)
    return violation, s_new, s_old, gap, hard_for_sample, similarity_for_sample


def hard_relative_relation(
    features,
    targets,
    cache,
    old_prototypes,
    current_prototypes,
    similarities,
    known_classes,
    beta,
    margin=0.05,
):
    """S2/S3: hard switch between AM-Drift absolute and adapted LUCIR margin."""
    _validate_inputs(features, targets, cache, similarities, known_classes)
    absolute, distances, rho, nearest, nearest_d2, nearest_tau = _absolute_samples(features, cache)
    relative, s_new, s_old, gap, hard_old, hard_similarity = _relative_samples(
        features,
        targets,
        old_prototypes,
        current_prototypes,
        similarities,
        known_classes,
        margin,
    )
    use_relative = hard_similarity >= float(beta)
    absolute_weighted = torch.where(use_relative, torch.zeros_like(absolute), absolute)
    relative_weighted = torch.where(use_relative, relative, torch.zeros_like(relative))
    sample_loss = absolute_weighted + relative_weighted
    return RelationLossResult(
        loss=sample_loss.mean(),
        absolute_component=absolute_weighted.mean(),
        relative_component=relative_weighted.mean(),
        diagnostics={
            "absolute_sample_loss": absolute.detach(),
            "relative_sample_loss": relative.detach(),
            "weighted_absolute": absolute_weighted.detach(),
            "weighted_relative": relative_weighted.detach(),
            "nearest_old_index": nearest.detach(),
            "hard_old_index": hard_old.detach(),
            "hard_class_similarity": hard_similarity.detach(),
            "s_new": s_new.detach(),
            "s_old": s_old.detach(),
            "cosine_gap": gap.detach(),
            "margin_violation": (relative > 0).detach(),
            "nearest_d2": nearest_d2.detach(),
            "nearest_tau": nearest_tau.detach(),
            "nearest_rho": (nearest_d2 / nearest_tau.clamp_min(1e-8)).detach(),
            "intrusion": (nearest_d2 < nearest_tau).detach(),
            "all_d2": distances.detach(),
            "all_rho": rho.detach(),
            "gate_weight": use_relative.float().detach(),
            "used_relative": use_relative.detach(),
        },
    )


def soft_hybrid_relation(
    features,
    targets,
    cache,
    old_prototypes,
    current_prototypes,
    similarities,
    known_classes,
    beta,
    temperature=0.1,
    margin=0.05,
):
    """S4: smoothly blend AM-Drift absolute and adapted relative margin."""
    if temperature <= 0:
        raise ValueError("soft gate temperature must be positive")
    _validate_inputs(features, targets, cache, similarities, known_classes)
    absolute, distances, rho, nearest, nearest_d2, nearest_tau = _absolute_samples(features, cache)
    relative, s_new, s_old, gap, hard_old, hard_similarity = _relative_samples(
        features,
        targets,
        old_prototypes,
        current_prototypes,
        similarities,
        known_classes,
        margin,
    )
    gate = torch.sigmoid((hard_similarity - float(beta)) / float(temperature))
    absolute_weighted = (1.0 - gate) * absolute
    relative_weighted = gate * relative
    sample_loss = absolute_weighted + relative_weighted
    return RelationLossResult(
        loss=sample_loss.mean(),
        absolute_component=absolute_weighted.mean(),
        relative_component=relative_weighted.mean(),
        diagnostics={
            "absolute_sample_loss": absolute.detach(),
            "relative_sample_loss": relative.detach(),
            "weighted_absolute": absolute_weighted.detach(),
            "weighted_relative": relative_weighted.detach(),
            "nearest_old_index": nearest.detach(),
            "hard_old_index": hard_old.detach(),
            "hard_class_similarity": hard_similarity.detach(),
            "s_new": s_new.detach(),
            "s_old": s_old.detach(),
            "cosine_gap": gap.detach(),
            "margin_violation": (relative > 0).detach(),
            "nearest_d2": nearest_d2.detach(),
            "nearest_tau": nearest_tau.detach(),
            "nearest_rho": (nearest_d2 / nearest_tau.clamp_min(1e-8)).detach(),
            "intrusion": (nearest_d2 < nearest_tau).detach(),
            "all_d2": distances.detach(),
            "all_rho": rho.detach(),
            "gate_weight": gate.detach(),
            "used_relative": (gate >= 0.5).detach(),
        },
    )


def similarity_weighted_scc(
    outputs,
    old_class_means,
    targets,
    similarities,
    known_classes,
    alpha=0.5,
    minimum_weight=0.5,
    eps=1e-8,
):
    """S3 SCC adaptation with a frozen class-similarity sample weight.

    ORIGINAL Structure SCC compares teacher/student relations to all stored old
    prototypes and averages the per-sample squared relation error.
    ADAPTED here: only that per-sample scalar is weighted; prototypes and SCC
    relation construction remain unchanged.
    """
    teacher = outputs["tfeatures"]
    student = outputs["sfeatures"]
    old = torch.as_tensor(old_class_means, device=student.device).float()
    if old.ndim != 2 or old.shape[0] == 0:
        zero = student.sum() * 0.0
        return SCCLossResult(zero, zero, {})
    offsets = targets.long() - int(known_classes)
    similarities = torch.as_tensor(similarities, device=student.device).float()
    if (offsets < 0).any() or (offsets >= similarities.shape[0]).any():
        raise ValueError("SCC target does not index the similarity table")
    old_norm = F.normalize(old, dim=1, eps=eps)
    teacher_relation = F.normalize(teacher, dim=1, eps=eps) @ old_norm.T
    student_relation = F.normalize(student, dim=1, eps=eps) @ old_norm.T
    raw_per_sample = (student_relation - teacher_relation).square().sum(dim=1)
    max_similarity = similarities.max(dim=1).values[offsets]
    normalized_similarity = ((max_similarity + 1.0) * 0.5).clamp(0.0, 1.0)
    weights = (1.0 - float(alpha) * normalized_similarity).clamp_min(float(minimum_weight))
    weighted_per_sample = weights * raw_per_sample
    return SCCLossResult(
        raw_loss=raw_per_sample.mean(),
        weighted_loss=weighted_per_sample.mean(),
        diagnostics={
            "raw_scc_loss": raw_per_sample.detach(),
            "weighted_scc_loss": weighted_per_sample.detach(),
            "scc_weight": weights.detach(),
            "max_old_similarity": max_similarity.detach(),
        },
    )

