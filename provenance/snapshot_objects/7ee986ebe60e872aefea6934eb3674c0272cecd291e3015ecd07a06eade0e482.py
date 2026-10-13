"""Old/new separation losses used by the follow-up experiments.

The comments marked ORIGINAL and ADAPTED are scientifically important.  None
of the adapted objectives should be reported as a verbatim reproduction of
the reference method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch.nn import functional as F


@dataclass
class LossResult:
    loss: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def _check_feature_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] tensor")


def diagonal_region_loss(
    features: torch.Tensor,
    cache,
    *,
    normalize: bool = False,
    return_all: bool = False,
    eps: float = 1e-8,
) -> LossResult:
    """Compute the E3 max-relative-intrusion loss and richer diagnostics.

    With ``normalize=False`` this uses exactly the same loss equation as the
    base repository's diagonal ``SeparationCache.loss``:

        mean_i relu(1 - min_k D²(i,k) / tau_k)

    ``normalize=True`` is an *ADAPTED FOR STRUCTURE* E3-Norm ablation.  It is
    not part of Structure. Means and features are L2-normalized and diagonal
    covariance is transformed with the first-order scale 1 / ||mean||².  This
    intentionally remains an approximation because no old raw features are
    retained to estimate the exact covariance on the unit sphere.
    """
    _check_feature_matrix("features", features)
    if getattr(cache, "covariance_view", None) != "diagonal":
        raise ValueError("follow-up diagonal loss requires a diagonal cache")
    if features.shape[1] != cache.means.shape[1]:
        raise ValueError("student feature dimension does not match cache")

    query = F.normalize(features, dim=1, eps=eps) if normalize else features
    best_rho = torch.full((features.shape[0],), float("inf"), device=features.device)
    best_d2 = torch.full_like(best_rho, float("inf"))
    best_tau = torch.zeros_like(best_rho)
    best_class = torch.full(
        (features.shape[0],), -1, dtype=torch.long, device=features.device
    )
    all_distances = []
    all_relative_distances = []

    for start in range(0, cache.region_count, cache.chunk_size):
        stop = min(start + cache.chunk_size, cache.region_count)
        means_raw = cache.means[start:stop].to(features.device, non_blocking=True)
        variances = cache.factors[start:stop].to(features.device, non_blocking=True)
        if normalize:
            mean_norm_sq = means_raw.square().sum(dim=1, keepdim=True).clamp_min(eps)
            means = F.normalize(means_raw, dim=1, eps=eps)
            variances = (variances / mean_norm_sq).clamp_min(eps)
        else:
            means = means_raw
        taus = cache.taus[start:stop].to(features.device, non_blocking=True)
        class_ids = cache.class_ids[start:stop].to(features.device, non_blocking=True)
        difference = query[:, None, :] - means[None, :, :]
        distances = (difference.square() / variances.unsqueeze(0)).sum(dim=2)
        rho = distances / taus.unsqueeze(0)
        if return_all:
            all_distances.append(distances.detach())
            all_relative_distances.append(rho.detach())
        chunk_rho, chunk_index = rho.min(dim=1)
        update = chunk_rho < best_rho
        selected_d2 = distances.gather(1, chunk_index[:, None]).squeeze(1)
        selected_tau = taus[chunk_index]
        selected_class = class_ids[chunk_index]
        best_rho = torch.where(update, chunk_rho, best_rho)
        best_d2 = torch.where(update, selected_d2, best_d2)
        best_tau = torch.where(update, selected_tau, best_tau)
        best_class = torch.where(update, selected_class, best_class)

    relative_penalty = F.relu(1.0 - best_rho)
    absolute_violation = F.relu(best_tau - best_d2)
    diagnostics = {
            "nearest_d2": best_d2.detach(),
            "nearest_rho": best_rho.detach(),
            "nearest_tau": best_tau.detach(),
            "nearest_old_class": best_class.detach(),
            "intrusion": (best_rho < 1.0).detach(),
            "margin_violation": absolute_violation.detach(),
    }
    if return_all:
        diagnostics.update(
            {
                "all_d2": torch.cat(all_distances, dim=1),
                "all_rho": torch.cat(all_relative_distances, dim=1),
                "region_class_ids": cache.class_ids.detach().to(features.device),
                "region_taus": cache.taus.detach().to(features.device),
            }
        )
    return LossResult(relative_penalty.mean(), diagnostics)


def prototype_geometry(
    features: torch.Tensor,
    targets: torch.Tensor,
    old_prototypes: torch.Tensor,
    current_prototypes: torch.Tensor,
    known_classes: int,
) -> Dict[str, torch.Tensor]:
    """Common normalized old/new geometry diagnostics."""
    _check_feature_matrix("features", features)
    _check_feature_matrix("old_prototypes", old_prototypes)
    _check_feature_matrix("current_prototypes", current_prototypes)
    z = F.normalize(features, dim=1)
    old = F.normalize(old_prototypes.to(features.device), dim=1)
    current = F.normalize(current_prototypes.to(features.device), dim=1)
    target_offsets = targets.long() - int(known_classes)
    if (target_offsets < 0).any() or (target_offsets >= current.shape[0]).any():
        raise ValueError("targets do not index the current-task prototype table")
    old_cosine = z @ old.T
    nearest_old_cosine, nearest_index = old_cosine.max(dim=1)
    correct_new_cosine = (z * current[target_offsets]).sum(dim=1)
    nearest_old_euclidean = torch.cdist(features.float(), old_prototypes.to(features).float()).min(dim=1).values
    return {
        "nearest_old_cosine": nearest_old_cosine.detach(),
        "correct_new_cosine": correct_new_cosine.detach(),
        "cosine_gap": (correct_new_cosine - nearest_old_cosine).detach(),
        "nearest_old_class": nearest_index.detach(),
        "nearest_old_euclidean": nearest_old_euclidean.detach(),
    }


def pic_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    old_prototypes: torch.Tensor,
    current_prototypes: torch.Tensor,
    known_classes: int,
    temperature: float = 0.07,
) -> LossResult:
    """FCS-PIC-inspired contrastive old/new separation.

    ORIGINAL (FCS, CVPR 2024): normalized query/key features from two
    augmentations form supervised positives; calibrated old prototypes are
    appended to the denominator as negatives.  FCS ``SupContrastive``
    computes log(1 + sum_neg exp(l_neg) * exp(-l_pos)) over positive pairs.

    ADAPTED FOR STRUCTURE: Structure's loader yields one view.  The correct
    current-class epoch prototype is therefore added as a guaranteed positive;
    other same-label batch features are additional positives, and APR-corrected
    old prototypes are negatives.  This tests the PIC separation mechanism,
    not the complete FCS feature-calibration method.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    geometry = prototype_geometry(
        features, targets, old_prototypes, current_prototypes, known_classes
    )
    z = F.normalize(features, dim=1)
    old = F.normalize(old_prototypes.to(features.device), dim=1)
    current = F.normalize(current_prototypes.to(features.device), dim=1)
    offsets = targets.long() - int(known_classes)
    positive_prototypes = current[offsets]

    # Candidate layout: detached same-batch keys, one correct prototype per
    # query, then all old prototypes.  Self keys are explicitly excluded.
    keys = torch.cat((z.detach(), positive_prototypes, old), dim=0)
    logits = z @ keys.T / float(temperature)
    batch_size = features.shape[0]
    candidate_labels = torch.cat(
        (
            targets.long(),
            targets.long(),
            torch.full((old.shape[0],), -1, device=targets.device, dtype=torch.long),
        )
    )
    positive_mask = targets[:, None].eq(candidate_labels[None, :])
    positive_mask[:, :batch_size].fill_diagonal_(False)
    negative_mask = ~positive_mask
    negative_mask[:, :batch_size].fill_diagonal_(False)

    sample_losses = []
    for row in range(batch_size):
        positive_logits = logits[row][positive_mask[row]]
        negative_logits = logits[row][negative_mask[row]]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            raise RuntimeError("PIC needs at least one positive and one negative")
        log_negative_sum = torch.logsumexp(negative_logits, dim=0)
        # log(1 + sum_neg exp(l_neg-l_pos)), evaluated stably.
        pair_losses = torch.logaddexp(
            torch.zeros_like(positive_logits), log_negative_sum - positive_logits
        )
        sample_losses.append(pair_losses.mean())
    loss = torch.stack(sample_losses).mean()
    diagnostics = {
        **geometry,
        "positive_similarity": (z * positive_prototypes).sum(dim=1).detach(),
        "margin_violation": (geometry["cosine_gap"] <= 0.0).detach(),
    }
    return LossResult(loss, diagnostics)


def angular_hard_negative_loss(
    features: torch.Tensor,
    targets: torch.Tensor,
    old_prototypes: torch.Tensor,
    current_prototypes: torch.Tensor,
    known_classes: int,
    *,
    margin: float = 0.5,
    top_k: int = 2,
) -> LossResult:
    """LUCIR-inspired angular margin adapted to current new samples.

    ORIGINAL (LUCIR, CVPR 2019): for reserved *old* samples, the ground-truth
    old-class cosine must exceed the top-K *new*-class cosines by a margin.

    ADAPTED FOR STRUCTURE: for a current *new* sample, its current-class epoch
    prototype cosine must exceed the top-K APR-corrected *old*-prototype
    cosines.  This reversed direction is an E3 extension, not LUCIR's loss.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    geometry = prototype_geometry(
        features, targets, old_prototypes, current_prototypes, known_classes
    )
    z = F.normalize(features, dim=1)
    old = F.normalize(old_prototypes.to(features.device), dim=1)
    current = F.normalize(current_prototypes.to(features.device), dim=1)
    offsets = targets.long() - int(known_classes)
    correct = (z * current[offsets]).sum(dim=1)
    k = min(int(top_k), old.shape[0])
    hard_values, hard_indices = (z @ old.T).topk(k, dim=1, largest=True, sorted=True)
    violations = F.relu(float(margin) - correct[:, None] + hard_values)
    diagnostics = {
        **geometry,
        "margin_violation": (violations > 0).any(dim=1).detach(),
        "margin_violation_amount": violations.mean(dim=1).detach(),
        "hard_negative_classes": hard_indices.detach(),
        "hard_negative_cosines": hard_values.detach(),
    }
    return LossResult(violations.mean(), diagnostics)


def scalar_tensor_summary(values: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Return the common mean/std/quantile summary used in epoch CSVs."""
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {}
    result = {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
        f"{prefix}_max": float(values.max()),
    }
    for label, quantile in (
        ("p10", 0.10),
        ("p25", 0.25),
        ("p50", 0.50),
        ("p75", 0.75),
        ("p90", 0.90),
        ("p95", 0.95),
    ):
        result[f"{prefix}_{label}"] = float(torch.quantile(values, quantile))
    return result
