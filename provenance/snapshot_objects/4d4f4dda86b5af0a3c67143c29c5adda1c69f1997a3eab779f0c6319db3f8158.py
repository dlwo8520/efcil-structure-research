"""RP-ProtoCLAD with only the old projected prototype source replaced."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rp.projection import project_relu
from selective.losses import _common, _result


def stored_apr_rp_relation(
    features, targets, cache, apr_old_prototypes, current_prototypes,
    stored_old_projected, similarities, known_classes, top_p, rp_matrix, eps=1e-8,
):
    common = _common(features, targets, cache, similarities, known_classes, top_p, eps=eps)
    rp_matrix = torch.as_tensor(rp_matrix, device=features.device).float()
    if rp_matrix.requires_grad:
        raise ValueError("RP matrix must be frozen")
    old_original = torch.as_tensor(apr_old_prototypes, device=features.device).float()
    current_original = torch.as_tensor(current_prototypes, device=features.device).float()
    rp_old_raw = torch.as_tensor(stored_old_projected, device=features.device).float()
    if rp_old_raw.shape != (int(known_classes), int(rp_matrix.shape[1])):
        raise ValueError("stored old RP prototypes have incorrect shape")
    offsets = common["offsets"]

    z_original = F.normalize(features.float(), dim=1, eps=eps)
    old_original_normalized = F.normalize(old_original, dim=1, eps=eps)
    current_original_normalized = F.normalize(current_original, dim=1, eps=eps)
    original_cosine = z_original @ old_original_normalized.T
    correct_new_cosine_original = (z_original * current_original_normalized[offsets]).sum(dim=1)

    rp_features_raw = project_relu(features, rp_matrix)
    # q_birth + APR-derived projected displacement; intentionally no final ReLU.
    rp_current_raw = project_relu(current_original, rp_matrix)
    rp_features = F.normalize(rp_features_raw, dim=1, eps=eps)
    rp_old = F.normalize(rp_old_raw, dim=1, eps=eps)
    rp_current = F.normalize(rp_current_raw, dim=1, eps=eps)
    rp_cosine = rp_features @ rp_old.T
    correct_new_cosine_rp = (rp_features * rp_current[offsets]).sum(dim=1)

    pair_loss = 1.0 + rp_cosine
    sample_loss = torch.where(common["selected"], pair_loss, 0.0).sum(dim=1)
    sample_loss = sample_loss / common["selected_count"]
    result = _result(sample_loss, pair_loss, common, common["drift_tau"])
    result.diagnostics.update({
        "pair_cosine": rp_cosine.detach(),
        "original_cosine": original_cosine.detach(),
        "rp_cosine": rp_cosine.detach(),
        "delta_cosine": (rp_cosine - original_cosine).detach(),
        "correct_new_cosine_original": correct_new_cosine_original.detach(),
        "correct_new_cosine_rp": correct_new_cosine_rp.detach(),
        "gap_original": (correct_new_cosine_original[:, None] - original_cosine).detach(),
        "gap_rp": (correct_new_cosine_rp[:, None] - rp_cosine).detach(),
        "delta_gap": (
            correct_new_cosine_rp[:, None] - rp_cosine
            - correct_new_cosine_original[:, None] + original_cosine
        ).detach(),
        "original_feature_norm": features.float().norm(dim=1).detach(),
        "rp_feature_norm": rp_features_raw.norm(dim=1).detach(),
        "rp_sample_zero_ratio": rp_features_raw.eq(0).float().mean(dim=1).detach(),
    })
    return result
