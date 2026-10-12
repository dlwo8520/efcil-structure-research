"""Pure, testable computations for the StoredAPR prototype definition."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rp.projection import project_relu


def birth_prototypes(features, labels, class_ids, matrix, *, projected=None):
    """Return class-level mean(z) and mean(ReLU(zR)); never retain samples."""
    features = torch.as_tensor(features).detach().float()
    labels = torch.as_tensor(labels).detach().long()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features and labels must have shapes [N,D] and [N]")
    projected = (
        project_relu(features, matrix).detach()
        if projected is None else torch.as_tensor(projected).detach().float()
    )
    if projected.shape != (len(features), int(matrix.shape[1])):
        raise ValueError("projected sample features have incorrect shape")
    result = {}
    for class_id in class_ids:
        class_id = int(class_id)
        mask = labels == class_id
        if not bool(mask.any()):
            raise ValueError(f"missing birth samples for class {class_id}")
        result[class_id] = {
            "mu_birth": features[mask].mean(dim=0).cpu().clone(),
            "q_birth": projected[mask].mean(dim=0).cpu().clone(),
            "sample_count": int(mask.sum()),
        }
    return result


def apr_corrected_prototypes(records, apr_means, matrix, *, device=None):
    """q_birth + ReLU(mu_APR R) - ReLU(mu_birth R), in class-ID order."""
    apr_means = torch.as_tensor(apr_means, dtype=torch.float32, device=device)
    if apr_means.ndim != 2 or len(records) != len(apr_means):
        raise ValueError("APR rows and stored birth class count differ")
    expected = list(range(len(apr_means)))
    if sorted(records) != expected:
        raise ValueError(f"stored birth class IDs must equal {expected}")
    birth_mu = torch.stack([records[c]["mu_birth"] for c in expected]).to(apr_means.device)
    birth_q = torch.stack([records[c]["q_birth"] for c in expected]).to(apr_means.device)
    matrix = matrix.to(apr_means.device)
    oldstyle = project_relu(apr_means, matrix)
    projected_birth_mean = project_relu(birth_mu, matrix)
    corrected = birth_q + oldstyle - projected_birth_mean
    return corrected, oldstyle, birth_q, projected_birth_mean


def comparison(left, right):
    """Cosine and raw/relative L2 of two class-level projected vectors."""
    left = torch.as_tensor(left).float()
    right = torch.as_tensor(right).float()
    difference = left - right
    return {
        "cosine": float(F.cosine_similarity(left[None], right[None], dim=1, eps=1e-8)[0]),
        "l2": float(difference.norm()),
        "relative_l2": float(difference.norm() / right.norm().clamp_min(1e-8)),
    }
