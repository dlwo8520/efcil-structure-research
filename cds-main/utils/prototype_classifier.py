"""Cosine prototype inference and evaluation-only representation-drift metrics."""

import torch
from torch.nn import functional as F


def ordered_prototype_matrix(prototypes, total_classes, device, dtype=torch.float32):
    expected = list(range(int(total_classes)))
    missing = [class_id for class_id in expected if class_id not in prototypes]
    if missing:
        raise RuntimeError("Missing stored prototypes for seen classes: {}".format(missing))
    matrix = torch.stack([prototypes[class_id] for class_id in expected], dim=0)
    return matrix.to(device=device, dtype=dtype)


def prototype_logits(features, prototypes):
    """Cosine scores over every seen class; no oracle task ID is used."""
    if features.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("features and prototypes must both be two-dimensional")
    if features.shape[1] != prototypes.shape[1]:
        raise ValueError("Feature/prototype dimensions do not match")
    return F.normalize(features, p=2, dim=1) @ F.normalize(prototypes, p=2, dim=1).T


def class_means_from_sums(sums, counts):
    return {class_id: sums[class_id] / float(counts[class_id]) for class_id in sums}


def prototype_drift_rows(stored_prototypes, current_prototypes, class_ids, task_index):
    rows = []
    valid_ids = [
        int(class_id) for class_id in class_ids
        if class_id in stored_prototypes and class_id in current_prototypes
    ]
    for class_id in valid_ids:
        stored = stored_prototypes[class_id].float().view(1, -1)
        current = current_prototypes[class_id].float().view(1, -1)
        cosine = F.cosine_similarity(stored, current).item()
        rows.append({
            "task": int(task_index),
            "class_id": class_id,
            "prototype_cosine_drift": float(1.0 - cosine),
        })
    return rows


def pairwise_geometry_drift(stored_prototypes, current_prototypes, class_ids):
    valid_ids = [
        int(class_id) for class_id in class_ids
        if class_id in stored_prototypes and class_id in current_prototypes
    ]
    if len(valid_ids) < 2:
        return 0.0
    stored = torch.stack([stored_prototypes[class_id].float() for class_id in valid_ids])
    current = torch.stack([current_prototypes[class_id].float() for class_id in valid_ids])
    stored_similarity = F.normalize(stored, p=2, dim=1) @ F.normalize(stored, p=2, dim=1).T
    current_similarity = F.normalize(current, p=2, dim=1) @ F.normalize(current, p=2, dim=1).T
    off_diagonal = ~torch.eye(len(valid_ids), dtype=torch.bool)
    return float(torch.abs(stored_similarity - current_similarity)[off_diagonal].mean().item())
