from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass
class KNNResult:
    predictions: np.ndarray
    neighbor_labels: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_indices: np.ndarray


def _vote(labels: np.ndarray, similarities: np.ndarray) -> np.ndarray:
    predictions = np.empty(labels.shape[0], dtype=np.int64)
    for row in range(labels.shape[0]):
        values, counts = np.unique(labels[row], return_counts=True)
        candidates = values[counts == counts.max()]
        if candidates.size > 1:
            similarity_sums = np.asarray(
                [similarities[row, labels[row] == candidate].sum() for candidate in candidates]
            )
            candidates = candidates[similarity_sums == similarity_sums.max()]
        predictions[row] = int(candidates.min())
    return predictions


def cosine_knn(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    k: int,
    device: torch.device,
    *,
    reference_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    chunk_size: int = 512,
) -> KNNResult:
    if k <= 0 or k >= reference_features.shape[0]:
        raise ValueError(f"k must be in [1, reference_size-1], got {k}")
    reference = torch.as_tensor(reference_features, dtype=torch.float32, device=device)
    query = torch.as_tensor(query_features, dtype=torch.float32, device=device)
    reference = torch.nn.functional.normalize(reference, dim=1)
    query = torch.nn.functional.normalize(query, dim=1)
    id_to_reference: dict[str, list[int]] = defaultdict(list)
    if reference_ids is not None or query_ids is not None:
        if reference_ids is None or query_ids is None:
            raise ValueError("reference_ids and query_ids must be supplied together")
        if len(reference_ids) != reference.shape[0] or len(query_ids) != query.shape[0]:
            raise ValueError("sample ID length mismatch")
        for index, sample_id in enumerate(reference_ids):
            id_to_reference[str(sample_id)].append(index)
    all_indices, all_similarities = [], []
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        similarities = query[start:stop] @ reference.T
        if query_ids is not None:
            for local, sample_id in enumerate(query_ids[start:stop]):
                excluded = id_to_reference.get(str(sample_id), [])
                if excluded:
                    similarities[local, torch.as_tensor(excluded, device=device)] = -torch.inf
        if torch.isneginf(similarities).all(dim=1).any():
            raise RuntimeError("self-exclusion removed every reference candidate")
        values, indices = torch.topk(similarities, k=k, dim=1, largest=True, sorted=True)
        if torch.isneginf(values).any():
            raise RuntimeError(
                "fewer than k reference candidates remain after original-sample exclusion"
            )
        all_indices.append(indices.cpu().numpy())
        all_similarities.append(values.cpu().numpy())
    neighbor_indices = np.concatenate(all_indices)
    neighbor_similarities = np.concatenate(all_similarities)
    neighbor_labels = np.asarray(reference_labels, dtype=np.int64)[neighbor_indices]
    predictions = _vote(neighbor_labels, neighbor_similarities)
    return KNNResult(
        predictions=predictions,
        neighbor_labels=neighbor_labels,
        neighbor_distances=1.0 - neighbor_similarities,
        neighbor_indices=neighbor_indices,
    )


def purity(neighbor_labels: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    return np.mean(neighbor_labels == np.asarray(true_labels)[:, None], axis=1)


def macro_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for class_id in np.unique(labels):
        mask = labels == class_id
        values.append(float(np.mean(predictions[mask] == labels[mask]) * 100.0))
    return float(np.mean(values)) if values else float("nan")
