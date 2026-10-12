from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


EPS = 1e-12


def normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), EPS)


@dataclass
class SupportResult:
    own_support: np.ndarray
    rival_support: np.ndarray
    support_margin: np.ndarray
    nearest_own_cosine: np.ndarray
    nearest_rival_cosine: np.ndarray
    rival_class: np.ndarray
    knn_purity: np.ndarray


def cosine_support(
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    query_features: np.ndarray,
    query_labels: np.ndarray,
    class_ids: np.ndarray,
    k: int,
    device: torch.device,
    *,
    reference_ids: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    chunk_size: int = 256,
) -> SupportResult:
    """Compute class-conditional top-k cosine support and global k-NN purity.

    Features are normalized inside this function.  If IDs are supplied, every
    matching reference is excluded before top-k selection; no replication is
    used when a class has fewer than ``k`` available references.
    """
    reference_features = np.asarray(reference_features, dtype=np.float32)
    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    query_features = np.asarray(query_features, dtype=np.float32)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if reference_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("support features must be 2-D")
    if reference_features.shape[1] != query_features.shape[1]:
        raise ValueError("reference/query feature dimensions differ")
    if reference_labels.size != reference_features.shape[0]:
        raise ValueError("reference label count mismatch")
    if query_labels.size != query_features.shape[0]:
        raise ValueError("query label count mismatch")
    if k <= 0:
        raise ValueError("k must be positive")
    class_indices: dict[int, torch.Tensor] = {}
    for class_id in class_ids.tolist():
        indices = np.flatnonzero(reference_labels == class_id)
        if not indices.size:
            raise ValueError(f"reference bank is empty for class {class_id}")
        class_indices[class_id] = torch.as_tensor(indices, dtype=torch.long, device=device)
    id_to_reference: dict[str, list[int]] = defaultdict(list)
    if reference_ids is not None or query_ids is not None:
        if reference_ids is None or query_ids is None:
            raise ValueError("reference_ids and query_ids must be supplied together")
        if len(reference_ids) != reference_features.shape[0] or len(query_ids) != query_features.shape[0]:
            raise ValueError("sample ID length mismatch")
        for index, sample_id in enumerate(reference_ids):
            id_to_reference[str(sample_id)].append(index)

    reference = torch.as_tensor(reference_features, dtype=torch.float32, device=device)
    reference = torch.nn.functional.normalize(reference, dim=1)
    query = torch.as_tensor(query_features, dtype=torch.float32, device=device)
    query = torch.nn.functional.normalize(query, dim=1)
    class_to_column = {class_id: column for column, class_id in enumerate(class_ids.tolist())}
    outputs: dict[str, list[np.ndarray]] = defaultdict(list)
    global_k = min(k, reference.shape[0])

    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        similarities = query[start:stop] @ reference.T
        if query_ids is not None:
            for local, sample_id in enumerate(query_ids[start:stop]):
                excluded = id_to_reference.get(str(sample_id), [])
                if excluded:
                    similarities[local, torch.as_tensor(excluded, dtype=torch.long, device=device)] = -torch.inf
        if torch.isneginf(similarities).all(dim=1).any():
            raise RuntimeError("self exclusion removed all reference points")

        class_support = torch.empty((stop - start, class_ids.size), dtype=torch.float32, device=device)
        class_nearest = torch.empty_like(class_support)
        for column, class_id in enumerate(class_ids.tolist()):
            values = similarities.index_select(1, class_indices[class_id])
            top_values = torch.topk(values, k=min(k, values.shape[1]), dim=1).values
            finite = torch.isfinite(top_values)
            if (~finite).all(dim=1).any():
                raise RuntimeError(f"no finite class reference remains for class {class_id}")
            class_support[:, column] = torch.where(finite, top_values, 0.0).sum(1) / finite.sum(1)
            class_nearest[:, column] = top_values[:, 0]

        labels = query_labels[start:stop]
        label_columns = torch.as_tensor(
            [class_to_column[int(label)] for label in labels], dtype=torch.long, device=device
        )
        rows = torch.arange(stop - start, device=device)
        own_support = class_support[rows, label_columns]
        nearest_own = class_nearest[rows, label_columns]
        rival_values = class_support.clone()
        rival_values[rows, label_columns] = -torch.inf
        rival_support, rival_columns = torch.max(rival_values, dim=1)
        nearest_values = class_nearest.clone()
        nearest_values[rows, label_columns] = -torch.inf
        nearest_rival = torch.max(nearest_values, dim=1).values
        global_values, global_indices = torch.topk(similarities, k=global_k, dim=1)
        finite_global = torch.isfinite(global_values)
        neighbor_labels = torch.as_tensor(reference_labels, dtype=torch.long, device=device)[global_indices]
        true_labels = torch.as_tensor(labels, dtype=torch.long, device=device)[:, None]
        purity = ((neighbor_labels == true_labels) & finite_global).sum(1) / finite_global.sum(1)

        outputs["own_support"].append(own_support.cpu().numpy())
        outputs["rival_support"].append(rival_support.cpu().numpy())
        outputs["support_margin"].append((own_support - rival_support).cpu().numpy())
        outputs["nearest_own_cosine"].append(nearest_own.cpu().numpy())
        outputs["nearest_rival_cosine"].append(nearest_rival.cpu().numpy())
        outputs["rival_class"].append(class_ids[rival_columns.cpu().numpy()])
        outputs["knn_purity"].append(purity.cpu().numpy())

    return SupportResult(**{key: np.concatenate(value) for key, value in outputs.items()})


def distribution_summary(values: np.ndarray, prefix: str = "") -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {f"{prefix}{key}": float("nan") for key in ("mean", "std", "median", "q10", "q05") } | {f"{prefix}count": 0}
    return {
        f"{prefix}count": int(array.size),
        f"{prefix}mean": float(array.mean()),
        f"{prefix}std": float(array.std()),
        f"{prefix}median": float(np.median(array)),
        f"{prefix}q10": float(np.quantile(array, 0.10)),
        f"{prefix}q05": float(np.quantile(array, 0.05)),
    }


def class_macro(values: np.ndarray, labels: np.ndarray) -> float:
    per_class = [float(np.mean(values[labels == class_id])) for class_id in np.unique(labels)]
    return float(np.mean(per_class)) if per_class else float("nan")


def macro_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return class_macro((np.asarray(predictions) == np.asarray(labels)).astype(np.float64), labels) * 100.0
