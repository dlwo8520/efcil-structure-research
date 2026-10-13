from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


CONDITIONS = (
    ("A_STRUCTURE", False, False),
    ("B_ORACLE_MEAN", True, False),
    ("C_ORACLE_COVARIANCE", False, True),
    ("D_FULL_ORACLE", True, True),
)


class DiagnosticCosineHead(nn.Module):
    """Cosine head that preserves Structure's task-wise parameter layout.

    ``SimpleContinualLinear`` stores one ``nn.Linear.weight`` per task and
    evaluates each task head independently before concatenating logits.  A
    single concatenated Parameter is mathematically equivalent, but it does
    not necessarily follow the same CUDA kernels/update path.  Keeping the
    task-wise Parameters is therefore required by the Condition-A regression
    check.  Parameters are constructed directly so no random initializer
    advances the saved pre-CA RNG stream.
    """

    def __init__(self, weights: torch.Tensor, task_sizes: list[int] | None = None):
        super().__init__()
        weights = weights.detach().clone().float()
        sizes = list(map(int, task_sizes or [weights.shape[0]]))
        if any(size <= 0 for size in sizes) or sum(sizes) != int(weights.shape[0]):
            raise ValueError(
                f"task sizes {sizes} do not partition {weights.shape[0]} classifier rows"
            )
        self.task_sizes = sizes
        self.weights = nn.ParameterList(
            [nn.Parameter(chunk.clone()) for chunk in torch.split(weights, sizes, dim=0)]
        )

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat(list(self.weights), dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(features, p=2, dim=1)
        outputs = [
            F.linear(normalized, F.normalize(weight, p=2, dim=1))
            for weight in self.weights
        ]
        return torch.cat(outputs, dim=1)


def _head_key_order(key: str) -> tuple[int, str]:
    parts = key.split(".")
    for index, value in enumerate(parts):
        if value == "heads" and index + 1 < len(parts):
            return int(parts[index + 1]), key
    return (10**9, key)


def concatenate_classifier_weights(state: dict[str, torch.Tensor]) -> torch.Tensor:
    keys = [
        key for key, value in state.items()
        if key.endswith("weight") and "heads" in key and value.ndim == 2
    ]
    keys.sort(key=_head_key_order)
    if not keys:
        raise KeyError("classifier head weights not found")
    return torch.cat([state[key].detach().cpu().float() for key in keys], dim=0)


def checkpoint_classifier_weights(checkpoint: dict) -> torch.Tensor:
    return concatenate_classifier_weights(checkpoint["network_state_dict"])


def find_pre_ca_sidecar(checkpoint_path: Path, task: int) -> Path | None:
    candidates = [
        checkpoint_path.parent / "pre_ca_classifier" / f"task_{task:02d}.pt",
        checkpoint_path.parent.parent / "pre_ca_classifier" / f"task_{task:02d}.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_classifier_initialization(
    checkpoint: dict,
    checkpoint_path: Path,
    class_ids: np.ndarray,
    task: int,
) -> tuple[torch.Tensor, dict[str, Any] | None, str]:
    sidecar_path = find_pre_ca_sidecar(checkpoint_path, task)
    if sidecar_path is not None:
        sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        weights = concatenate_classifier_weights(sidecar["classifier_state_dict"])
        source = "exact_pre_ca_sidecar"
    else:
        sidecar = None
        weights = checkpoint_classifier_weights(checkpoint)
        source = "shared_post_ca_legacy_init"
    return weights[torch.as_tensor(class_ids, dtype=torch.long)].clone(), sidecar, source


def array_checksum(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _distribution(
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> tuple[torch.distributions.MultivariateNormal, float]:
    # The first attempt intentionally matches vanilla Structure exactly.
    # Stabilization is only a diagnostic fallback for an invalid oracle
    # covariance; Condition A must take this zero-jitter branch.
    mean = mean.float()
    covariance = covariance.float()
    try:
        return torch.distributions.MultivariateNormal(mean, covariance), 0.0
    except (ValueError, RuntimeError):
        covariance = 0.5 * (covariance + covariance.T)
        identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        jitter = 1e-6
        for _ in range(7):
            try:
                return torch.distributions.MultivariateNormal(mean, covariance + jitter * identity), jitter
            except (ValueError, RuntimeError):
                jitter *= 10.0
        raise


@dataclass
class HeadTrainingResult:
    weights: np.ndarray
    final_loss: float
    numerical_fallback_count: int
    max_extra_jitter: float
    optimizer_parameter_count: int
    input_checksum_unchanged: bool


def train_diagnostic_head(
    initial_weights: torch.Tensor,
    means: np.ndarray,
    covariances: np.ndarray,
    class_ids: np.ndarray,
    task_index: int,
    task_size: int,
    epochs: int,
    samples_per_class: int,
    init_lr: float,
    weight_decay: float,
    scale: float,
    device: torch.device,
    seed: int,
    rng_state: dict[str, Any] | None = None,
    task_sizes: list[int] | None = None,
) -> HeadTrainingResult:
    if epochs <= 0:
        raise ValueError("classifier adaptation epochs must be positive")
    if samples_per_class <= 0:
        raise ValueError("synthetic samples per class must be positive")
    if class_ids.size == 0:
        raise ValueError("at least one class is required")
    if not (initial_weights.shape[0] == means.shape[0] == covariances.shape[0] == class_ids.size):
        raise ValueError("classifier/statistics/class-id row counts must match")
    before = array_checksum(initial_weights.numpy(), means, covariances)
    saved_rng = _capture_rng()
    try:
        if rng_state is not None:
            _restore_rng(rng_state)
        else:
            _seed_all(seed)
        head = DiagnosticCosineHead(initial_weights, task_sizes=task_sizes).to(device)
        parameters = list(head.parameters())
        optimizer = torch.optim.SGD(
            parameters, lr=init_lr, momentum=0.9, weight_decay=weight_decay
        )
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        head_ids = {id(parameter) for parameter in parameters}
        if optimizer_ids != head_ids:
            raise AssertionError("diagnostic optimizer contains non-head parameters")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        means_tensor = torch.as_tensor(means, dtype=torch.float64, device=device)
        covariances_tensor = torch.as_tensor(covariances, dtype=torch.float64, device=device)
        last_loss = float("nan")
        fallback_count = 0
        max_extra_jitter = 0.0
        for _epoch in range(epochs):
            samples = []
            labels = []
            for local_id, global_id in enumerate(class_ids.tolist()):
                task_id = int(global_id) // int(task_size)
                decay = (task_id + 1) / (task_index + 1) * 0.1
                mean_scale = 0.9 + decay
                distribution, jitter = _distribution(
                    means_tensor[local_id] * mean_scale,
                    covariances_tensor[local_id],
                )
                fallback_count += int(jitter > 0.0)
                max_extra_jitter = max(max_extra_jitter, jitter)
                samples.append(distribution.sample((samples_per_class,)))
                labels.extend([local_id] * samples_per_class)
            inputs = torch.cat(samples, dim=0).float()
            targets = torch.as_tensor(labels, dtype=torch.long, device=device)
            permutation = torch.randperm(inputs.shape[0])
            inputs = inputs[permutation]
            targets = targets[permutation]
            for iteration in range(class_ids.size):
                start = iteration * samples_per_class
                stop = (iteration + 1) * samples_per_class
                logits = scale * head(inputs[start:stop])
                loss = F.cross_entropy(logits, targets[start:stop])
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite diagnostic classifier loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                last_loss = float(loss.detach().cpu())
            scheduler.step()
            # Vanilla Structure calls ``_compute_accuracy`` with a freshly
            # iterated test DataLoader after every CA epoch.  Creating that
            # iterator consumes one int64 draw from the global CPU generator
            # (DataLoader's worker base seed).  The resulting RNG advance
            # changes the next epoch's CPU ``randperm``.  Reproduce the draw
            # without reading evaluation data so Condition A follows the
            # saved training RNG stream exactly and no test data enters CA.
            torch.empty((), dtype=torch.int64).random_()
        weights = head.weight.detach().cpu().numpy().copy()
        del head, optimizer, scheduler, means_tensor, covariances_tensor
    finally:
        _restore_rng(saved_rng)
    after = array_checksum(initial_weights.numpy(), means, covariances)
    return HeadTrainingResult(
        weights=weights,
        final_loss=last_loss,
        numerical_fallback_count=fallback_count,
        max_extra_jitter=max_extra_jitter,
        optimizer_parameter_count=len(parameters),
        input_checksum_unchanged=before == after,
    )


def cosine_scores(
    features: np.ndarray,
    weights: np.ndarray,
    scale: float,
    chunk_size: int = 2048,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    outputs = []
    for start in range(0, features.shape[0], chunk_size):
        batch = features[start : start + chunk_size]
        batch = batch / np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-12)
        outputs.append(scale * (batch @ weights.T))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, weights.shape[0]))


def evaluate_scores(
    scores: np.ndarray,
    global_labels: np.ndarray,
    class_ids: np.ndarray,
    known_before: int,
) -> dict[str, float | int]:
    mapping = {int(class_id): index for index, class_id in enumerate(class_ids.tolist())}
    local_labels = np.asarray([mapping[int(label)] for label in global_labels], dtype=np.int64)
    predictions = np.argmax(scores, axis=1)
    old_local = np.asarray([class_id < known_before for class_id in class_ids], dtype=bool)
    old_sample = np.asarray([label < known_before for label in global_labels], dtype=bool)
    new_sample = ~old_sample
    predicted_old = old_local[predictions]
    correct = predictions == local_labels
    old_correct = old_sample & correct
    new_correct = new_sample & correct
    old_to_new = old_sample & ~predicted_old
    old_wrong_old = old_sample & predicted_old & ~correct
    new_to_old = new_sample & predicted_old
    new_wrong_new = new_sample & ~predicted_old & ~correct

    def accuracy(mask, values=correct):
        return float(np.mean(values[mask]) * 100.0) if mask.any() else float("nan")

    old_columns = np.flatnonzero(old_local)
    new_columns = np.flatnonzero(~old_local)
    old_only = np.argmax(scores[:, old_columns], axis=1) if old_columns.size else np.zeros(scores.shape[0], int)
    new_only = np.argmax(scores[:, new_columns], axis=1) if new_columns.size else np.zeros(scores.shape[0], int)
    old_only_global = class_ids[old_columns[old_only]] if old_columns.size else np.full(scores.shape[0], -1)
    new_only_global = class_ids[new_columns[new_only]] if new_columns.size else np.full(scores.shape[0], -1)
    return {
        "num_samples": int(global_labels.size),
        "seen_class_accuracy": float(np.mean(correct) * 100.0),
        "cumulative_accuracy": float(np.mean(correct) * 100.0),
        "old_accuracy": accuracy(old_sample),
        "new_accuracy": accuracy(new_sample),
        "old_only_accuracy": float(np.mean(old_only_global[old_sample] == global_labels[old_sample]) * 100.0) if old_sample.any() else float("nan"),
        "new_only_accuracy": float(np.mean(new_only_global[new_sample] == global_labels[new_sample]) * 100.0) if new_sample.any() else float("nan"),
        "old_to_new_count": int(old_to_new.sum()),
        "old_to_old_wrong_count": int(old_wrong_old.sum()),
        "old_correct_count": int(old_correct.sum()),
        "new_to_old_count": int(new_to_old.sum()),
        "new_to_new_wrong_count": int(new_wrong_new.sum()),
        "new_correct_count": int(new_correct.sum()),
        "old_to_new_rate": float(np.mean(old_to_new[old_sample])) if old_sample.any() else float("nan"),
        "old_to_old_wrong_rate": float(np.mean(old_wrong_old[old_sample])) if old_sample.any() else float("nan"),
        "new_to_old_rate": float(np.mean(new_to_old[new_sample])) if new_sample.any() else float("nan"),
        "new_to_new_wrong_rate": float(np.mean(new_wrong_new[new_sample])) if new_sample.any() else float("nan"),
    }
