from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .model import ContinualViT


@dataclass
class SubspaceUpdateStats:
    layer_id: int
    feature_dim: int
    old_rank: int
    added_rank: int
    protected_rank: int
    protected_ratio: float


class ProtectedSubspaceBank:
    def __init__(self, num_layers: int, feature_dim: int) -> None:
        self.num_layers = int(num_layers)
        self.feature_dim = int(feature_dim)
        self.bases = [torch.empty(feature_dim, 0, dtype=torch.float32) for _ in range(num_layers)]

    def basis(self, layer_id: int) -> torch.Tensor:
        return self.bases[layer_id]

    def update(
        self,
        candidates: dict[int, torch.Tensor],
        rank_cap: int,
        tolerance: float,
    ) -> list[SubspaceUpdateStats]:
        stats: list[SubspaceUpdateStats] = []
        for layer_id in range(self.num_layers):
            old = self.bases[layer_id].float()
            old_rank = int(old.shape[1])
            selected: list[torch.Tensor] = []
            matrix = candidates.get(layer_id)
            if matrix is not None:
                for column in range(matrix.shape[1]):
                    if old_rank + len(selected) >= rank_cap:
                        break
                    vector = matrix[:, column].float().clone()
                    if old_rank:
                        vector -= old @ (old.transpose(0, 1) @ vector)
                    for previous in selected:
                        vector -= previous * torch.dot(previous, vector)
                    norm = torch.linalg.vector_norm(vector)
                    if float(norm.item()) > tolerance:
                        selected.append(vector / norm)
            if selected:
                added = torch.stack(selected, dim=1)
                combined = torch.cat((old, added), dim=1)
                # A final QR controls numerical loss of orthogonality.
                q, r = torch.linalg.qr(combined, mode="reduced")
                diagonal = torch.abs(torch.diagonal(r))
                keep = diagonal > tolerance
                combined = q[:, keep]
            else:
                combined = old
            if combined.shape[1] > rank_cap:
                combined = combined[:, :rank_cap]
            self.bases[layer_id] = combined.contiguous().cpu()
            new_rank = int(combined.shape[1])
            stats.append(SubspaceUpdateStats(
                layer_id=layer_id,
                feature_dim=self.feature_dim,
                old_rank=old_rank,
                added_rank=new_rank - old_rank,
                protected_rank=new_rank,
                protected_ratio=new_rank / self.feature_dim,
            ))
        return stats

    def export(self) -> list[torch.Tensor]:
        return [basis.cpu() for basis in self.bases]

    def load(self, bases: list[torch.Tensor]) -> None:
        if len(bases) != self.num_layers:
            raise ValueError("Protected basis layer count mismatch")
        for basis in bases:
            if basis.ndim != 2 or basis.shape[0] != self.feature_dim:
                raise ValueError(f"Invalid protected basis shape {basis.shape}")
        self.bases = [basis.detach().float().cpu().contiguous() for basis in bases]


class ActivationCollector:
    def __init__(self, tokens_per_image: int) -> None:
        self.tokens_per_image = int(tokens_per_image)
        self.current_labels: torch.Tensor | None = None
        self.storage: dict[int, dict[int, list[torch.Tensor]]] = {}

    def set_labels(self, labels: torch.Tensor) -> None:
        self.current_labels = labels.detach().cpu()

    def __call__(self, layer_id: int, activation: torch.Tensor) -> None:
        if self.current_labels is None:
            raise RuntimeError("Capture labels were not set")
        if activation.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, dim], got {activation.shape}")
        token_count = activation.shape[1]
        positions = [0]
        if self.tokens_per_image > 1 and token_count > 1:
            patch_positions = torch.linspace(
                1,
                token_count - 1,
                steps=self.tokens_per_image - 1,
                device=activation.device,
            ).round().long().unique().tolist()
            positions.extend(int(position) for position in patch_positions)
        selected = activation[:, positions, :].detach().float().cpu()
        layer_storage = self.storage.setdefault(layer_id, {})
        for batch_index, label in enumerate(self.current_labels.tolist()):
            layer_storage.setdefault(int(label), []).append(selected[batch_index])


@torch.no_grad()
def collect_class_directions(
    model: ContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    tokens_per_image: int,
    topk: int,
    energy_threshold: float | None,
    tolerance: float,
) -> tuple[dict[int, torch.Tensor], list[dict]]:
    collector = ActivationCollector(tokens_per_image=tokens_per_image)
    model.set_capture_callbacks(collector)
    model.eval()
    try:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            collector.set_labels(labels)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                model.features(images)
    finally:
        model.clear_capture_callbacks()

    candidates: dict[int, torch.Tensor] = {}
    class_stats: list[dict] = []
    for layer_id, by_class in sorted(collector.storage.items()):
        layer_directions: list[torch.Tensor] = []
        for class_id, chunks in sorted(by_class.items()):
            matrix = torch.cat(chunks, dim=0).float()
            mean = matrix.mean(dim=0)
            mean_norm = torch.linalg.vector_norm(mean)
            if float(mean_norm.item()) > tolerance:
                layer_directions.append(mean / mean_norm)
            centered = matrix - mean
            total_energy = float(centered.pow(2).sum().item())
            singular_values_list: list[float] = []
            kept = 0
            retained_energy = 0.0
            if matrix.shape[0] > 1 and total_energy > tolerance:
                _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                singular_values_list = singular_values.tolist()
                if energy_threshold is None:
                    kept = min(int(topk), int(vh.shape[0]))
                else:
                    cumulative = torch.cumsum(singular_values.pow(2), dim=0) / total_energy
                    reached = torch.nonzero(cumulative >= float(energy_threshold))
                    kept = int(reached[0].item() + 1) if reached.numel() else int(vh.shape[0])
                if kept:
                    layer_directions.extend(vh[:kept])
                    retained_energy = float(
                        singular_values[:kept].pow(2).sum().item() / total_energy
                    )
            class_stats.append({
                "layer_id": layer_id,
                "class_id": class_id,
                "sample_vectors": int(matrix.shape[0]),
                "feature_dim": int(matrix.shape[1]),
                "mean_norm": float(mean_norm.item()),
                "centered_directions": kept,
                "retained_energy": retained_energy,
                "singular_values": json.dumps(singular_values_list[: max(kept, topk)]),
            })
        if layer_directions:
            candidates[layer_id] = torch.stack(layer_directions, dim=1)
    return candidates, class_stats

