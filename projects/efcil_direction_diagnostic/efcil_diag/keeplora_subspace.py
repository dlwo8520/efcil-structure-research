from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from .keeplora_model import KeepLoRAContinualViT


@dataclass(frozen=True)
class ProtectedSpaceStats:
    site_key: str
    layer_id: int
    target: str
    feature_dim: int
    wp_enabled: int
    m_enabled: int
    wp_rank: int
    old_m_rank: int
    added_m_rank: int
    m_rank: int
    unified_rank: int
    residual_rank: int
    protected_ratio: float
    epsilon_f: float
    existing_captured_ratio: float
    retained_energy_ratio: float


def _orthonormalize(matrix: torch.Tensor, tolerance: float) -> torch.Tensor:
    if matrix.numel() == 0:
        return torch.empty(matrix.shape[0], 0, device=matrix.device, dtype=torch.float32)
    q, r = torch.linalg.qr(matrix.float(), mode="reduced")
    diagonal = torch.abs(torch.diagonal(r))
    scale = max(float(diagonal.max().item()) if diagonal.numel() else 0.0, 1.0)
    keep = diagonal > tolerance * scale
    return q[:, keep].contiguous()


class KeepLoRASubspaceBank:
    def __init__(self, site_keys: list[str], feature_dim: int) -> None:
        self.site_keys = list(site_keys)
        self.feature_dim = int(feature_dim)
        self.wp_bases = {
            key: torch.empty(feature_dim, 0, dtype=torch.float32) for key in site_keys
        }
        self.m_bases = {
            key: torch.empty(feature_dim, 0, dtype=torch.float32) for key in site_keys
        }

    @torch.no_grad()
    def initialize_wp(
        self,
        model: KeepLoRAContinualViT,
        epsilon_w: float,
        tolerance: float,
    ) -> list[dict]:
        if not 0.0 < epsilon_w <= 1.0:
            raise ValueError("epsilon_w must be in (0, 1]")
        rows: list[dict] = []
        for module, target in model.projection_sites():
            site_key = module.site_key(target)
            weight = module.canonical_weight(target)
            U, singular_values, _ = torch.linalg.svd(weight, full_matrices=False)
            energies = singular_values.square()
            total = float(energies.sum().item())
            if total <= tolerance:
                selected_rank = 0
                retained = 0.0
                basis = torch.empty(self.feature_dim, 0, device=weight.device)
            else:
                cumulative = torch.cumsum(energies, dim=0) / total
                reached = torch.nonzero(cumulative >= epsilon_w)
                selected_rank = (
                    int(reached[0].item()) + 1 if reached.numel() else len(singular_values)
                )
                basis = U[:, :selected_rank]
                retained = float(cumulative[selected_rank - 1].item())
            basis = _orthonormalize(basis, tolerance)
            self.wp_bases[site_key] = basis.detach().cpu()
            rows.append({
                "site_key": site_key,
                "layer_id": module.layer_id,
                "target": target,
                "weight_shape": json.dumps(list(weight.shape)),
                "feature_dim": self.feature_dim,
                "epsilon_w": epsilon_w,
                "selected_rank": int(basis.shape[1]),
                "retained_energy_ratio": retained,
                "singular_values": json.dumps(singular_values.detach().cpu().tolist()),
            })
        return rows

    def protected_basis(
        self,
        site_key: str,
        use_wp: bool,
        use_m: bool,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        if use_wp:
            pieces.append(self.wp_bases[site_key])
        if use_m:
            pieces.append(self.m_bases[site_key])
        if not pieces:
            basis = torch.empty(self.feature_dim, 0, dtype=torch.float32)
        else:
            basis = torch.cat(pieces, dim=1)
        if device is not None:
            basis = basis.to(device=device)
        return basis

    @torch.no_grad()
    def update_task_features(
        self,
        model: KeepLoRAContinualViT,
        feature_matrices: dict[str, torch.Tensor],
        use_wp: bool,
        use_m: bool,
        epsilon_f: float,
        tolerance: float,
    ) -> tuple[list[ProtectedSpaceStats], list[dict]]:
        if not 0.0 < epsilon_f <= 1.0:
            raise ValueError("epsilon_f must be in (0, 1]")
        stats: list[ProtectedSpaceStats] = []
        spectra: list[dict] = []
        for module, target in model.projection_sites():
            site_key = module.site_key(target)
            group = module.capture_group
            old_m = self.m_bases[site_key]
            old_m_rank = int(old_m.shape[1])
            wp_rank = int(self.wp_bases[site_key].shape[1])
            added_rank = 0
            existing_ratio = 0.0
            retained_ratio = 0.0
            singular_values = torch.empty(0)

            if use_m:
                if group not in feature_matrices:
                    raise KeyError(f"Missing feature matrix for {group}")
                features = feature_matrices[group].float()
                if features.ndim != 2 or features.shape[0] != self.feature_dim:
                    raise ValueError(f"Invalid feature shape for {group}: {features.shape}")
                Q = self.protected_basis(site_key, use_wp, True, features.device)
                total_energy = float(features.square().sum().item())
                residual = features
                if Q.numel():
                    residual = features - Q @ (Q.transpose(0, 1) @ features)
                residual_energy = float(residual.square().sum().item())
                if total_energy > tolerance:
                    existing_ratio = min(
                        1.0, max(0.0, (total_energy - residual_energy) / total_energy)
                    )
                selected_rank = 0
                selected = torch.empty(
                    self.feature_dim, 0, device=features.device, dtype=torch.float32
                )
                if residual_energy > tolerance * max(total_energy, 1.0):
                    U, singular_values, _ = torch.linalg.svd(
                        residual, full_matrices=False
                    )
                    cumulative = (
                        (total_energy - residual_energy)
                        + torch.cumsum(singular_values.square(), dim=0)
                    ) / max(total_energy, tolerance)
                    reached = torch.nonzero(cumulative >= epsilon_f)
                    selected_rank = (
                        int(reached[0].item()) + 1
                        if reached.numel()
                        else int(singular_values.numel())
                    )
                    selected_rank = min(
                        selected_rank,
                        self.feature_dim - int(Q.shape[1]),
                    )
                    selected = U[:, :selected_rank]
                    if Q.numel() and selected.numel():
                        selected = selected - Q @ (Q.transpose(0, 1) @ selected)
                        selected = _orthonormalize(selected, tolerance)
                    if selected_rank:
                        retained_ratio = float(cumulative[selected_rank - 1].item())
                    else:
                        retained_ratio = existing_ratio
                else:
                    retained_ratio = existing_ratio

                if selected.numel():
                    old_m_device = old_m.to(features.device)
                    if use_wp:
                        wp = self.wp_bases[site_key].to(features.device)
                        combined = _orthonormalize(
                            torch.cat((wp, old_m_device, selected), dim=1), tolerance
                        )
                        new_m = combined[:, wp.shape[1] :]
                    else:
                        new_m = _orthonormalize(
                            torch.cat((old_m_device, selected), dim=1), tolerance
                        )
                    self.m_bases[site_key] = new_m.detach().cpu()
                added_rank = int(self.m_bases[site_key].shape[1]) - old_m_rank

            m_rank = int(self.m_bases[site_key].shape[1]) if use_m else 0
            unified_rank = min(
                self.feature_dim,
                (wp_rank if use_wp else 0) + (m_rank if use_m else 0),
            )
            stats.append(ProtectedSpaceStats(
                site_key=site_key,
                layer_id=module.layer_id,
                target=target,
                feature_dim=self.feature_dim,
                wp_enabled=int(use_wp),
                m_enabled=int(use_m),
                wp_rank=wp_rank,
                old_m_rank=old_m_rank if use_m else 0,
                added_m_rank=added_rank,
                m_rank=m_rank,
                unified_rank=unified_rank,
                residual_rank=self.feature_dim - unified_rank,
                protected_ratio=unified_rank / self.feature_dim,
                epsilon_f=epsilon_f,
                existing_captured_ratio=existing_ratio,
                retained_energy_ratio=retained_ratio,
            ))
            spectra.append({
                "site_key": site_key,
                "layer_id": module.layer_id,
                "target": target,
                "feature_group": group,
                "epsilon_f": epsilon_f,
                "existing_captured_ratio": existing_ratio,
                "retained_energy_ratio": retained_ratio,
                "added_rank": added_rank,
                "singular_values": json.dumps(singular_values.detach().cpu().tolist()),
            })
        return stats, spectra

    def export(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "wp_bases": {key: value.cpu() for key, value in self.wp_bases.items()},
            "m_bases": {key: value.cpu() for key, value in self.m_bases.items()},
        }

    def load(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for bank_name in ("wp_bases", "m_bases"):
            values = state[bank_name]
            if set(values) != set(self.site_keys):
                raise ValueError(f"{bank_name} site keys do not match the current model")
            for value in values.values():
                if value.ndim != 2 or value.shape[0] != self.feature_dim:
                    raise ValueError(f"Invalid protected basis shape {value.shape}")
            setattr(
                self,
                bank_name,
                {key: value.detach().float().cpu().contiguous() for key, value in values.items()},
            )


class OfficialStyleFeatureCollector:
    """Collect the official KeepLoRA batch-position feature mean.

    The official implementation averages d x (batch*tokens) matrices across
    equal-sized batches.  Its extra global scalar does not change singular
    directions or energy ratios, so this stores the numerically stable mean.
    """

    def __init__(self) -> None:
        self.matrices: dict[str, torch.Tensor] = {}
        self.batches: dict[str, int] = {}

    def __call__(self, group: str, activation: torch.Tensor) -> None:
        if activation.ndim not in (2, 3):
            raise ValueError(f"Expected 2D/3D activation, got {activation.shape}")
        flattened = activation.detach().reshape(-1, activation.shape[-1])
        current = flattened.transpose(0, 1).float()
        count = self.batches.get(group, 0)
        if count == 0:
            self.matrices[group] = current.clone()
        else:
            if self.matrices[group].shape != current.shape:
                raise ValueError(
                    f"Feature batch shape changed for {group}: "
                    f"{self.matrices[group].shape} -> {current.shape}"
                )
            self.matrices[group].mul_(count / (count + 1.0)).add_(
                current, alpha=1.0 / (count + 1.0)
            )
        self.batches[group] = count + 1


@torch.no_grad()
def collect_official_style_features(
    model: KeepLoRAContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int | None,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    collector = OfficialStyleFeatureCollector()
    model.set_capture_callbacks(collector)
    model.eval()
    observed = 0
    try:
        for images, _ in loader:
            if max_batches is not None and observed >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                model.features(images)
            observed += 1
    finally:
        model.clear_capture_callbacks()
    if observed == 0:
        raise RuntimeError("Feature loader was empty")
    rows = [
        {
            "feature_group": group,
            "batches": collector.batches[group],
            "matrix_rows": matrix.shape[0],
            "matrix_columns": matrix.shape[1],
            "matrix_norm": float(torch.linalg.vector_norm(matrix).item()),
        }
        for group, matrix in sorted(collector.matrices.items())
    ]
    return collector.matrices, rows
