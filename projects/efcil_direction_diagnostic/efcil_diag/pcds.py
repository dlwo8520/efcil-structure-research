from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class APRResult:
    means: torch.Tensor
    covariances: torch.Tensor
    attention_weights: torch.Tensor
    new_displacements: torch.Tensor
    old_corrections: torch.Tensor
    statistics: dict[str, float]


class PrototypeMemory:
    """Compressed class statistics; raw features and image paths are never stored."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = int(feature_dim)
        self.means = torch.empty(0, feature_dim, dtype=torch.float32)
        self.covariances = torch.empty(
            0, feature_dim, feature_dim, dtype=torch.float32
        )
        self.counts = torch.empty(0, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return int(self.means.shape[0])

    def replace_old(self, means: torch.Tensor, covariances: torch.Tensor) -> None:
        if means.shape != self.means.shape:
            raise ValueError(f"Old mean shape changed: {means.shape} != {self.means.shape}")
        if covariances.shape != self.covariances.shape:
            raise ValueError(
                f"Old covariance shape changed: {covariances.shape} != {self.covariances.shape}"
            )
        self.means = means.detach().float().cpu().contiguous()
        self.covariances = covariances.detach().float().cpu().contiguous()

    def append(
        self,
        means: torch.Tensor,
        covariances: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        means = means.detach().float().cpu()
        covariances = covariances.detach().float().cpu()
        counts = counts.detach().long().cpu()
        classes = int(means.shape[0])
        if means.shape != (classes, self.feature_dim):
            raise ValueError(f"Invalid mean shape {means.shape}")
        if covariances.shape != (classes, self.feature_dim, self.feature_dim):
            raise ValueError(f"Invalid covariance shape {covariances.shape}")
        if counts.shape != (classes,) or bool((counts < 2).any()):
            raise ValueError("Each prototype covariance requires at least two samples")
        self.means = torch.cat((self.means, means), dim=0).contiguous()
        self.covariances = torch.cat((self.covariances, covariances), dim=0).contiguous()
        self.counts = torch.cat((self.counts, counts), dim=0).contiguous()

    def export(self) -> dict[str, torch.Tensor]:
        return {
            "means": self.means.cpu(),
            "covariances": self.covariances.cpu(),
            "counts": self.counts.cpu(),
        }

    def load(self, state: dict[str, torch.Tensor]) -> None:
        means = state["means"].detach().float().cpu()
        covariances = state["covariances"].detach().float().cpu()
        counts = state["counts"].detach().long().cpu()
        if means.ndim != 2 or means.shape[1] != self.feature_dim:
            raise ValueError(f"Invalid restored means {means.shape}")
        if covariances.shape != (means.shape[0], self.feature_dim, self.feature_dim):
            raise ValueError(f"Invalid restored covariances {covariances.shape}")
        if counts.shape != (means.shape[0],):
            raise ValueError(f"Invalid restored counts {counts.shape}")
        self.means = means.contiguous()
        self.covariances = covariances.contiguous()
        self.counts = counts.contiguous()


class FeatureStatisticsAccumulator:
    """Streaming per-class mean/covariance for current-task features only."""

    def __init__(
        self,
        begin_class: int,
        num_classes: int,
        feature_dim: int,
        device: torch.device,
    ) -> None:
        self.begin_class = int(begin_class)
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.counts = torch.zeros(num_classes, device=device, dtype=torch.long)
        self.sums = torch.zeros(
            num_classes, feature_dim, device=device, dtype=torch.float32
        )
        self.outer_sums = torch.zeros(
            num_classes, feature_dim, feature_dim, device=device, dtype=torch.float32
        )

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        features = features.detach().to(self.device, dtype=torch.float32)
        local_labels = labels.detach().to(self.device, dtype=torch.long) - self.begin_class
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"Invalid feature batch {features.shape}")
        if bool((local_labels < 0).any()) or bool((local_labels >= self.num_classes).any()):
            raise RuntimeError(
                "Current-task statistics received an old/future class; refusing possible data leakage"
            )
        for class_id in local_labels.unique(sorted=True).tolist():
            selected = features[local_labels == class_id]
            self.counts[class_id] += selected.shape[0]
            self.sums[class_id] += selected.sum(dim=0)
            self.outer_sums[class_id] += selected.transpose(0, 1) @ selected

    @torch.no_grad()
    def finalize(self, epsilon: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool((self.counts < 2).any()):
            raise RuntimeError(f"Insufficient samples for covariance: {self.counts.tolist()}")
        means = self.sums / self.counts[:, None]
        covariances = torch.empty_like(self.outer_sums)
        identity = torch.eye(self.feature_dim, device=self.device, dtype=torch.float32)
        for class_id in range(self.num_classes):
            count = self.counts[class_id].float()
            covariance = (
                self.outer_sums[class_id]
                - count * torch.outer(means[class_id], means[class_id])
            ) / (count - 1.0)
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))
            covariances[class_id] = covariance + float(epsilon) * identity
        return means.cpu(), covariances.cpu(), self.counts.cpu()


def adaptive_prototype_rectification(
    old_means: torch.Tensor,
    old_covariances: torch.Tensor,
    new_means_before: torch.Tensor,
    new_means_after: torch.Tensor,
    gamma: float,
    rho: float,
    epsilon: float,
) -> APRResult:
    """PCDS Eq. (4)-(8), with softmax over current-task classes."""
    old_means = old_means.detach().float()
    old_covariances = old_covariances.detach().float()
    new_means_before = new_means_before.detach().float()
    new_means_after = new_means_after.detach().float()
    if old_means.ndim != 2 or new_means_before.ndim != 2:
        raise ValueError("Prototype means must be matrices")
    if new_means_before.shape != new_means_after.shape:
        raise ValueError("Before/after new-class means must have identical shapes")
    if old_means.shape[1] != new_means_before.shape[1]:
        raise ValueError("Old and new prototype feature dimensions differ")
    if old_covariances.shape != (
        old_means.shape[0], old_means.shape[1], old_means.shape[1]
    ):
        raise ValueError("Old covariance shape does not match old means")
    if old_means.shape[0] == 0:
        empty_weights = torch.empty(0, new_means_before.shape[0])
        empty_corrections = torch.empty(0, old_means.shape[1])
        return APRResult(
            means=old_means,
            covariances=old_covariances,
            attention_weights=empty_weights,
            new_displacements=new_means_after - new_means_before,
            old_corrections=empty_corrections,
            statistics={
                "mean_new_displacement_norm": 0.0,
                "mean_apr_correction_norm": 0.0,
                "max_apr_correction_norm": 0.0,
                "mean_covariance_trace_before": 0.0,
                "mean_covariance_trace_after": 0.0,
                "mean_covariance_frobenius_change": 0.0,
            },
        )

    similarities = F.normalize(old_means, dim=1) @ F.normalize(
        new_means_before, dim=1
    ).transpose(0, 1)
    attention = F.softmax(float(gamma) * similarities, dim=1)
    new_displacements = new_means_after - new_means_before
    old_corrections = attention @ new_displacements
    corrected_means = old_means + old_corrections
    covariance_update = float(rho) * torch.einsum(
        "ki,kj->kij", old_corrections, old_corrections
    )
    identity = torch.eye(old_means.shape[1], dtype=old_means.dtype).unsqueeze(0)
    corrected_covariances = old_covariances + covariance_update + float(epsilon) * identity
    corrected_covariances = 0.5 * (
        corrected_covariances + corrected_covariances.transpose(1, 2)
    )

    displacement_norms = torch.linalg.vector_norm(new_displacements, dim=1)
    correction_norms = torch.linalg.vector_norm(old_corrections, dim=1)
    covariance_changes = torch.linalg.matrix_norm(
        corrected_covariances - old_covariances, ord="fro", dim=(1, 2)
    )
    statistics = {
        "mean_new_displacement_norm": float(displacement_norms.mean().item()),
        "mean_apr_correction_norm": float(correction_norms.mean().item()),
        "max_apr_correction_norm": float(correction_norms.max().item()),
        "mean_covariance_trace_before": float(
            old_covariances.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item()
        ),
        "mean_covariance_trace_after": float(
            corrected_covariances.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item()
        ),
        "mean_covariance_frobenius_change": float(covariance_changes.mean().item()),
    }
    return APRResult(
        means=corrected_means.cpu(),
        covariances=corrected_covariances.cpu(),
        attention_weights=attention.cpu(),
        new_displacements=new_displacements.cpu(),
        old_corrections=old_corrections.cpu(),
        statistics=statistics,
    )


def structural_consistency_loss(
    current_features: torch.Tensor,
    reference_features: torch.Tensor,
    old_prototype_means: torch.Tensor,
) -> torch.Tensor:
    """PCDS Eq. (9); gradients flow only through current_features."""
    if old_prototype_means.numel() == 0:
        return current_features.sum() * 0.0
    anchors = F.normalize(
        old_prototype_means.detach().to(
            device=current_features.device, dtype=current_features.dtype
        ),
        dim=1,
    )
    current_relations = F.normalize(current_features, dim=1) @ anchors.transpose(0, 1)
    reference_relations = F.normalize(
        reference_features.detach().to(
            device=current_features.device, dtype=current_features.dtype
        ),
        dim=1,
    ) @ anchors.transpose(0, 1)
    return (current_relations - reference_relations).square().sum(dim=1).mean()


def stable_covariance_factor(
    covariance: torch.Tensor,
    epsilon: float,
    max_attempts: int = 5,
) -> tuple[torch.Tensor, float, str]:
    """Return L with covariance approximately L @ L.T, adding logged jitter only if needed."""
    covariance = covariance.float()
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    jitter = 0.0
    for attempt in range(max_attempts):
        candidate = covariance + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate)
        if int(info.item()) == 0:
            return factor, jitter, "cholesky"
        jitter = float(epsilon) * (10.0 ** attempt)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = max(float(epsilon), torch.finfo(covariance.dtype).eps)
    factor = eigenvectors @ torch.diag(eigenvalues.clamp_min(floor).sqrt())
    return factor, jitter, "eigh_fallback"
