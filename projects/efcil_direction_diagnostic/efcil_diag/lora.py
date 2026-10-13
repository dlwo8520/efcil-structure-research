from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAInitializationStats:
    layer_id: int
    target: str
    gradient_norm: float
    projected_gradient_norm: float
    removed_gradient_norm: float
    removed_ratio: float
    protected_rank: int
    effective_rank: int
    aq_norm: float


class QVFixedBasisLoRA(nn.Module):
    """Wrap a timm fused QKV projection with temporary Q/V LoRA branches.

    PyTorch stores a linear weight as [d_out, d_in]. Each branch uses
    delta_W = scaling * B @ A, where A is a fixed input-space basis.
    """

    def __init__(
        self,
        base: nn.Linear,
        layer_id: int,
        rank: int,
        alpha: float,
        targets: Iterable[str],
    ) -> None:
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError("Expected a fused QKV projection")
        target_set = set(targets)
        if not target_set.issubset({"q", "v"}):
            raise ValueError(f"Only q/v targets are supported, received {target_set}")
        self.base = base
        self.layer_id = int(layer_id)
        self.dim = int(base.in_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.targets = tuple(sorted(target_set))
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        for target in ("q", "v"):
            self.register_buffer(f"A_{target}", torch.zeros(rank, self.dim))
            parameter = nn.Parameter(torch.zeros(self.dim, rank), requires_grad=False)
            self.register_parameter(f"B_{target}", parameter)
        self.active = False
        self.capture_callback: Callable[[int, torch.Tensor], None] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture_callback is not None:
            self.capture_callback(self.layer_id, x)
        qkv = self.base(x)
        if not self.active:
            return qkv
        q, k, v = qkv.chunk(3, dim=-1)
        if "q" in self.targets:
            q = q + self._branch_forward(x, "q")
        if "v" in self.targets:
            v = v + self._branch_forward(x, "v")
        return torch.cat((q, k, v), dim=-1)

    def _branch_forward(self, x: torch.Tensor, target: str) -> torch.Tensor:
        A = getattr(self, f"A_{target}").to(dtype=x.dtype)
        B = getattr(self, f"B_{target}").to(dtype=x.dtype)
        return self.scaling * ((x @ A.transpose(0, 1)) @ B.transpose(0, 1))

    def enable_full_weight_gradient(self) -> None:
        self.active = False
        self.base.weight.requires_grad_(True)
        self.base.weight.grad = None
        for target in self.targets:
            getattr(self, f"B_{target}").requires_grad_(False)

    def full_weight_gradient(self, target: str) -> torch.Tensor:
        if self.base.weight.grad is None:
            raise RuntimeError(f"No full-weight gradient for layer {self.layer_id}")
        if target == "q":
            return self.base.weight.grad[: self.dim].detach().float().clone()
        if target == "v":
            return self.base.weight.grad[2 * self.dim :].detach().float().clone()
        raise KeyError(target)

    def disable_full_weight_gradient(self) -> None:
        self.base.weight.requires_grad_(False)
        self.base.weight.grad = None

    @torch.no_grad()
    def initialize_branch(
        self,
        target: str,
        gradient: torch.Tensor,
        protected_basis: torch.Tensor | None,
        orthogonal: bool,
        tolerance: float,
    ) -> LoRAInitializationStats:
        if target not in self.targets:
            raise KeyError(target)
        device = self.base.weight.device
        gradient = gradient.to(device=device, dtype=torch.float32)
        raw_norm = float(torch.linalg.vector_norm(gradient).item())
        Q = None
        if protected_basis is not None and protected_basis.numel() > 0:
            Q = protected_basis.to(device=device, dtype=torch.float32)
        projected = gradient
        if orthogonal and Q is not None:
            projected = gradient - (gradient @ Q) @ Q.transpose(0, 1)
        projected_norm = float(torch.linalg.vector_norm(projected).item())
        removed = gradient - projected
        removed_norm = float(torch.linalg.vector_norm(removed).item())

        _, singular_values, vh = torch.linalg.svd(projected, full_matrices=False)
        candidate_rows = vh[: min(self.rank, vh.shape[0])]
        selected: list[torch.Tensor] = []
        scale = max(float(singular_values[0].item()) if singular_values.numel() else 0.0, 1.0)
        for index, candidate in enumerate(candidate_rows):
            if index < singular_values.numel() and float(singular_values[index].item()) <= tolerance * scale:
                continue
            vector = candidate.clone()
            if orthogonal and Q is not None:
                vector -= Q @ (Q.transpose(0, 1) @ vector)
            for previous in selected:
                vector -= previous * torch.dot(previous, vector)
            norm = torch.linalg.vector_norm(vector)
            if float(norm.item()) > tolerance:
                selected.append(vector / norm)

        A = torch.zeros(self.rank, self.dim, device=device, dtype=torch.float32)
        effective_rank = len(selected)
        if selected:
            A[:effective_rank] = torch.stack(selected)
        # Least-squares coefficient for an orthonormal row basis.
        B = torch.zeros(self.dim, self.rank, device=device, dtype=torch.float32)
        if effective_rank:
            B[:, :effective_rank] = projected @ A[:effective_rank].transpose(0, 1)

        getattr(self, f"A_{target}").copy_(A.to(self.base.weight.dtype))
        getattr(self, f"B_{target}").copy_(B.to(self.base.weight.dtype))
        getattr(self, f"B_{target}").requires_grad_(True)

        delta = self.scaling * (B @ A)
        self._base_slice(target).sub_(delta.to(self.base.weight.dtype))
        self.active = True
        aq_norm = 0.0
        if Q is not None:
            aq_norm = float(torch.linalg.vector_norm(A @ Q).item())
        return LoRAInitializationStats(
            layer_id=self.layer_id,
            target=target,
            gradient_norm=raw_norm,
            projected_gradient_norm=projected_norm,
            removed_gradient_norm=removed_norm,
            removed_ratio=removed_norm / (raw_norm + 1.0e-12),
            protected_rank=0 if Q is None else int(Q.shape[1]),
            effective_rank=effective_rank,
            aq_norm=aq_norm,
        )

    def _base_slice(self, target: str) -> torch.Tensor:
        if target == "q":
            return self.base.weight[: self.dim]
        if target == "v":
            return self.base.weight[2 * self.dim :]
        raise KeyError(target)

    @torch.no_grad()
    def merge_and_reset(self) -> None:
        if not self.active:
            return
        for target in self.targets:
            A = getattr(self, f"A_{target}")
            B = getattr(self, f"B_{target}")
            delta = self.scaling * (B @ A)
            self._base_slice(target).add_(delta.to(self.base.weight.dtype))
            A.zero_()
            B.zero_()
            B.requires_grad_(False)
        self.active = False

    def trainable_B_parameters(self) -> list[nn.Parameter]:
        if not self.active:
            return []
        return [getattr(self, f"B_{target}") for target in self.targets]

    def branch_items(self) -> list[tuple[str, torch.Tensor, nn.Parameter]]:
        return [
            (target, getattr(self, f"A_{target}"), getattr(self, f"B_{target}"))
            for target in self.targets
        ]

    def effective_delta_from_B_delta(self, target: str, delta_B: torch.Tensor) -> torch.Tensor:
        A = getattr(self, f"A_{target}")
        return self.scaling * (delta_B @ A)


def install_qv_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float,
    targets: Iterable[str],
) -> list[QVFixedBasisLoRA]:
    wrappers: list[QVFixedBasisLoRA] = []
    if not hasattr(backbone, "blocks"):
        raise TypeError("Backbone does not expose transformer blocks")
    for layer_id, block in enumerate(backbone.blocks):
        wrapper = QVFixedBasisLoRA(
            block.attn.qkv,
            layer_id=layer_id,
            rank=rank,
            alpha=alpha,
            targets=targets,
        )
        block.attn.qkv = wrapper
        wrappers.append(wrapper)
    return wrappers

