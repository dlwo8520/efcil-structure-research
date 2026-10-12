from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class KeepLoRAInitializationStats:
    layer_id: int
    target: str
    site_key: str
    gradient_norm: float
    projected_gradient_norm: float
    removed_gradient_norm: float
    removed_ratio: float
    protected_rank: int
    residual_rank: int
    effective_rank: int
    aq_norm: float
    initial_delta_norm: float


class _KeepLoRABranchMixin:
    """Shared paper-oriented initialization for PyTorch linear weights.

    The paper writes a projection as x @ A @ B with A=[d_in, r].  PyTorch
    stores W as [d_out, d_in], so this implementation stores A.T as
    ``A_target=[r, d_in]`` and B.T as ``B_target=[d_out, r]``.  Consequently
    the effective PyTorch update is B_target @ A_target.
    """

    layer_id: int
    dim: int
    rank: int
    scaling: float
    targets: tuple[str, ...]
    active: bool
    base: nn.Linear

    def _register_branches(self) -> None:
        for target in self.targets:
            self.register_buffer(f"A_{target}", torch.zeros(self.rank, self.dim))
            self.register_parameter(
                f"B_{target}",
                nn.Parameter(torch.zeros(self.dim, self.rank), requires_grad=False),
            )

    def site_key(self, target: str) -> str:
        if target not in self.targets:
            raise KeyError(target)
        return f"block_{self.layer_id:02d}.{target}"

    def branch_items(self):
        for target in self.targets:
            yield target, getattr(self, f"A_{target}"), getattr(self, f"B_{target}")

    def trainable_B_parameters(self) -> list[nn.Parameter]:
        return [
            getattr(self, f"B_{target}")
            for target in self.targets
            if getattr(self, f"B_{target}").requires_grad
        ]

    def enable_full_weight_gradient(self) -> None:
        self.active = False
        self.base.weight.requires_grad_(True)
        self.base.weight.grad = None
        for target in self.targets:
            getattr(self, f"B_{target}").requires_grad_(False)

    def full_weight_gradient(self, target: str) -> torch.Tensor:
        if self.base.weight.grad is None:
            raise RuntimeError(f"No full-weight gradient for {self.site_key(target)}")
        return self._gradient_slice(target).detach().float().clone()

    def disable_full_weight_gradient(self) -> None:
        self.base.weight.requires_grad_(False)
        self.base.weight.grad = None

    def canonical_weight(self, target: str) -> torch.Tensor:
        """Return the paper/forward-oriented [d_in, d_out] weight."""
        return self._base_slice(target).detach().float().transpose(0, 1).contiguous()

    @torch.no_grad()
    def initialize_branch(
        self,
        target: str,
        gradient_weight: torch.Tensor,
        protected_basis: torch.Tensor | None,
        tolerance: float,
    ) -> KeepLoRAInitializationStats:
        if target not in self.targets:
            raise KeyError(target)
        device = self.base.weight.device
        # PyTorch G_W=[d_out,d_in] -> paper G=[d_in,d_out].
        gradient = gradient_weight.to(device=device, dtype=torch.float32).transpose(0, 1)
        raw_norm = float(torch.linalg.vector_norm(gradient).item())
        if protected_basis is None or protected_basis.numel() == 0:
            Q = torch.empty(self.dim, 0, device=device, dtype=torch.float32)
        else:
            Q = protected_basis.to(device=device, dtype=torch.float32)
            if Q.ndim != 2 or Q.shape[0] != self.dim:
                raise ValueError(
                    f"Protected basis for {self.site_key(target)} has shape {tuple(Q.shape)}"
                )

        projected = gradient
        if Q.numel():
            projected = gradient - Q @ (Q.transpose(0, 1) @ gradient)
        projected_norm = float(torch.linalg.vector_norm(projected).item())
        removed_norm = float(torch.linalg.vector_norm(gradient - projected).item())

        U, singular_values, _ = torch.linalg.svd(projected, full_matrices=False)
        effective_rank = 0
        if singular_values.numel() and float(singular_values[0].item()) > 0.0:
            cutoff = tolerance * max(float(singular_values[0].item()), 1.0)
            effective_rank = min(
                self.rank,
                int((singular_values > cutoff).sum().item()),
                self.dim - int(Q.shape[1]),
            )

        paper_A = torch.zeros(self.dim, self.rank, device=device, dtype=torch.float32)
        if effective_rank:
            selected = U[:, :effective_rank]
            if Q.numel():
                selected = selected - Q @ (Q.transpose(0, 1) @ selected)
            selected, _ = torch.linalg.qr(selected, mode="reduced")
            paper_A[:, :effective_rank] = selected[:, :effective_rank]

        # Least-squares coefficients equal S V.T for exact singular vectors.
        paper_B = paper_A.transpose(0, 1) @ projected
        code_A = paper_A.transpose(0, 1).contiguous()
        code_B = paper_B.transpose(0, 1).contiguous()
        getattr(self, f"A_{target}").copy_(code_A.to(self.base.weight.dtype))
        getattr(self, f"B_{target}").copy_(code_B.to(self.base.weight.dtype))
        getattr(self, f"B_{target}").requires_grad_(True)

        delta_weight = self.scaling * (code_B @ code_A)
        self._base_slice(target).sub_(delta_weight.to(self.base.weight.dtype))
        self.active = True
        aq_norm = (
            float(torch.linalg.vector_norm(code_A @ Q).item()) if Q.numel() else 0.0
        )
        return KeepLoRAInitializationStats(
            layer_id=self.layer_id,
            target=target,
            site_key=self.site_key(target),
            gradient_norm=raw_norm,
            projected_gradient_norm=projected_norm,
            removed_gradient_norm=removed_norm,
            removed_ratio=removed_norm / (raw_norm + 1.0e-12),
            protected_rank=int(Q.shape[1]),
            residual_rank=self.dim - int(Q.shape[1]),
            effective_rank=effective_rank,
            aq_norm=aq_norm,
            initial_delta_norm=float(torch.linalg.vector_norm(delta_weight).item()),
        )

    def effective_delta_from_B_delta(
        self, target: str, delta_B: torch.Tensor
    ) -> torch.Tensor:
        A = getattr(self, f"A_{target}").to(dtype=delta_B.dtype)
        return self.scaling * (delta_B @ A)

    @torch.no_grad()
    def merge_and_reset(self) -> None:
        if not self.active:
            return
        for target, A, B in self.branch_items():
            self._base_slice(target).add_(
                (self.scaling * (B @ A)).to(self.base.weight.dtype)
            )
            A.zero_()
            B.zero_()
            B.requires_grad_(False)
        self.active = False


class KeepLoRAQKV(nn.Module, _KeepLoRABranchMixin):
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
        if not target_set or not target_set.issubset({"q", "k", "v"}):
            raise ValueError(f"Invalid QKV targets: {target_set}")
        self.base = base
        self.layer_id = int(layer_id)
        self.dim = int(base.in_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.targets = tuple(target for target in ("q", "k", "v") if target in target_set)
        self.active = False
        self.capture_callback: Callable[[str, torch.Tensor], None] | None = None
        self.capture_group = f"block_{self.layer_id:02d}.qkv_input"
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._register_branches()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture_callback is not None:
            self.capture_callback(self.capture_group, x)
        qkv = self.base(x)
        if not self.active:
            return qkv
        chunks = list(qkv.chunk(3, dim=-1))
        target_index = {"q": 0, "k": 1, "v": 2}
        for target in self.targets:
            A = getattr(self, f"A_{target}").to(dtype=x.dtype)
            B = getattr(self, f"B_{target}").to(dtype=x.dtype)
            chunks[target_index[target]] = chunks[target_index[target]] + self.scaling * (
                (x @ A.transpose(0, 1)) @ B.transpose(0, 1)
            )
        return torch.cat(chunks, dim=-1)

    def _base_slice(self, target: str) -> torch.Tensor:
        index = {"q": 0, "k": 1, "v": 2}[target]
        return self.base.weight[index * self.dim : (index + 1) * self.dim]

    def _gradient_slice(self, target: str) -> torch.Tensor:
        index = {"q": 0, "k": 1, "v": 2}[target]
        return self.base.weight.grad[index * self.dim : (index + 1) * self.dim]


class KeepLoRAOutputProjection(nn.Module, _KeepLoRABranchMixin):
    def __init__(
        self,
        base: nn.Linear,
        layer_id: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if base.in_features != base.out_features:
            raise ValueError("KeepLoRA output projection currently requires a square weight")
        self.base = base
        self.layer_id = int(layer_id)
        self.dim = int(base.in_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.targets = ("o",)
        self.active = False
        self.capture_callback: Callable[[str, torch.Tensor], None] | None = None
        self.capture_group = f"block_{self.layer_id:02d}.o_input"
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self._register_branches()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.capture_callback is not None:
            self.capture_callback(self.capture_group, x)
        output = self.base(x)
        if self.active:
            A = self.A_o.to(dtype=x.dtype)
            B = self.B_o.to(dtype=x.dtype)
            output = output + self.scaling * (
                (x @ A.transpose(0, 1)) @ B.transpose(0, 1)
            )
        return output

    def _base_slice(self, target: str) -> torch.Tensor:
        if target != "o":
            raise KeyError(target)
        return self.base.weight

    def _gradient_slice(self, target: str) -> torch.Tensor:
        if target != "o":
            raise KeyError(target)
        return self.base.weight.grad

