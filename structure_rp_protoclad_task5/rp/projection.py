"""Deterministic, frozen Gaussian random projections used only by Proto-CLAD."""

from __future__ import annotations

import hashlib
import math

import torch
from torch.nn import functional as F


def make_projection(input_dim: int, output_dim: int = 2048, seed: int = 1993) -> torch.Tensor:
    """Return R ~ N(0, 1/input_dim) without changing global RNG state."""
    input_dim = int(input_dim)
    output_dim = int(output_dim)
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        input_dim,
        output_dim,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ) / math.sqrt(input_dim)
    matrix.requires_grad_(False)
    return matrix.contiguous()


def projection_checksum(matrix: torch.Tensor) -> str:
    value = torch.as_tensor(matrix).detach().cpu().contiguous().to(torch.float32)
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def projection_metadata(matrix: torch.Tensor, seed: int) -> dict:
    value = torch.as_tensor(matrix)
    return {
        "seed": int(seed),
        "shape": [int(value.shape[0]), int(value.shape[1])],
        "checksum_sha256": projection_checksum(value),
        "distribution": "gaussian_N(0,1/input_dim)",
        "activation": "relu",
        "bias": False,
        "trainable": False,
        "requires_grad": bool(value.requires_grad),
    }


def project_relu(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or matrix.ndim != 2:
        raise ValueError("features and RP matrix must both be rank-2")
    if features.shape[1] != matrix.shape[0]:
        raise ValueError(
            f"RP input mismatch: features={tuple(features.shape)} matrix={tuple(matrix.shape)}"
        )
    return F.relu(features.float() @ matrix.to(device=features.device, dtype=torch.float32))

