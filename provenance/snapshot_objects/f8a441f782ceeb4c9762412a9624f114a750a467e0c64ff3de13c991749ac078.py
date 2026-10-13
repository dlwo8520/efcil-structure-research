"""Configurable AMGC-inspired E3 threshold recalibration.

AMGC's original AMarX is a classifier-margin derivation.  The rules below do
not reproduce AMarX: they are deliberately simple, independently switched E3
threshold extensions using only stored/APR-corrected statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch


VALID_MODES = {"none", "variance", "age", "drift"}


@dataclass
class AdaptiveMarginResult:
    taus: torch.Tensor
    rows: List[Dict[str, float]]


def recalibrate_taus(
    *,
    birth_taus: torch.Tensor,
    variances: torch.Tensor,
    class_ids: torch.Tensor,
    birth_tasks: torch.Tensor,
    current_task: int,
    drift_norms: torch.Tensor,
    mode: str,
    strength: float,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
    eps: float = 1e-8,
) -> AdaptiveMarginResult:
    """Recalibrate each region independently without old raw exemplars.

    variance: larger-than-median diagonal variance increases tau.
    age: older classes increase tau linearly.
    drift: larger APR displacement *reduces* tau (relaxes an uncertain region).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown adaptive margin mode: {mode}")
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("invalid adaptive scale bounds")
    tensors = (birth_taus, class_ids, birth_tasks, drift_norms)
    if any(tensor.numel() != birth_taus.numel() for tensor in tensors):
        raise ValueError("adaptive-margin metadata lengths differ")
    if variances.ndim != 2 or variances.shape[0] != birth_taus.numel():
        raise ValueError("variances must be [regions, dimensions]")

    mean_variance = variances.float().mean(dim=1)
    max_variance = variances.float().max(dim=1).values
    trace = variances.float().sum(dim=1)
    ages = (int(current_task) - birth_tasks.long()).clamp_min(0).float()

    if mode == "none":
        scale = torch.ones_like(birth_taus.float())
    elif mode == "variance":
        reference = mean_variance.median().clamp_min(eps)
        scale = (mean_variance / reference).clamp_min(eps).pow(float(strength))
    elif mode == "age":
        scale = 1.0 + float(strength) * ages
    else:
        positive = drift_norms.float()[drift_norms.float() > 0]
        reference = positive.median().clamp_min(eps) if positive.numel() else torch.tensor(1.0)
        scale = 1.0 / (1.0 + float(strength) * drift_norms.float() / reference)
    scale = scale.clamp(float(min_scale), float(max_scale))
    taus = birth_taus.float() * scale

    rows: List[Dict[str, float]] = []
    for index in range(birth_taus.numel()):
        rows.append(
            {
                "class_id": int(class_ids[index]),
                "birth_task": int(birth_tasks[index]),
                "class_age": float(ages[index]),
                "variance_trace": float(trace[index]),
                "variance_mean": float(mean_variance[index]),
                "variance_max": float(max_variance[index]),
                "apr_shift_norm": float(drift_norms[index]),
                "tau_birth": float(birth_taus[index]),
                "adaptive_scale": float(scale[index]),
                "tau": float(taus[index]),
            }
        )
    return AdaptiveMarginResult(taus=taus, rows=rows)
