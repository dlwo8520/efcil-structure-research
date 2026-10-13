"""Unbiased pooled covariance including between-source scatter."""
from __future__ import annotations

import numpy as np


def mixed_moments(real, generated):
    real, generated = np.asarray(real, np.float64), np.asarray(generated, np.float64)
    n, m = len(real), len(generated)
    if n < 1 or m < 1 or n + m < 2:
        raise ValueError("mixed statistics need nonempty real/generated sources")
    mean_r, mean_g = real.mean(0), generated.mean(0)
    centered_r, centered_g = real - mean_r, generated - mean_g
    scatter = (centered_r.T @ centered_r + centered_g.T @ centered_g
               + n * m / (n + m) * np.outer(mean_r - mean_g, mean_r - mean_g))
    return (n * mean_r + m * mean_g) / (n + m), scatter / (n + m - 1)
