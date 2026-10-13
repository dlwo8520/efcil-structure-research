from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


EPS = 1e-12


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return float("nan")
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    cosine = safe_cosine(left, right)
    return float(np.degrees(np.arccos(cosine))) if np.isfinite(cosine) else float("nan")


def vector_error(estimate: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    error = float(np.linalg.norm(estimate - oracle))
    return {
        "cosine": safe_cosine(estimate, oracle),
        "l2": error,
        "relative_l2": error / (float(np.linalg.norm(oracle)) + EPS),
        "angle_error_deg": angle_degrees(estimate, oracle),
        "estimate_norm": float(np.linalg.norm(estimate)),
        "oracle_norm": float(np.linalg.norm(oracle)),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 2 or np.std(left) <= EPS or np.std(right) <= EPS:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    keep = np.isfinite(left) & np.isfinite(right)
    if int(keep.sum()) < 2:
        return float("nan")
    return pearson(rankdata(left[keep]), rankdata(right[keep]))


def covariance_metrics(structure: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    structure = np.asarray(structure, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    structure_diag = np.diag(structure)
    oracle_diag = np.diag(oracle)
    diagonal_error = float(np.linalg.norm(structure_diag - oracle_diag))
    full_error = float(np.linalg.norm(structure - oracle, ord="fro"))
    structure_trace = float(np.trace(structure))
    oracle_trace = float(np.trace(oracle))
    structure_fro_sq = float(np.square(structure).sum())
    oracle_fro_sq = float(np.square(oracle).sum())
    return {
        "trace_structure": structure_trace,
        "trace_oracle": oracle_trace,
        "trace_ratio": structure_trace / (oracle_trace + EPS),
        "diag_variance_pearson": pearson(structure_diag, oracle_diag),
        "diag_variance_spearman": spearman(structure_diag, oracle_diag),
        "diag_relative_l2": diagonal_error / (float(np.linalg.norm(oracle_diag)) + EPS),
        "frobenius_relative_error": full_error / (float(np.linalg.norm(oracle, ord="fro")) + EPS),
        "structure_effective_rank_participation": structure_trace**2 / (structure_fro_sq + EPS),
        "oracle_effective_rank_participation": oracle_trace**2 / (oracle_fro_sq + EPS),
    }


def gaussian_shape_metrics(features: np.ndarray) -> dict[str, float]:
    values = np.asarray(features, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=1) if values.shape[0] > 1 else np.zeros(values.shape[1])
    valid = scale > 1e-10
    if not valid.any() or values.shape[0] < 3:
        return {
            "mean_skewness": float("nan"),
            "mean_absolute_skewness": float("nan"),
            "mean_excess_kurtosis": float("nan"),
            "mean_absolute_excess_kurtosis": float("nan"),
            "valid_dimensions": int(valid.sum()),
        }
    standardized = centered[:, valid] / scale[valid]
    skewness = np.mean(np.power(standardized, 3), axis=0)
    excess_kurtosis = np.mean(np.power(standardized, 4), axis=0) - 3.0
    return {
        "mean_skewness": float(np.mean(skewness)),
        "mean_absolute_skewness": float(np.mean(np.abs(skewness))),
        "mean_excess_kurtosis": float(np.mean(excess_kurtosis)),
        "mean_absolute_excess_kurtosis": float(np.mean(np.abs(excess_kurtosis))),
        "valid_dimensions": int(valid.sum()),
    }


def spectrum_metrics(features: np.ndarray) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < 2:
        return {"top1_eigenvalue": float("nan"), "top5_explained_variance": float("nan"), "effective_rank": float("nan")}
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = np.square(singular) / max(features.shape[0] - 1, 1)
    eigenvalues = eigenvalues[eigenvalues > EPS]
    if not eigenvalues.size:
        return {"top1_eigenvalue": 0.0, "top5_explained_variance": 0.0, "effective_rank": 0.0}
    probabilities = eigenvalues / eigenvalues.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities + EPS)))
    return {
        "top1_eigenvalue": float(eigenvalues[0]),
        "top5_explained_variance": float(probabilities[:5].sum()),
        "effective_rank": float(math.exp(entropy)),
    }


def mahalanobis_squared_diagonal(
    features: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    floor: float = 1e-6,
) -> np.ndarray:
    variance = np.maximum(np.diag(np.asarray(covariance, dtype=np.float64)), floor)
    delta = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    return np.sum(np.square(delta) / variance, axis=1)


def mahalanobis_squared_full(
    features: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    floor: float = 1e-6,
) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = (covariance + covariance.T) * 0.5
    delta = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    jitter = floor
    identity = np.eye(covariance.shape[0], dtype=np.float64)
    for _ in range(7):
        try:
            factor = np.linalg.cholesky(covariance + identity * jitter)
            solved = np.linalg.solve(factor, delta.T)
            return np.sum(np.square(solved), axis=0)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    inverse = np.linalg.pinv(covariance + identity * jitter, hermitian=True)
    return np.einsum("ni,ij,nj->n", delta, inverse, delta)


def distribution_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {f"{prefix}_mean": float("nan"), f"{prefix}_median": float("nan"), f"{prefix}_p90": float("nan"), f"{prefix}_p95": float("nan")}
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p90": float(np.quantile(finite, 0.90)),
        f"{prefix}_p95": float(np.quantile(finite, 0.95)),
    }


def cosine_nearest(features: np.ndarray, prototypes: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    proto_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), EPS)
    predictions = []
    for start in range(0, features.shape[0], chunk_size):
        batch = features[start : start + chunk_size]
        batch = batch / np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), EPS)
        predictions.append(np.argmax(batch @ proto_norm.T, axis=1))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def classify_drift(direction_cosine: float, relative_magnitude_error: float) -> str:
    direction_ok = np.isfinite(direction_cosine) and direction_cosine >= 0.8
    magnitude_ok = np.isfinite(relative_magnitude_error) and relative_magnitude_error <= 0.25
    if direction_ok and magnitude_ok:
        return "C_both_match"
    if direction_ok:
        return "A_direction_match_magnitude_mismatch"
    if magnitude_ok:
        return "B_magnitude_match_direction_mismatch"
    return "D_both_mismatch"


@dataclass(frozen=True)
class ClassStats:
    mean: np.ndarray
    covariance: np.ndarray
    features: np.ndarray
