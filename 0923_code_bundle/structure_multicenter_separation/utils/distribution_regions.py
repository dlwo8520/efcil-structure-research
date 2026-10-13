"""Multi-center statistics, Gaussian replay, and old-region separation.

The multi-center representation is inspired by MPFR's per-class K-means
representation. Full local covariance, APR propagation, fixed birth quantiles,
and the normalized Mahalanobis hinge are project-specific extensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import torch
from torch.nn import functional as F


COVARIANCE_VIEWS = ("isotropic", "diagonal", "full")


def _as_cpu_float(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {tuple(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("cannot estimate a distribution from zero features")
    if not torch.isfinite(features).all():
        raise ValueError("features contain NaN or Inf")
    return features.detach().to(device="cpu", dtype=torch.float32).contiguous()


def empirical_covariance(features: torch.Tensor, jitter: float) -> Tuple[torch.Tensor, int, str]:
    """Return a symmetric full covariance with a deterministic small-sample rule."""
    x = _as_cpu_float(features)
    count, dimension = x.shape
    centered = x - x.mean(dim=0, keepdim=True)
    rank = int(torch.linalg.matrix_rank(centered).item()) if count > 1 else 0
    if count == 1:
        covariance = torch.zeros((dimension, dimension), dtype=torch.float32)
        fallback = "singleton_zero_covariance_plus_jitter"
    else:
        covariance = centered.T @ centered / float(count - 1)
        fallback = "none"
    covariance = 0.5 * (covariance + covariance.T)
    covariance.diagonal().add_(float(jitter))
    return covariance, rank, fallback


def _squared_distances(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (
        x.square().sum(dim=1, keepdim=True)
        + centers.square().sum(dim=1).unsqueeze(0)
        - 2.0 * x @ centers.T
    ).clamp_min_(0.0)


def _kmeans_plus_plus(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    count = x.shape[0]
    first = int(torch.randint(count, (1,), generator=generator).item())
    selected = [first]
    closest = _squared_distances(x, x[first : first + 1]).squeeze(1)
    for _ in range(1, k):
        total = closest.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            remaining = [idx for idx in range(count) if idx not in selected]
            selected.append(remaining[0])
        else:
            candidate = int(torch.multinomial(closest / total, 1, generator=generator).item())
            if candidate in selected:
                remaining = [idx for idx in range(count) if idx not in selected]
                candidate = remaining[0]
            selected.append(candidate)
        distance = _squared_distances(x, x[selected[-1] : selected[-1] + 1]).squeeze(1)
        closest = torch.minimum(closest, distance)
    return x[selected].clone()


def deterministic_kmeans(
    features: torch.Tensor,
    requested_k: int,
    seed: int,
    n_init: int,
    max_iter: int,
    min_cluster_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """K-means++ with a separate CPU RNG and deterministic K reduction.

    K starts at min(requested K, unique samples, floor(N/min_cluster_size)). If
    every initialization produces an empty/undersized cluster, K is reduced by
    one and retried. Ties are resolved by the lowest component index.
    """
    x = _as_cpu_float(features)
    if requested_k < 1 or n_init < 1 or max_iter < 1 or min_cluster_size < 1:
        raise ValueError("invalid K-means configuration")
    unique_count = int(torch.unique(x, dim=0).shape[0])
    initial_k = min(requested_k, unique_count, max(1, x.shape[0] // min_cluster_size))
    fallback_reasons = []
    if unique_count < requested_k:
        fallback_reasons.append("fewer_unique_features_than_requested_k")
    if x.shape[0] // min_cluster_size < requested_k:
        fallback_reasons.append("insufficient_samples_for_min_cluster_size")
    best = None
    attempted: List[Dict[str, Any]] = []
    for k in range(initial_k, 0, -1):
        for initialization in range(n_init):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + 104729 * k + initialization)
            centers = _kmeans_plus_plus(x, k, generator)
            assignments = torch.full((x.shape[0],), -1, dtype=torch.long)
            for iteration in range(max_iter):
                updated = _squared_distances(x, centers).argmin(dim=1)
                if torch.equal(updated, assignments):
                    break
                assignments = updated
                counts = torch.bincount(assignments, minlength=k)
                if int(counts.min().item()) == 0:
                    break
                centers = torch.stack([x[assignments == idx].mean(dim=0) for idx in range(k)])
            counts = torch.bincount(assignments.clamp_min(0), minlength=k)
            singleton_class_fallback = k == 1 and x.shape[0] < min_cluster_size
            valid = assignments.min().item() >= 0 and (
                int(counts.min().item()) >= min_cluster_size or singleton_class_fallback
            )
            inertia = float(_squared_distances(x, centers).gather(1, assignments.clamp_min(0)[:, None]).sum())
            attempted.append(
                {"k": k, "initialization": initialization, "valid": bool(valid), "inertia": inertia}
            )
            if valid and (best is None or inertia < best[0]):
                best = (inertia, centers.clone(), assignments.clone(), iteration + 1, counts.clone())
        if best is not None:
            break
    if best is None:
        raise RuntimeError("deterministic K-means failed even for K=1")
    _, centers, assignments, iterations, counts = best
    if centers.shape[0] < initial_k:
        fallback_reasons.append("empty_or_undersized_cluster_reduced_k_and_reran")
    if x.shape[0] < min_cluster_size:
        fallback_reasons.append("singleton_or_tiny_class_kept_as_one_cluster")
    metadata = {
        "requested_k": int(requested_k),
        "actual_k": int(centers.shape[0]),
        "unique_count": unique_count,
        "initialization": "kmeans++",
        "n_init": int(n_init),
        "max_iter": int(max_iter),
        "iterations": int(iterations),
        "min_cluster_size": int(min_cluster_size),
        "counts": counts.tolist(),
        "reduced_k": int(centers.shape[0]) != int(requested_k),
        "fallback_applied": bool(fallback_reasons),
        "fallback_reasons": fallback_reasons,
        "small_cluster_policy": "reduce_k_and_rerun_without_merging_or_dropping_samples",
        "assignments": assignments.tolist(),
        "attempts": attempted,
    }
    return centers, assignments, metadata


def factorize_full_covariance(
    covariance: torch.Tensor,
    eigen_floor: float,
    initial_jitter: float,
    jitter_multiplier: float,
    max_tries: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Stabilize and Cholesky-factor a covariance without diagonal fallback."""
    covariance = covariance.detach().to(device="cpu", dtype=torch.float32)
    covariance = 0.5 * (covariance + covariance.T)
    dimension = covariance.shape[0]
    identity = torch.eye(dimension, dtype=covariance.dtype)
    jitter = 0.0
    for attempt in range(max_tries + 1):
        candidate = covariance if attempt == 0 else covariance + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate)
        if int(info.item()) == 0 and torch.isfinite(factor).all():
            return candidate, factor, {
                "factorization": "cholesky",
                "jitter": float(jitter),
                "attempts": int(attempt + 1),
                "eigen_clip": False,
                "fallback_applied": float(jitter) > 0.0,
            }
        jitter = float(initial_jitter) if attempt == 0 else jitter * float(jitter_multiplier)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    clipped = eigenvalues.clamp_min(float(eigen_floor))
    candidate = (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.T
    candidate = 0.5 * (candidate + candidate.T)
    factor = torch.linalg.cholesky(candidate)
    return candidate, factor, {
        "factorization": "eigenvalue_clip_then_cholesky",
        "jitter": float(jitter),
        "attempts": int(max_tries + 2),
        "eigen_clip": True,
        "fallback_applied": True,
        "min_eigenvalue_before": float(eigenvalues.min().item()),
        "min_eigenvalue_after": float(clipped.min().item()),
    }


def mahalanobis_squared(
    features: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    covariance_view: str,
    eigen_floor: float,
    initial_jitter: float,
    jitter_multiplier: float,
    max_tries: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    x = _as_cpu_float(features)
    mean = mean.detach().to(device="cpu", dtype=torch.float32)
    covariance = covariance.detach().to(device="cpu", dtype=torch.float32)
    difference = x - mean
    if covariance_view == "isotropic":
        scale = max(float(torch.trace(covariance).item() / covariance.shape[0]), float(eigen_floor))
        return difference.square().sum(dim=1) / scale, {"isotropic_scale": scale}
    if covariance_view == "diagonal":
        diagonal = covariance.diagonal().clamp_min(float(eigen_floor))
        return (difference.square() / diagonal).sum(dim=1), {
            "diagonal_floor_count": int((covariance.diagonal() < eigen_floor).sum().item())
        }
    if covariance_view != "full":
        raise ValueError(f"unknown covariance view: {covariance_view}")
    _, factor, factor_info = factorize_full_covariance(
        covariance, eigen_floor, initial_jitter, jitter_multiplier, max_tries
    )
    solved = torch.linalg.solve_triangular(factor, difference.T, upper=False)
    return solved.square().sum(dim=0), factor_info


def build_regions_for_class(
    features: torch.Tensor,
    class_id: int,
    birth_task: int,
    feature_space: str,
    num_prototypes: int,
    seed: int,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    x = _as_cpu_float(features)
    _, assignments, kmeans = deterministic_kmeans(
        x,
        requested_k=num_prototypes,
        seed=seed,
        n_init=int(config["kmeans_n_init"]),
        max_iter=int(config["kmeans_max_iter"]),
        min_cluster_size=int(config["cluster_min_samples"]),
    )
    actual_k = int(assignments.max().item()) + 1
    regions: List[Dict[str, Any]] = []
    total = int(x.shape[0])
    for component_id in range(actual_k):
        component_features = x[assignments == component_id]
        mean = component_features.mean(dim=0)
        covariance, rank, fallback = empirical_covariance(
            component_features, float(config["stats_cov_jitter"])
        )
        tau_by_covariance: Dict[str, float] = {}
        tau_details: Dict[str, Any] = {}
        for covariance_view in COVARIANCE_VIEWS:
            distances, details = mahalanobis_squared(
                component_features,
                mean,
                covariance,
                covariance_view,
                float(config["sep_cov_eigen_floor"]),
                float(config["sep_cholesky_initial_jitter"]),
                float(config["sep_cholesky_jitter_multiplier"]),
                int(config["sep_cholesky_max_tries"]),
            )
            tau = torch.quantile(distances, float(config["sep_percentile"]) / 100.0)
            tau_by_covariance[covariance_view] = float(tau.item())
            tau_details[covariance_view] = details
        count = int(component_features.shape[0])
        regions.append(
            {
                "class_id": int(class_id),
                "component_id": component_id,
                "mean": mean.clone(),
                "covariance": covariance.clone(),
                "count": count,
                "weight": count / total,
                "tau_by_covariance": tau_by_covariance,
                "birth_task": int(birth_task),
                "feature_space": str(feature_space),
                "rank": rank,
                "regularization": {"estimation_jitter": float(config["stats_cov_jitter"])},
                "fallback": fallback,
                "tau_details": tau_details,
            }
        )
    if not math.isclose(sum(region["weight"] for region in regions), 1.0, rel_tol=0.0, abs_tol=1e-7):
        raise AssertionError("cluster weights do not sum to one")
    return regions, kmeans


def build_single_region(
    features: torch.Tensor,
    class_id: int,
    birth_task: int,
    feature_space: str,
    mean: torch.Tensor,
    covariance: torch.Tensor,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    x = _as_cpu_float(features)
    mean = mean.detach().cpu().float()
    covariance = covariance.detach().cpu().float()
    tau_by_covariance: Dict[str, float] = {}
    tau_details: Dict[str, Any] = {}
    for covariance_view in COVARIANCE_VIEWS:
        distances, details = mahalanobis_squared(
            x,
            mean,
            covariance,
            covariance_view,
            float(config["sep_cov_eigen_floor"]),
            float(config["sep_cholesky_initial_jitter"]),
            float(config["sep_cholesky_jitter_multiplier"]),
            int(config["sep_cholesky_max_tries"]),
        )
        tau_by_covariance[covariance_view] = float(
            torch.quantile(distances, float(config["sep_percentile"]) / 100.0).item()
        )
        tau_details[covariance_view] = details
    centered = x - x.mean(dim=0, keepdim=True)
    return {
        "class_id": int(class_id),
        "component_id": 0,
        "mean": mean,
        "covariance": covariance,
        "count": int(x.shape[0]),
        "weight": 1.0,
        "tau_by_covariance": tau_by_covariance,
        "birth_task": int(birth_task),
        "feature_space": str(feature_space),
        "rank": int(torch.linalg.matrix_rank(centered).item()) if x.shape[0] > 1 else 0,
        "regularization": {"estimation_jitter": float(config["stats_cov_jitter"])},
        "fallback": "none",
        "tau_details": tau_details,
    }


def clone_regions(regions_by_class: Mapping[int, Sequence[Mapping[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
    cloned: Dict[int, List[Dict[str, Any]]] = {}
    for class_id, regions in regions_by_class.items():
        cloned[int(class_id)] = []
        for region in regions:
            item = dict(region)
            item["mean"] = region["mean"].detach().cpu().clone()
            item["covariance"] = region["covariance"].detach().cpu().clone()
            item["tau_by_covariance"] = dict(region["tau_by_covariance"])
            item["regularization"] = dict(region.get("regularization", {}))
            item["tau_details"] = dict(region.get("tau_details", {}))
            cloned[int(class_id)].append(item)
    return cloned


def rectify_local_regions(
    source_regions: Mapping[int, Sequence[Mapping[str, Any]]],
    mean_displacements: torch.Tensor,
    covariance_additions: torch.Tensor,
    feature_space: str,
    apr_epsilon: float,
) -> Dict[int, List[Dict[str, Any]]]:
    """Apply one class-level APR transform to every local region from source."""
    result = clone_regions(source_regions)
    identity = None
    for class_id, regions in result.items():
        if class_id >= mean_displacements.shape[0]:
            raise ValueError(f"missing APR correction for class {class_id}")
        delta = mean_displacements[class_id].detach().cpu().float()
        addition = covariance_additions[class_id].detach().cpu().float()
        if identity is None:
            identity = torch.eye(delta.shape[0], dtype=torch.float32)
        for region in regions:
            region["mean"] = region["mean"] + delta
            region["covariance"] = 0.5 * (
                region["covariance"] + addition + float(apr_epsilon) * identity
                + (region["covariance"] + addition + float(apr_epsilon) * identity).T
            )
            region["feature_space"] = str(feature_space)
            region["regularization"]["apr_epsilon"] = float(apr_epsilon)
    return result


def largest_remainder_counts(weights: Sequence[float], total: int) -> List[int]:
    if total < 0 or len(weights) == 0:
        raise ValueError("invalid allocation request")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
    if (weight_tensor < 0).any() or float(weight_tensor.sum()) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    exact = weight_tensor / weight_tensor.sum() * int(total)
    base = torch.floor(exact).to(torch.long)
    remaining = int(total - int(base.sum().item()))
    if remaining:
        fractional = exact - base
        order = sorted(range(len(weights)), key=lambda idx: (-float(fractional[idx]), idx))
        for idx in order[:remaining]:
            base[idx] += 1
    allocation = base.tolist()
    if sum(allocation) != total:
        raise AssertionError("largest-remainder allocation did not preserve the total")
    return allocation


def sample_gaussian_mixture(
    regions: Sequence[Mapping[str, Any]],
    total: int,
    device: torch.device,
    mean_scale: float,
    config: Mapping[str, Any],
    return_details: bool = False,
) -> Any:
    allocations = largest_remainder_counts([float(region["weight"]) for region in regions], total)
    samples: List[torch.Tensor] = []
    factorization_details: List[Dict[str, Any]] = []
    for region, count in zip(regions, allocations):
        if count == 0:
            continue
        _, factor, details = factorize_full_covariance(
            region["covariance"],
            float(config["sep_cov_eigen_floor"]),
            float(config["sep_cholesky_initial_jitter"]),
            float(config["sep_cholesky_jitter_multiplier"]),
            int(config["sep_cholesky_max_tries"]),
        )
        factorization_details.append(details)
        mean = region["mean"].to(device=device, dtype=torch.float32) * float(mean_scale)
        factor = factor.to(device=device, dtype=torch.float32)
        noise = torch.randn((count, mean.shape[0]), device=device, dtype=torch.float32)
        samples.append(mean.unsqueeze(0) + noise @ factor.T)
    result = torch.cat(samples, dim=0)
    if result.shape[0] != total:
        raise AssertionError("mixture replay returned the wrong sample count")
    return (result, factorization_details) if return_details else result


def region_storage_bytes(regions_by_class: Mapping[int, Sequence[Mapping[str, Any]]]) -> int:
    total = 0
    for regions in regions_by_class.values():
        for region in regions:
            total += region["mean"].numel() * region["mean"].element_size()
            total += region["covariance"].numel() * region["covariance"].element_size()
            total += 8 * len(region["tau_by_covariance"])
    return int(total)


@dataclass
class SeparationDiagnostics:
    nearest_d2: torch.Tensor
    nearest_rho: torch.Tensor
    intrusion: torch.Tensor


class SeparationCache:
    """Detached region parameters and pre-factorized covariance views."""

    def __init__(
        self,
        regions: Sequence[Mapping[str, Any]],
        covariance_view: str,
        feature_space: str,
        config: Mapping[str, Any],
    ) -> None:
        if covariance_view not in COVARIANCE_VIEWS:
            raise ValueError(f"unknown covariance view: {covariance_view}")
        if not regions:
            raise ValueError("a separation cache needs at least one region")
        mismatches = [region["feature_space"] for region in regions if region["feature_space"] != feature_space]
        if mismatches:
            raise ValueError("region feature-space identifier mismatch")
        self.covariance_view = covariance_view
        self.feature_space = feature_space
        self.chunk_size = int(config["sep_region_chunk_size"])
        self.tau_floor = float(config["sep_tau_floor"])
        self.means = torch.stack([region["mean"].detach().cpu().float() for region in regions])
        self.taus = torch.tensor(
            [max(float(region["tau_by_covariance"][covariance_view]), self.tau_floor) for region in regions],
            dtype=torch.float32,
        )
        self.class_ids = torch.tensor([int(region["class_id"]) for region in regions], dtype=torch.long)
        self.factorization_details: List[Dict[str, Any]] = []
        if covariance_view == "isotropic":
            scales = []
            for region in regions:
                raw_scale = float(
                    torch.trace(region["covariance"]).item() / region["covariance"].shape[0]
                )
                floor_applied = raw_scale < float(config["sep_cov_eigen_floor"])
                scales.append(max(raw_scale, float(config["sep_cov_eigen_floor"])))
                self.factorization_details.append(
                    {
                        "factorization": "isotropic_elementwise",
                        "scale_floor_applied": floor_applied,
                        "fallback_applied": floor_applied,
                    }
                )
            self.factors = torch.tensor(scales, dtype=torch.float32)
        elif covariance_view == "diagonal":
            diagonals = []
            for region in regions:
                diagonal = region["covariance"].detach().cpu().float().diagonal()
                floor_count = int((diagonal < float(config["sep_cov_eigen_floor"])).sum().item())
                diagonals.append(diagonal.clamp_min(float(config["sep_cov_eigen_floor"])))
                self.factorization_details.append(
                    {
                        "factorization": "diagonal_elementwise",
                        "diagonal_floor_count": floor_count,
                        "fallback_applied": floor_count > 0,
                    }
                )
            self.factors = torch.stack(diagonals)
        else:
            factors = []
            for region in regions:
                _, factor, details = factorize_full_covariance(
                    region["covariance"],
                    float(config["sep_cov_eigen_floor"]),
                    float(config["sep_cholesky_initial_jitter"]),
                    float(config["sep_cholesky_jitter_multiplier"]),
                    int(config["sep_cholesky_max_tries"]),
                )
                factors.append(factor)
                self.factorization_details.append(details)
            self.factors = torch.stack(factors)
        if str(config["sep_factor_device"]) == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("sep_factor_device=cuda but CUDA is unavailable")
            self.means = self.means.cuda()
            self.taus = self.taus.cuda()
            self.factors = self.factors.cuda()

    @property
    def region_count(self) -> int:
        return int(self.means.shape[0])

    @property
    def storage_bytes(self) -> int:
        tensors = (self.means, self.taus, self.class_ids, self.factors)
        return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))

    def loss(self, features: torch.Tensor) -> Tuple[torch.Tensor, SeparationDiagnostics]:
        if features.ndim != 2 or features.shape[1] != self.means.shape[1]:
            raise ValueError("student feature shape does not match the separation cache")
        nearest_rho = torch.full((features.shape[0],), float("inf"), device=features.device)
        nearest_d2 = torch.full((features.shape[0],), float("inf"), device=features.device)
        for start in range(0, self.region_count, self.chunk_size):
            stop = min(start + self.chunk_size, self.region_count)
            means = self.means[start:stop].to(features.device, non_blocking=True)
            taus = self.taus[start:stop].to(features.device, non_blocking=True)
            difference = features[:, None, :] - means[None, :, :]
            factors = self.factors[start:stop].to(features.device, non_blocking=True)
            if self.covariance_view == "isotropic":
                distances = difference.square().sum(dim=2) / factors.unsqueeze(0)
            elif self.covariance_view == "diagonal":
                distances = (difference.square() / factors.unsqueeze(0)).sum(dim=2)
            else:
                solved = torch.linalg.solve_triangular(
                    factors, difference.permute(1, 2, 0).contiguous(), upper=False
                )
                distances = solved.square().sum(dim=1).T
            rho = distances / taus.unsqueeze(0)
            nearest_rho = torch.minimum(nearest_rho, rho.min(dim=1).values)
            nearest_d2 = torch.minimum(nearest_d2, distances.min(dim=1).values)
        per_sample = F.relu(1.0 - nearest_rho)
        diagnostics = SeparationDiagnostics(
            nearest_d2=nearest_d2.detach(),
            nearest_rho=nearest_rho.detach(),
            intrusion=(nearest_rho < 1.0).detach(),
        )
        return per_sample.mean(), diagnostics


def flatten_regions(regions_by_class: Mapping[int, Sequence[Mapping[str, Any]]]) -> List[Mapping[str, Any]]:
    return [region for class_id in sorted(regions_by_class) for region in regions_by_class[class_id]]


def summarize_regions(regions_by_class: Mapping[int, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for class_id in sorted(regions_by_class):
        for region in regions_by_class[class_id]:
            rows.append(
                {
                    "class_id": int(class_id),
                    "component_id": int(region["component_id"]),
                    "count": int(region["count"]),
                    "weight": float(region["weight"]),
                    "birth_task": int(region["birth_task"]),
                    "feature_space": region["feature_space"],
                    "rank": int(region["rank"]),
                    "covariance_shape": list(region["covariance"].shape),
                    "fallback": region["fallback"],
                    "regularization": region["regularization"],
                    "tau_by_covariance": region["tau_by_covariance"],
                }
            )
    return rows


def compare_mixture_to_class_statistics(
    regions_by_class: Mapping[int, Sequence[Mapping[str, Any]]],
    class_means: Any,
    class_covariances: torch.Tensor,
) -> List[Dict[str, float]]:
    """Report, but do not force away, mixture/class moment discrepancies."""
    rows = []
    for class_id in sorted(regions_by_class):
        regions = regions_by_class[class_id]
        weights = torch.tensor([float(region["weight"]) for region in regions], dtype=torch.float32)
        weights = weights / weights.sum()
        means = torch.stack([region["mean"].detach().cpu().float() for region in regions])
        mixture_mean = torch.sum(weights[:, None] * means, dim=0)
        mixture_covariance = torch.zeros_like(regions[0]["covariance"], dtype=torch.float32)
        for weight, mean, region in zip(weights, means, regions):
            offset = mean - mixture_mean
            mixture_covariance += weight * (
                region["covariance"].detach().cpu().float() + torch.outer(offset, offset)
            )
        target_mean = torch.as_tensor(class_means[class_id]).detach().cpu().float()
        target_covariance = class_covariances[class_id].detach().cpu().float()
        mean_difference = torch.linalg.vector_norm(mixture_mean - target_mean)
        covariance_difference = torch.linalg.matrix_norm(mixture_covariance - target_covariance)
        rows.append(
            {
                "class_id": int(class_id),
                "mean_l2_difference": float(mean_difference),
                "covariance_frobenius_difference": float(covariance_difference),
                "covariance_relative_frobenius_difference": float(
                    covariance_difference / target_covariance.norm().clamp_min(1e-12)
                ),
            }
        )
    return rows
