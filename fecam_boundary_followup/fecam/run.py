from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, fixed_scores, inputs, metrics, write_csv, write_json


def _moments(values: np.ndarray, jitter: float) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(values, dtype=np.float64)
    mean = value.mean(axis=0)
    centered = value - mean
    covariance = centered.T @ centered / max(len(value) - 1, 1)
    covariance.flat[:: covariance.shape[0] + 1] += jitter
    return mean, covariance


def _fecam_covariance(covariance: torch.Tensor) -> torch.Tensor:
    """FeCAM/models/base.py shrink_cov + normalize_cov; alpha1=alpha2=10 in ViT script."""
    covariance = 0.5 * (covariance + covariance.T)
    diagonal = torch.diagonal(covariance)
    off = covariance.clone()
    off.fill_diagonal_(0)
    nonzero = off != 0
    off_mean = off[nonzero].mean() if bool(nonzero.any()) else off.new_zeros(())
    dimension = covariance.shape[0]
    identity = torch.eye(dimension, device=covariance.device, dtype=covariance.dtype)
    shrunk = covariance + 10.0 * diagonal.mean() * identity + 10.0 * off_mean * (1 - identity)
    sd = torch.sqrt(torch.diagonal(shrunk))
    if not bool(torch.isfinite(sd).all()) or bool((sd <= 0).any()):
        raise FloatingPointError("FeCAM shrinkage produced invalid standard deviation")
    return shrunk / (sd[:, None] * sd[None, :])


def _mahalanobis_scores(
    test_features: np.ndarray,
    means: np.ndarray,
    covariances: list[np.ndarray | torch.Tensor],
    device: torch.device,
    tag: str,
) -> tuple[np.ndarray, list[dict]]:
    query = F.normalize(torch.as_tensor(test_features, dtype=torch.float32, device=device), dim=1)
    scores = np.empty((len(test_features), len(means)), dtype=np.float32)
    statuses = []
    for class_id, (mean, covariance) in enumerate(zip(means, covariances)):
        cov = torch.as_tensor(covariance, dtype=torch.float32, device=device)
        reg = _fecam_covariance(cov)
        # FeCAM calls pinv; for a full-rank positive matrix, Cholesky solve is algebraically identical.
        factor, info = torch.linalg.cholesky_ex(reg)
        if int(info.item()) == 0:
            method = "cholesky_solve"
        else:
            method = "pinv_fallback"
            inverse = torch.linalg.pinv(reg, hermitian=True)
        mu = F.normalize(torch.as_tensor(mean, dtype=torch.float32, device=device), dim=0)
        with torch.no_grad():
            for start in range(0, len(query), 1024):
                delta = query[start : start + 1024] - mu
                if method == "cholesky_solve":
                    solved = torch.cholesky_solve(delta.T, factor).T
                else:
                    solved = delta @ inverse
                scores[start : start + len(delta), class_id] = (-(delta * solved).sum(dim=1)).cpu().numpy()
        statuses.append({"condition": tag, "class_id": class_id, "inverse": method,
                         "pre_shrink_diag_mean": float(torch.diagonal(cov).mean().item()),
                         "post_shrink_min_diag": float(torch.diagonal(reg).min().item())})
        if (class_id + 1) % 20 == 0 or class_id + 1 == len(means):
            print(f"[{tag}] covariance {class_id + 1}/{len(means)}", flush=True)
    return scores, statuses


def run(protocol: str, device: torch.device, seed: int = 1993) -> dict:
    _, checkpoint, cache, known = inputs(protocol, seed)
    out = ROOT / "fecam" / protocol
    out.mkdir(parents=True, exist_ok=True)
    train, train_labels = cache["train_features"], cache["train_labels"]
    test, labels = cache["test_features"], cache["test_labels"]
    classes = int(checkpoint["total_classes"])
    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle = [_moments(train[train_labels == c], jitter) for c in range(classes)]
    oracle_means = np.stack([v[0] for v in oracle])
    stored_means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float32)
    # Raw Structure moments are used for both Stored and Oracle; no unavailable
    # per-sample L2-normalized moments are inferred from stored raw moments.
    stored_means[known:] = oracle_means[known:]
    stored_covs = [checkpoint["class_covs"][c] if c < known else oracle[c][1] for c in range(classes)]
    oracle_covs = [v[1] for v in oracle]
    f0 = fixed_scores(test, checkpoint)
    query = F.normalize(torch.as_tensor(test, dtype=torch.float32, device=device), dim=1)
    # Official NCM_vit_imagenet-r.py averages per-sample normalized features;
    # it does not normalize the resulting class mean again.
    normalized_train = F.normalize(torch.as_tensor(train, dtype=torch.float32, device=device), dim=1)
    train_targets = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    prototypes = torch.stack([normalized_train[train_targets == c].mean(dim=0) for c in range(classes)])
    f1 = (-torch.cdist(query, prototypes).square()).cpu().numpy()
    f2, stat2 = _mahalanobis_scores(test, stored_means, stored_covs, device, "F2")
    f3, stat3 = _mahalanobis_scores(test, oracle_means, oracle_covs, device, "F3")
    data = {}
    for name, scores in (("F0", f0), ("F1", f1), ("F2", f2), ("F3", f3)):
        data[name] = metrics(scores, labels, classes, known)
        print(f"[{protocol} {name}] seen={data[name]['seen_class_accuracy']:.2f} old={data[name]['old_accuracy']:.2f} new={data[name]['new_accuracy']:.2f}", flush=True)
    counts = np.bincount(train_labels, minlength=classes)
    rows = []
    for status in stat2 + stat3:
        class_id = status["class_id"]
        status["train_count"] = int(counts[class_id])
        status["empirical_cov_rank_upper_bound"] = int(min(counts[class_id] - 1, train.shape[1]))
        rows.append(status)
    write_json(out / "metrics.json", data)
    write_csv(out / "covariance_status.csv", rows)
    return data
