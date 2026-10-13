from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class PCA2:
    def fit(self, values: np.ndarray):
        values = np.asarray(values, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        centered = values - self.mean_
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        self.components_ = eigenvectors[:, np.argsort(eigenvalues)[-2:][::-1]]
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean_) @ self.components_


def _pca_reduce(values: np.ndarray, dimensions: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - values.mean(axis=0, keepdims=True)
    if values.shape[1] <= dimensions:
        return values
    covariance = values.T @ values
    _, eigenvectors = np.linalg.eigh(covariance)
    return values @ eigenvectors[:, -dimensions:]


def _joint_probabilities(values: torch.Tensor, perplexity: float) -> torch.Tensor:
    distances = torch.cdist(values, values).square()
    count = values.shape[0]
    diagonal = torch.eye(count, dtype=torch.bool, device=values.device)
    target_entropy = float(np.log(perplexity))
    beta = torch.ones((count, 1), device=values.device)
    beta_min = torch.full_like(beta, -float("inf"))
    beta_max = torch.full_like(beta, float("inf"))
    for _ in range(45):
        weights = torch.exp(-distances * beta).masked_fill(diagonal, 0.0)
        sums = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        probabilities = weights / sums
        entropy = torch.log(sums[:, 0]) + beta[:, 0] * (probabilities * distances).sum(dim=1)
        high = entropy > target_entropy
        beta_min[high] = beta[high]
        beta_max[~high] = beta[~high]
        beta = torch.where(
            high[:, None],
            torch.where(torch.isinf(beta_max), beta * 2.0, (beta + beta_max) / 2.0),
            torch.where(torch.isinf(beta_min), beta / 2.0, (beta + beta_min) / 2.0),
        )
    probabilities = (probabilities + probabilities.T) / (2.0 * count)
    return probabilities.clamp_min(1e-12)


def torch_tsne(
    values: np.ndarray,
    perplexity: float,
    seed: int,
    iterations: int = 750,
) -> np.ndarray:
    """Exact deterministic t-SNE for a bounded visualization subset.

    The caller caps sample points; prototypes are always retained.  This keeps
    the O(N^2) diagnostic independent of scikit-learn availability.
    """
    reduced = _pca_reduce(values, dimensions=50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.as_tensor(reduced, dtype=torch.float32, device=device)
    x = (x - x.mean(dim=0)) / x.std(dim=0).clamp_min(1e-6)
    probabilities = _joint_probabilities(x, perplexity).detach()
    generator = torch.Generator(device=device).manual_seed(seed)
    coordinates = (torch.randn((x.shape[0], 2), generator=generator, device=device) * 1e-4).requires_grad_(True)
    optimizer = torch.optim.Adam([coordinates], lr=200.0)
    print(f"[visualize] exact t-SNE points={x.shape[0]} device={device} iterations={iterations}", flush=True)
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        distances = torch.cdist(coordinates, coordinates).square()
        numerator = 1.0 / (1.0 + distances)
        numerator.fill_diagonal_(0.0)
        q = (numerator / numerator.sum().clamp_min(1e-12)).clamp_min(1e-12)
        p = probabilities * (12.0 if iteration < 250 else 1.0)
        loss = torch.sum(p * (torch.log(probabilities) - torch.log(q)))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            coordinates -= coordinates.mean(dim=0, keepdim=True)
        if (iteration + 1) % 250 == 0:
            print(f"[visualize] t-SNE iteration {iteration + 1}/{iterations}", flush=True)
    result = coordinates.detach().cpu().numpy()
    del x, probabilities, coordinates, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _stratified_cap(
    features: np.ndarray,
    labels: np.ndarray,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if features.shape[0] <= cap:
        return features, labels
    rng = np.random.RandomState(seed)
    classes = np.unique(labels)
    quota = max(1, cap // classes.size)
    selected = []
    for class_id in classes:
        indices = np.flatnonzero(labels == class_id)
        selected.extend(rng.choice(indices, min(quota, indices.size), replace=False).tolist())
    if len(selected) < cap:
        remaining = np.setdiff1d(np.arange(labels.size), np.asarray(selected), assume_unique=False)
        selected.extend(rng.choice(remaining, min(cap - len(selected), remaining.size), replace=False).tolist())
    selected = np.asarray(sorted(selected[:cap]), dtype=int)
    return features[selected], labels[selected]


def _rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _task_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / "features").glob("task_*.npz"))


def _task_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _focused_classes(rows: list[dict], task: int, available: np.ndarray) -> list[int]:
    candidates = [
        row for row in rows
        if row.get("oracle_split") == "train" and int(row["task"]) == task and row.get("is_old") == "True"
    ]
    if not candidates:
        return list(map(int, available[: min(12, available.size)]))
    selected: list[int] = []
    selectors = (
        ("prototype_relative_l2", True, 4),
        ("prototype_relative_l2", False, 3),
        ("forgetting", True, 3),
        ("old_to_new_confusion", True, 3),
    )
    for key, reverse, count in selectors:
        valid = []
        for row in candidates:
            try:
                valid.append((float(row[key]), int(row["class_id"])))
            except (KeyError, TypeError, ValueError):
                continue
        for _, class_id in sorted(valid, reverse=reverse)[:count]:
            if class_id not in selected:
                selected.append(class_id)
    return selected[:15]


def _plot_embedding(
    coordinates: np.ndarray,
    labels: np.ndarray,
    birth_tasks: np.ndarray,
    class_ids: np.ndarray,
    oracle_xy: np.ndarray,
    structure_xy: np.ndarray,
    preapr_xy: np.ndarray | None,
    preapr_available: np.ndarray | None,
    destination: Path,
    title: str,
    focus: list[int] | None = None,
) -> None:
    if focus is not None:
        sample_keep = np.isin(labels, focus)
        prototype_keep = np.isin(class_ids, focus)
    else:
        sample_keep = np.ones(labels.size, dtype=bool)
        prototype_keep = np.ones(class_ids.size, dtype=bool)
    birth_lookup = {int(class_id): int(birth) for class_id, birth in zip(class_ids, birth_tasks)}
    sample_birth = np.asarray([birth_lookup[int(label)] for label in labels])
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    scatter = axis.scatter(
        coordinates[sample_keep, 0], coordinates[sample_keep, 1],
        c=sample_birth[sample_keep], cmap="tab10", s=7, alpha=0.35, linewidths=0,
    )
    for index, class_id in enumerate(class_ids):
        if not prototype_keep[index]:
            continue
        has_preapr = (
            preapr_xy is not None
            and preapr_available is not None
            and bool(preapr_available[index])
            and np.all(np.isfinite(preapr_xy[index]))
        )
        if has_preapr:
            axis.plot(
                [preapr_xy[index, 0], structure_xy[index, 0]],
                [preapr_xy[index, 1], structure_xy[index, 1]],
                color="tab:green", linewidth=0.8, alpha=0.8,
            )
        axis.plot(
            [structure_xy[index, 0], oracle_xy[index, 0]],
            [structure_xy[index, 1], oracle_xy[index, 1]],
            color="0.45", linewidth=0.45, alpha=0.6,
        )
    if preapr_xy is not None and preapr_available is not None:
        preapr_keep = prototype_keep & preapr_available.astype(bool)
        axis.scatter(
            preapr_xy[preapr_keep, 0], preapr_xy[preapr_keep, 1],
            marker="^", s=42, c="tab:green", label="pre-APR", zorder=4,
        )
    axis.scatter(
        oracle_xy[prototype_keep, 0], oracle_xy[prototype_keep, 1],
        marker="*", s=65, c="black", label="oracle mean", zorder=4,
    )
    axis.scatter(
        structure_xy[prototype_keep, 0], structure_xy[prototype_keep, 1],
        marker="x", s=38, c="red", label="Structure/APR", zorder=5,
    )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(loc="best", frameon=True)
    figure.colorbar(scatter, ax=axis, label="birth task")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def build_visualizations(output_dir: Path, partition: list[int], seed: int) -> None:
    files = _task_files(output_dir)
    if not files:
        return
    prototype_rows = _rows(output_dir / "prototype_oracle_audit.csv")
    loaded = [(path, np.load(path)) for path in files]

    for path, data in loaded:
        task = _task_id(path)
        features, labels = _stratified_cap(data["features"], data["labels"], 1200, seed + task)
        class_ids = data["class_ids"]
        oracle = data["oracle_means"]
        structure = data["structure_means"]
        preapr = data["preapr_means"] if "preapr_means" in data.files else None
        preapr_available = (
            data["preapr_available"].astype(bool)
            if "preapr_available" in data.files
            else np.zeros(class_ids.size, dtype=bool)
        )
        preapr_selected = preapr[preapr_available] if preapr is not None else np.empty((0, structure.shape[1]))
        combined = np.concatenate([features, oracle, structure, preapr_selected], axis=0)
        if combined.shape[0] < 4:
            continue
        perplexity = min(30.0, max(2.0, (combined.shape[0] - 1) / 3.0))
        embedding = torch_tsne(combined, perplexity=perplexity, seed=seed + task)
        sample_count = features.shape[0]
        class_count = class_ids.size
        sample_xy = embedding[:sample_count]
        oracle_xy = embedding[sample_count : sample_count + class_count]
        structure_xy = embedding[sample_count + class_count :]
        structure_xy = structure_xy[:class_count]
        preapr_xy = np.full((class_count, 2), np.nan, dtype=float)
        if preapr_selected.shape[0]:
            preapr_xy[preapr_available] = embedding[sample_count + 2 * class_count :]
        _plot_embedding(
            sample_xy, labels, data["birth_tasks"], class_ids, oracle_xy, structure_xy,
            preapr_xy, preapr_available,
            output_dir / "tsne" / f"task_{task:02d}_tsne.png",
            f"{output_dir.name} T{task}: task-local t-SNE (coordinates not comparable across tasks)",
        )
        focus = _focused_classes(prototype_rows, task, class_ids)
        _plot_embedding(
            sample_xy, labels, data["birth_tasks"], class_ids, oracle_xy, structure_xy,
            preapr_xy, preapr_available,
            output_dir / "tsne" / f"task_{task:02d}_focused_tsne.png",
            f"{output_dir.name} T{task}: focused task-local t-SNE",
            focus=focus,
        )

    reference = np.concatenate([data["features"] for _, data in loaded], axis=0)
    pca = PCA2().fit(reference)
    trajectory: dict[int, dict[str, list]] = {}
    for path, data in loaded:
        task = _task_id(path)
        class_ids = data["class_ids"]
        feature_xy = pca.transform(data["features"])
        oracle_xy = pca.transform(data["oracle_means"])
        structure_xy = pca.transform(data["structure_means"])
        preapr_available = (
            data["preapr_available"].astype(bool)
            if "preapr_available" in data.files
            else np.zeros(class_ids.size, dtype=bool)
        )
        preapr_xy = (
            pca.transform(data["preapr_means"])
            if "preapr_means" in data.files
            else np.full_like(structure_xy, np.nan)
        )
        _plot_embedding(
            feature_xy, data["labels"], data["birth_tasks"], class_ids, oracle_xy, structure_xy,
            preapr_xy, preapr_available,
            output_dir / "pca" / f"task_{task:02d}_shared_pca.png",
            f"{output_dir.name} T{task}: shared PCA basis",
        )
        apr_rows = [
            row for row in _rows(output_dir / "apr_ablation_audit.csv")
            if int(row["task"]) == task
        ]
        for result, reverse in (("improved", True), ("worsened", False)):
            candidates = [row for row in apr_rows if row.get("APR_result") == result]
            candidates.sort(
                key=lambda row: float(row.get("APR_error_reduction") or 0.0),
                reverse=reverse,
            )
            focus = [int(row["class_id"]) for row in candidates[:8]]
            if focus:
                _plot_embedding(
                    feature_xy,
                    data["labels"],
                    data["birth_tasks"],
                    class_ids,
                    oracle_xy,
                    structure_xy,
                    preapr_xy,
                    preapr_available,
                    output_dir / "pca" / f"task_{task:02d}_APR_{result}_shared_pca.png",
                    f"{output_dir.name} T{task}: APR {result} classes (shared PCA)",
                    focus=focus,
                )
        for index, class_id in enumerate(class_ids):
            entry = trajectory.setdefault(int(class_id), {"task": [], "oracle": [], "structure": []})
            entry["task"].append(task)
            entry["oracle"].append(oracle_xy[index])
            entry["structure"].append(structure_xy[index])

    final_task = max(_task_id(path) for path, _ in loaded)
    candidates = [
        row for row in prototype_rows
        if row.get("oracle_split") == "train" and int(row["task"]) == final_task
    ]
    candidates.sort(key=lambda row: float(row.get("prototype_relative_l2") or -1), reverse=True)
    selected = [int(row["class_id"]) for row in candidates if len(trajectory[int(row["class_id"])]["task"]) > 1][:8]
    if not selected:
        selected = [class_id for class_id, value in trajectory.items() if len(value["task"]) > 1][:8]
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for index, class_id in enumerate(selected):
        values = trajectory[class_id]
        oracle_xy = np.asarray(values["oracle"])
        structure_xy = np.asarray(values["structure"])
        color = colors(index % 10)
        axis.plot(oracle_xy[:, 0], oracle_xy[:, 1], "-o", color=color, label=f"c{class_id} oracle")
        axis.plot(structure_xy[:, 0], structure_xy[:, 1], "--x", color=color, label=f"c{class_id} Structure")
        for task, point in zip(values["task"], oracle_xy):
            axis.annotate(f"T{task}", point, fontsize=7, color=color)
    axis.set_title(f"{output_dir.name}: oracle vs Structure prototype trajectories (shared PCA)")
    if selected:
        axis.legend(ncol=2, fontsize=7)
    else:
        axis.text(0.5, 0.5, "At least two task checkpoints are required", ha="center", va="center", transform=axis.transAxes)
    axis.set_xlabel("shared PCA 1")
    axis.set_ylabel("shared PCA 2")
    destination = output_dir / "prototype_trajectory" / "prototype_trajectory_shared_pca.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    for _, data in loaded:
        data.close()
