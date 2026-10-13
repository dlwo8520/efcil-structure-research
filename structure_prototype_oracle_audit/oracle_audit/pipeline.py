from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from oracle_audit.metrics import (
    EPS,
    classify_drift,
    cosine_nearest,
    covariance_metrics,
    distribution_summary,
    mahalanobis_squared_diagonal,
    mahalanobis_squared_full,
    pearson,
    safe_cosine,
    spearman,
    spectrum_metrics,
    vector_error,
)
from oracle_audit.diagnostics import (
    boundary_direction_rows,
    competition_row,
    gaussian_rows,
    margin_rows,
)
from oracle_audit.oracle_head import (
    CONDITIONS,
    checkpoint_classifier_weights,
    cosine_scores,
    evaluate_scores,
    load_classifier_initialization,
    train_diagnostic_head,
)
from oracle_audit.runtime import (
    birth_task,
    build_data_manager,
    class_statistics,
    discover_checkpoints,
    extract_split,
    load_frozen_learner,
    release_learner,
    torch_load,
)


@dataclass
class AuditOptions:
    protocol: str
    run_dir: Path
    output_dir: Path
    data_root: Path
    device: torch.device
    explicit_checkpoints: list[str]
    splits: tuple[str, ...] = ("train", "test")
    batch_size: int = 64
    num_workers: int = 4
    covariance_mode: str = "both"
    sample_cap: int | None = None
    class_cap: int | None = None
    task_filter: set[int] | None = None
    visualization_samples_per_class: int = 50
    seed: int = 1993
    skip_visualizations: bool = False
    oracle_head_eval_tasks: str = "final"
    oracle_head_epochs: int | None = None
    oracle_head_samples_per_class: int | None = None


def _finite(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _finite(row.get(key)) for key in fields})
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_finite) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(stat.st_size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "edge_sha256": digest.hexdigest()}


def _sample_for_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in class_ids:
        indices = np.flatnonzero(labels == int(class_id))
        if indices.size > per_class:
            indices = np.sort(rng.choice(indices, per_class, replace=False))
        selected.extend(indices.tolist())
    selected = np.asarray(sorted(selected), dtype=np.int64)
    return features[selected].astype(np.float32, copy=False), labels[selected]


def _nearest_old_diagonal(
    features: np.ndarray,
    means: np.ndarray,
    covariances: list[np.ndarray],
    chunk: int = 128,
) -> np.ndarray:
    predictions = []
    means = np.asarray(means, dtype=np.float64)
    variances = np.stack([np.maximum(np.diag(value), 1e-6) for value in covariances])
    for start in range(0, features.shape[0], chunk):
        batch = np.asarray(features[start : start + chunk], dtype=np.float64)
        distance = np.sum(np.square(batch[:, None, :] - means[None, :, :]) / variances[None, :, :], axis=2)
        predictions.append(np.argmin(distance, axis=1))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.int64)


def _task_seen(partition: list[int], task: int) -> int:
    return int(sum(partition[:task]))


def _correlation_rows(
    protocol: str,
    task: int | str,
    class_records: list[dict],
) -> list[dict]:
    rows = []
    predictors = (
        "prototype_relative_l2",
        "absolute_boundary_direction_error",
        "angle_error_deg",
        "drift_angle_error_deg",
        "drift_magnitude_relative_error",
        "covariance_frobenius_relative_error",
    )
    outcomes = ("forgetting", "old_to_new_confusion", "class_accuracy_drop")
    for predictor in predictors:
        for outcome in outcomes:
            x = np.asarray([row.get(predictor, np.nan) for row in class_records], dtype=float)
            y = np.asarray([row.get(outcome, np.nan) for row in class_records], dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            rows.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "predictor": predictor,
                    "outcome": outcome,
                    "n": int(keep.sum()),
                    "pearson": pearson(x, y),
                    "spearman": spearman(x, y),
                }
            )
    return rows


def audit_protocol(options: AuditOptions) -> dict:
    started = time.time()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = options.output_dir / "audit_progress.json"
    write_json(
        progress_path,
        {
            "protocol": options.protocol,
            "status": "running",
            "current_task": None,
            "completed_tasks": [],
            "started_at_unix": started,
            "updated_at_unix": started,
        },
    )
    checkpoints = discover_checkpoints(options.run_dir, options.explicit_checkpoints)
    if options.task_filter:
        selected = []
        for path in checkpoints:
            checkpoint = torch_load(path)
            task = int(checkpoint["cur_task"]) + 1
            del checkpoint
            if task in options.task_filter:
                selected.append(path)
        checkpoints = selected
    if not checkpoints:
        raise ValueError("no checkpoints remain after --task filtering")

    data_manager, protocol_config = build_data_manager(options.protocol, options.data_root, options.seed)
    partition = list(map(int, protocol_config["task_partition"]))
    expected_tasks = int(protocol_config["expected_tasks"])
    expected_order = list(map(int, data_manager.protocol_manifest["class_order_original_ids"]))
    before_fingerprints = [_file_fingerprint(path) for path in checkpoints]

    prototype_rows: list[dict] = []
    drift_rows: list[dict] = []
    covariance_rows: list[dict] = []
    mahalanobis_rows: list[dict] = []
    classifier_rows: list[dict] = []
    class_history_rows: list[dict] = []
    correlation_rows: list[dict] = []
    ranking_rows: list[dict] = []
    apr_ablation_rows: list[dict] = []
    oracle_replacement_rows: list[dict] = []
    competition_rows: list[dict] = []
    boundary_rows: list[dict] = []
    margin_diagnostic_rows: list[dict] = []
    gaussian_diagnostic_rows: list[dict] = []
    ca_regression_rows: list[dict] = []
    previous_oracle_means: dict[int, np.ndarray] = {}
    previous_oracle_task: int | None = None
    best_class_accuracy: dict[int, float] = {}
    birth_class_accuracy: dict[int, float] = {}
    all_correlation_records: list[dict] = []
    previous_structure_traces: dict[int, float] = {}
    previous_oracle_traces: dict[int, float] = {}
    audited_tasks = []

    for checkpoint_path in checkpoints:
        print(f"[audit] loading {checkpoint_path}", flush=True)
        checkpoint = torch_load(checkpoint_path)
        task = int(checkpoint["cur_task"]) + 1
        write_json(
            progress_path,
            {
                "protocol": options.protocol,
                "status": "running",
                "current_task": task,
                "completed_tasks": audited_tasks,
                "started_at_unix": started,
                "updated_at_unix": time.time(),
            },
        )
        checkpoint_seed = int(checkpoint.get("config", {}).get("seed", options.seed))
        if checkpoint_seed != options.seed:
            raise ValueError(f"seed mismatch: checkpoint={checkpoint_seed}, audit={options.seed}")
        order_path = checkpoint_path.parent / "class_order.json"
        if order_path.is_file():
            stored_order = json.loads(order_path.read_text(encoding="utf-8")).get(
                "class_order_original_ids"
            )
            if stored_order is not None and list(map(int, stored_order)) != expected_order:
                raise ValueError(f"class-order mismatch between {checkpoint_path} and {options.protocol}")
        total_classes = int(checkpoint["total_classes"])
        known_before = total_classes - partition[task - 1]
        expected_seen = _task_seen(partition, task)
        if total_classes != expected_seen:
            raise ValueError(
                f"{options.protocol} task {task} expects {expected_seen} classes, checkpoint has {total_classes}"
            )
        if list(map(int, checkpoint["task_sizes"])) != partition[:task]:
            raise ValueError(
                f"task partition mismatch in {checkpoint_path}: {checkpoint['task_sizes']} vs {partition[:task]}"
            )
        class_ids = np.arange(total_classes, dtype=np.int64)
        if options.class_cap is not None:
            cap = min(int(options.class_cap), total_classes)
            if task > 1 and cap >= 2:
                old_count = min((cap + 1) // 2, known_before)
                current_start = known_before
                new_count = min(cap - old_count, total_classes - known_before)
                class_ids = np.concatenate(
                    [np.arange(old_count), np.arange(current_start, current_start + new_count)]
                ).astype(np.int64)
            else:
                class_ids = class_ids[:cap]
        if class_ids.size == 0:
            raise ValueError("class cap selected zero classes")
        audited_tasks.append(task)
        scratch = options.output_dir / ".scratch" / f"task_{task:02d}"
        learner = load_frozen_learner(checkpoint, options.device, scratch)

        structure_means = np.asarray(checkpoint["class_means"][:total_classes], dtype=np.float64)
        structure_covariances_tensor = checkpoint["class_covs"][:total_classes]
        source_means = checkpoint.get("task_source_means")
        source_covariances = checkpoint.get("task_source_covs")
        task_train_means: dict[int, np.ndarray] = {}
        task_train_covariances: dict[int, np.ndarray] = {}
        task_primary_rows: dict[int, dict] = {}
        test_features = test_labels = test_predictions = None
        current_structure_traces: dict[int, float] = {}
        current_oracle_traces: dict[int, float] = {}
        preapr_means = structure_means[class_ids].copy()
        preapr_available = np.zeros(class_ids.size, dtype=bool)
        if task > 1 and source_means is not None:
            source_means_np_for_plot = np.asarray(source_means, dtype=np.float64)
            for local_index, class_id in enumerate(class_ids):
                if int(class_id) < known_before:
                    preapr_means[local_index] = source_means_np_for_plot[int(class_id)]
                    preapr_available[local_index] = True

        for split in options.splits:
            print(f"[audit] {options.protocol} T{task} oracle_{split}_statistics", flush=True)
            features, labels, predictions = extract_split(
                learner,
                data_manager,
                class_ids,
                source=split,
                batch_size=options.batch_size,
                num_workers=options.num_workers,
                max_samples_per_class=options.sample_cap,
                seed=options.seed + task,
            )
            oracle = class_statistics(
                features,
                labels,
                class_ids,
                covariance_jitter=float(checkpoint["config"].get("stats_cov_jitter", 1e-3)),
            )

            oracle_means_matrix = np.stack([oracle[int(class_id)].mean for class_id in class_ids])
            structure_predictions = cosine_nearest(features, structure_means[class_ids])
            oracle_predictions = cosine_nearest(features, oracle_means_matrix)
            remapped_labels = np.searchsorted(class_ids, labels)
            structure_accuracy = float(np.mean(structure_predictions == remapped_labels) * 100.0)
            oracle_accuracy = float(np.mean(oracle_predictions == remapped_labels) * 100.0)
            classifier_rows.append(
                {
                    "dataset": protocol_config["source_dataset"],
                    "protocol": options.protocol,
                    "task": task,
                    "oracle_split": split,
                    "metric": "cosine_nearest_mean",
                    "num_samples": int(labels.size),
                    "structure_prototype_accuracy": structure_accuracy,
                    "oracle_prototype_accuracy": oracle_accuracy,
                    "oracle_minus_structure_gap": oracle_accuracy - structure_accuracy,
                }
            )

            for class_id in class_ids:
                class_id = int(class_id)
                stats = oracle[class_id]
                proto_error = vector_error(structure_means[class_id], stats.mean)
                birth = birth_task(class_id, partition)
                row = {
                    "dataset": protocol_config["source_dataset"],
                    "protocol": options.protocol,
                    "task": task,
                    "class_id": class_id,
                    "birth_task": birth,
                    "class_age": task - birth,
                    "is_old": class_id < known_before,
                    "oracle_split": split,
                    "oracle_statistic_name": f"oracle_{split}_statistics",
                    "sample_count": int(stats.features.shape[0]),
                    "structure_cosine": proto_error["cosine"],
                    "prototype_l2": proto_error["l2"],
                    "prototype_relative_l2": proto_error["relative_l2"],
                    "angle_error_deg": proto_error["angle_error_deg"],
                    "structure_norm": proto_error["estimate_norm"],
                    "oracle_norm": proto_error["oracle_norm"],
                    "structure_value": "checkpoint.class_means (APR-corrected; classifier-adaptation source)",
                }
                prototype_rows.append(row)

                structure_covariance = structure_covariances_tensor[class_id].detach().cpu().numpy()
                cov_metrics = covariance_metrics(structure_covariance, stats.covariance)
                spectrum = spectrum_metrics(stats.features)
                delta_trace_structure = float("nan")
                delta_trace_oracle = float("nan")
                if split == "train":
                    if (
                        previous_oracle_task == task - 1
                        and class_id in previous_structure_traces
                    ):
                        delta_trace_structure = cov_metrics["trace_structure"] - previous_structure_traces[class_id]
                        delta_trace_oracle = cov_metrics["trace_oracle"] - previous_oracle_traces[class_id]
                    current_structure_traces[class_id] = cov_metrics["trace_structure"]
                    current_oracle_traces[class_id] = cov_metrics["trace_oracle"]
                covariance_rows.append(
                    {
                        "dataset": protocol_config["source_dataset"],
                        "protocol": options.protocol,
                        "task": task,
                        "class_id": class_id,
                        "birth_task": birth,
                        "class_age": task - birth,
                        "is_old": class_id < known_before,
                        "oracle_split": split,
                        "structure_covariance_type": (
                            "full stored/APR-corrected covariance"
                            if class_id < known_before
                            else "full current-class sample covariance + jitter"
                        ),
                        "oracle_covariance_type": "full sample covariance + matched jitter",
                        "delta_trace_structure": delta_trace_structure,
                        "delta_trace_oracle": delta_trace_oracle,
                        **cov_metrics,
                        **spectrum,
                    }
                )

                modes = ("diagonal", "full") if options.covariance_mode == "both" else (options.covariance_mode,)
                for mode in modes:
                    distance = mahalanobis_squared_diagonal if mode == "diagonal" else mahalanobis_squared_full
                    structure_d2 = distance(stats.features, structure_means[class_id], structure_covariance)
                    oracle_d2 = distance(stats.features, stats.mean, stats.covariance)
                    mahalanobis_rows.append(
                        {
                            "dataset": protocol_config["source_dataset"],
                            "protocol": options.protocol,
                            "task": task,
                            "class_id": class_id,
                            "birth_task": birth,
                            "class_age": task - birth,
                            "is_old": class_id < known_before,
                            "oracle_split": split,
                            "covariance_mode": mode,
                            **distribution_summary(structure_d2, "structure_d2"),
                            **distribution_summary(oracle_d2, "oracle_d2"),
                            "distance_pearson": pearson(structure_d2, oracle_d2),
                            "distance_spearman": spearman(structure_d2, oracle_d2),
                        }
                    )

                if split == "train":
                    task_train_means[class_id] = stats.mean.copy()
                    task_train_covariances[class_id] = stats.covariance.copy()
                    task_primary_rows[class_id] = row

            if split == "train":
                gaussian_diagnostic_rows.extend(
                    gaussian_rows(
                        {
                            "dataset": protocol_config["source_dataset"],
                            "protocol": options.protocol,
                            "task": task,
                            "oracle_split": "train",
                        },
                        oracle,
                        class_ids,
                    )
                )

            if split == "train":
                sampled_features, sampled_labels = _sample_for_visualization(
                    features,
                    labels,
                    class_ids,
                    options.visualization_samples_per_class,
                    options.seed + task,
                )
                feature_path = options.output_dir / "features" / f"task_{task:02d}.npz"
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    feature_path,
                    features=sampled_features,
                    labels=sampled_labels,
                    class_ids=class_ids,
                    oracle_means=np.stack([oracle[int(item)].mean for item in class_ids]).astype(np.float32),
                    structure_means=structure_means[class_ids].astype(np.float32),
                    preapr_means=preapr_means.astype(np.float32),
                    preapr_available=preapr_available,
                    birth_tasks=np.asarray([birth_task(int(item), partition) for item in class_ids]),
                )
            if split == "test":
                test_features, test_labels, test_predictions = features, labels, predictions
            del oracle, features, labels, predictions
            gc.collect()

        if "train" in options.splits:
            previous_structure_traces = current_structure_traces
            previous_oracle_traces = current_oracle_traces

        if "train" in options.splits and task > 1 and source_means is not None:
            source_means_np = np.asarray(source_means, dtype=np.float64)
            for class_id in class_ids[class_ids < known_before]:
                class_id = int(class_id)
                oracle_now = task_train_means[class_id]
                after = structure_means[class_id]
                before = source_means_np[class_id]
                before_error = vector_error(before, oracle_now)
                after_error = vector_error(after, oracle_now)
                apr_delta = after - before
                oracle_previous = (
                    previous_oracle_means.get(class_id)
                    if previous_oracle_task == task - 1
                    else None
                )
                drift = {
                    "oracle_drift_norm": float("nan"),
                    "apr_drift_norm": float(np.linalg.norm(apr_delta)),
                    "drift_cosine": float("nan"),
                    "drift_l2_error": float("nan"),
                    "drift_magnitude_relative_error": float("nan"),
                    "drift_angle_error_deg": float("nan"),
                    "drift_match_category": "unavailable_nonconsecutive_checkpoint",
                }
                if oracle_previous is not None:
                    oracle_delta = oracle_now - oracle_previous
                    oracle_norm = float(np.linalg.norm(oracle_delta))
                    apr_norm = float(np.linalg.norm(apr_delta))
                    direction = safe_cosine(apr_delta, oracle_delta)
                    magnitude_error = abs(apr_norm - oracle_norm) / (oracle_norm + EPS)
                    drift.update(
                        {
                            "oracle_drift_norm": oracle_norm,
                            "drift_cosine": direction,
                            "drift_l2_error": float(np.linalg.norm(apr_delta - oracle_delta)),
                            "drift_magnitude_relative_error": magnitude_error,
                            "drift_angle_error_deg": float(np.degrees(np.arccos(np.clip(direction, -1, 1)))) if np.isfinite(direction) else float("nan"),
                            "drift_match_category": classify_drift(direction, magnitude_error),
                        }
                    )
                reduction = before_error["l2"] - after_error["l2"]
                tolerance = max(1e-10, before_error["l2"] * 1e-6)
                improved = "improved" if reduction > tolerance else "worsened" if reduction < -tolerance else "unchanged"
                drift_rows.append(
                    {
                        "dataset": protocol_config["source_dataset"],
                        "protocol": options.protocol,
                        "task": task,
                        "class_id": class_id,
                        "birth_task": birth_task(class_id, partition),
                        "class_age": task - birth_task(class_id, partition),
                        "error_before_APR": before_error["l2"],
                        "error_after_APR": after_error["l2"],
                        "relative_error_before_APR": before_error["relative_l2"],
                        "relative_error_after_APR": after_error["relative_l2"],
                        "cosine_before_APR": before_error["cosine"],
                        "cosine_after_APR": after_error["cosine"],
                        "APR_error_reduction": reduction,
                        "APR_result": improved,
                        **drift,
                    }
                )
                apr_ablation_rows.append(
                    {
                        "dataset": protocol_config["source_dataset"],
                        "protocol": options.protocol,
                        "task": task,
                        "class_id": class_id,
                        "birth_task": birth_task(class_id, partition),
                        "class_age": task - birth_task(class_id, partition),
                        "prototype_role": "old_class",
                        "preAPR_source": "checkpoint.task_source_means",
                        "APR_source": "checkpoint.class_means",
                        "oracle_source": "old_train_samples_forwarded_at_current_checkpoint",
                        "cosine_preAPR": before_error["cosine"],
                        "cosine_APR": after_error["cosine"],
                        "l2_preAPR": before_error["l2"],
                        "l2_APR": after_error["l2"],
                        "relative_l2_preAPR": before_error["relative_l2"],
                        "relative_l2_APR": after_error["relative_l2"],
                        "angle_preAPR": before_error["angle_error_deg"],
                        "angle_APR": after_error["angle_error_deg"],
                        "APR_error_reduction": reduction,
                        "APR_result": improved,
                        "one_step_oracle_available": oracle_previous is not None,
                        "one_step_drift_error": drift["drift_l2_error"],
                        "cumulative_error": after_error["l2"],
                        "true_one_step_drift_norm": drift["oracle_drift_norm"],
                        "APR_one_step_drift_norm": drift["apr_drift_norm"],
                        "one_step_drift_cosine": drift["drift_cosine"],
                    }
                )

        if test_labels is not None:
            for class_id in class_ids:
                class_id = int(class_id)
                mask = test_labels == class_id
                accuracy = float(np.mean(test_predictions[mask] == class_id) * 100.0)
                previous_best = best_class_accuracy.get(class_id, accuracy)
                best_class_accuracy[class_id] = max(previous_best, accuracy)
                birth_class_accuracy.setdefault(class_id, accuracy)
                old_to_new = (
                    float(np.mean(test_predictions[mask] >= known_before))
                    if class_id < known_before and known_before > 0
                    else float("nan")
                )
                history = {
                    "dataset": protocol_config["source_dataset"],
                    "protocol": options.protocol,
                    "task": task,
                    "class_id": class_id,
                    "birth_task": birth_task(class_id, partition),
                    "class_age": task - birth_task(class_id, partition),
                    "is_old": class_id < known_before,
                    "num_test_samples": int(mask.sum()),
                    "class_accuracy": accuracy,
                    "best_previous_or_current_accuracy": best_class_accuracy[class_id],
                    "forgetting": best_class_accuracy[class_id] - accuracy,
                    "class_accuracy_drop": birth_class_accuracy[class_id] - accuracy,
                    "old_to_new_confusion": old_to_new,
                }
                class_history_rows.append(history)
                if class_id in task_primary_rows:
                    task_primary_rows[class_id].update(
                        {
                            "forgetting": history["forgetting"],
                            "class_accuracy_drop": history["class_accuracy_drop"],
                            "old_to_new_confusion": old_to_new,
                        }
                    )

            if known_before > 0 and task_train_covariances:
                new_mask = test_labels >= known_before
                if new_mask.any():
                    old_ids = np.asarray([item for item in class_ids if item < known_before], dtype=int)
                    old_structure_cov = [structure_covariances_tensor[int(item)].detach().cpu().numpy() for item in old_ids]
                    old_oracle_cov = [task_train_covariances[int(item)] for item in old_ids]
                    struct_rank = _nearest_old_diagonal(
                        test_features[new_mask], structure_means[old_ids], old_structure_cov
                    )
                    oracle_rank = _nearest_old_diagonal(
                        test_features[new_mask], np.stack([task_train_means[int(item)] for item in old_ids]), old_oracle_cov
                    )
                    ranking_rows.append(
                        {
                            "dataset": protocol_config["source_dataset"],
                            "protocol": options.protocol,
                            "task": task,
                            "sample_group": "current_new_test_samples",
                            "covariance_mode": "diagonal",
                            "num_samples": int(new_mask.sum()),
                            "nearest_old_agreement": float(np.mean(struct_rank == oracle_rank)),
                            "nearest_old_changed_rate": float(np.mean(struct_rank != oracle_rank)),
                        }
                    )

            config = checkpoint.get("config", {})
            scale = float(config.get("scale", 20.0))
            checkpoint_weights = checkpoint_classifier_weights(checkpoint).numpy()[class_ids]
            checkpoint_scores = cosine_scores(test_features, checkpoint_weights, scale)
            checkpoint_metrics = evaluate_scores(
                checkpoint_scores, test_labels, class_ids, known_before
            )
            diagnostic_base = {
                "dataset": protocol_config["source_dataset"],
                "protocol": options.protocol,
                "task": task,
                "known_before": known_before,
                "total_seen_classes": total_classes,
                "audited_class_count": int(class_ids.size),
                "evaluation_split": "test",
            }
            competition_rows.append(
                competition_row(diagnostic_base, "CHECKPOINT_MAIN", checkpoint_metrics)
            )
            checkpoint_margin = margin_rows(
                diagnostic_base,
                "CHECKPOINT_MAIN",
                checkpoint_scores,
                test_labels,
                class_ids,
                known_before,
            )
            margin_diagnostic_rows.extend(checkpoint_margin)

            if task_train_means:
                oracle_means_selected = np.stack(
                    [task_train_means[int(class_id)] for class_id in class_ids]
                )
                task_boundary_rows = boundary_direction_rows(
                    diagnostic_base,
                    checkpoint_scores,
                    test_labels,
                    class_ids,
                    known_before,
                    checkpoint_weights,
                    structure_means[class_ids],
                    oracle_means_selected,
                )
                history_lookup = {
                    int(row["class_id"]): row
                    for row in class_history_rows
                    if int(row["task"]) == task
                }
                for row in task_boundary_rows:
                    history = history_lookup.get(int(row["class_id"]), {})
                    row["forgetting"] = history.get("forgetting")
                    row["class_accuracy_drop"] = history.get("class_accuracy_drop")
                    row["old_to_new_confusion"] = history.get("old_to_new_confusion")
                boundary_rows.extend(task_boundary_rows)

                prototype_error_lookup = {
                    int(class_id): vector_error(
                        structure_means[int(class_id)], task_train_means[int(class_id)]
                    )["relative_l2"]
                    for class_id in class_ids
                }
                boundary_lookup = {
                    int(row["class_id"]): row["absolute_boundary_direction_error"]
                    for row in task_boundary_rows
                }
                for row in checkpoint_margin:
                    competitor = int(row["strongest_old_class_id"])
                    row["strongest_old_prototype_relative_l2"] = prototype_error_lookup.get(
                        competitor, float("nan")
                    )
                    row["strongest_old_boundary_direction_error"] = boundary_lookup.get(
                        competitor, float("nan")
                    )

            head_scheduled = task > 1 and (
                options.oracle_head_eval_tasks == "all"
                or (
                    options.oracle_head_eval_tasks == "final"
                    and task == expected_tasks
                )
            )
            if head_scheduled:
                if "train" not in options.splits or not task_train_means:
                    raise ValueError(
                        "oracle classifier replacement requires --splits to include train"
                    )
                if config.get("logit_norm") not in (None, 0, 0.0, ""):
                    raise NotImplementedError(
                        "diagnostic CA currently requires vanilla Structure logit_norm=None"
                    )
                initial_weights, sidecar, initialization_source = load_classifier_initialization(
                    checkpoint, checkpoint_path, class_ids, task
                )
                initialization_checksum = hashlib.sha256(
                    np.ascontiguousarray(initial_weights.numpy()).tobytes()
                ).hexdigest()
                sidecar_config = sidecar or {}
                epochs = int(
                    options.oracle_head_epochs
                    if options.oracle_head_epochs is not None
                    else sidecar_config.get("ca_epochs", config.get("ca_epochs", 5))
                )
                samples_per_class = int(
                    options.oracle_head_samples_per_class
                    if options.oracle_head_samples_per_class is not None
                    else sidecar_config.get(
                        "replay_samples_per_class",
                        config.get("replay_samples_per_class", 256),
                    )
                )
                task_size_argument = int(
                    sidecar_config.get("task_size_argument", partition[task - 1])
                )
                init_lr = float(sidecar_config.get("init_lr", config.get("init_lr", 0.01)))
                weight_decay = float(
                    sidecar_config.get("weight_decay", config.get("weight_decay", 5e-4))
                )
                head_scale = float(sidecar_config.get("scale", scale))
                exact_full_selection = (
                    initialization_source == "exact_pre_ca_sidecar"
                    and np.array_equal(class_ids, np.arange(total_classes))
                    and epochs == int(sidecar_config["ca_epochs"])
                    and samples_per_class
                    == int(sidecar_config["replay_samples_per_class"])
                )
                diagnostic_task_sizes = (
                    list(map(int, sidecar_config.get("task_sizes", partition[:task])))
                    if np.array_equal(class_ids, np.arange(total_classes))
                    else [int(class_ids.size)]
                )
                common_rng = (
                    sidecar_config.get("rng_state_before_ca")
                    if initialization_source == "exact_pre_ca_sidecar"
                    else None
                )
                structure_means_selected = structure_means[class_ids].copy()
                structure_covariances_selected = np.stack(
                    [
                        structure_covariances_tensor[int(class_id)].detach().cpu().numpy()
                        for class_id in class_ids
                    ]
                )
                oracle_means_selected = np.stack(
                    [task_train_means[int(class_id)] for class_id in class_ids]
                )
                oracle_covariances_selected = np.stack(
                    [task_train_covariances[int(class_id)] for class_id in class_ids]
                )
                condition_records = []
                for condition, replace_mean, replace_covariance in CONDITIONS:
                    condition_means = structure_means_selected.copy()
                    condition_covariances = structure_covariances_selected.copy()
                    old_positions = class_ids < known_before
                    if replace_mean:
                        condition_means[old_positions] = oracle_means_selected[old_positions]
                    if replace_covariance:
                        condition_covariances[old_positions] = oracle_covariances_selected[old_positions]
                    training_result = train_diagnostic_head(
                        initial_weights,
                        condition_means,
                        condition_covariances,
                        class_ids,
                        task_index=task - 1,
                        task_size=task_size_argument,
                        epochs=epochs,
                        samples_per_class=samples_per_class,
                        init_lr=init_lr,
                        weight_decay=weight_decay,
                        scale=head_scale,
                        device=options.device,
                        seed=options.seed + task * 10000,
                        rng_state=common_rng,
                        task_sizes=diagnostic_task_sizes,
                    )
                    condition_scores = cosine_scores(
                        test_features, training_result.weights, head_scale
                    )
                    condition_metrics = evaluate_scores(
                        condition_scores, test_labels, class_ids, known_before
                    )
                    record = {
                        **diagnostic_base,
                        "condition": condition,
                        "is_main_cil_result": False,
                        "diagnostic_head_only": True,
                        "initialization_source": initialization_source,
                        "initialization_checksum": initialization_checksum,
                        "exact_original_CA_replay_candidate": exact_full_selection,
                        "classifier_adaptation_epochs": epochs,
                        "synthetic_samples_per_class": samples_per_class,
                        "synthetic_batch_size": samples_per_class,
                        "task_size_argument": task_size_argument,
                        "classifier_parameter_layout": diagnostic_task_sizes,
                        "optimizer": "SGD(momentum=0.9)+CosineAnnealingLR",
                        "init_lr": init_lr,
                        "weight_decay": weight_decay,
                        "scale": head_scale,
                        "normalization": "L2-normalized feature and classifier rows (cosine head)",
                        "sampling_rng_policy": (
                            "same_saved_pre_CA_rng_for_every_condition"
                            if common_rng is not None
                            else "same_fixed_seed_for_every_condition"
                        ),
                        "sampling_seed_when_no_sidecar": options.seed + task * 10000,
                        "current_class_statistics_replaced": False,
                        "old_mean_replaced": replace_mean,
                        "old_covariance_replaced": replace_covariance,
                        "final_loss": training_result.final_loss,
                        "numerical_fallback_count": training_result.numerical_fallback_count,
                        "max_extra_jitter": training_result.max_extra_jitter,
                        "optimizer_parameter_count": training_result.optimizer_parameter_count,
                        "input_checksum_unchanged": training_result.input_checksum_unchanged,
                        **condition_metrics,
                    }
                    if condition == "A_STRUCTURE":
                        denominator = float(np.linalg.norm(checkpoint_weights)) + EPS
                        weight_relative_l2 = float(
                            np.linalg.norm(training_result.weights - checkpoint_weights)
                            / denominator
                        )
                        accuracy_delta = (
                            condition_metrics["seen_class_accuracy"]
                            - checkpoint_metrics["seen_class_accuracy"]
                        )
                        prediction_mismatches = int(
                            np.sum(
                                np.argmax(condition_scores, axis=1)
                                != np.argmax(checkpoint_scores, axis=1)
                            )
                        )
                        a_predictions = np.argmax(condition_scores, axis=1)
                        checkpoint_predictions = np.argmax(checkpoint_scores, axis=1)
                        per_class_prediction_mismatches = {
                            str(int(class_id)): int(
                                np.sum(
                                    (test_labels == int(class_id))
                                    & (a_predictions != checkpoint_predictions)
                                )
                            )
                            for class_id in class_ids
                        }
                        regression_pass = bool(
                            exact_full_selection
                            and training_result.numerical_fallback_count == 0
                            and weight_relative_l2 <= 1e-7
                            and abs(accuracy_delta) <= 1e-9
                            and prediction_mismatches == 0
                        )
                        record.update(
                            {
                                "A_vs_checkpoint_weight_relative_l2": weight_relative_l2,
                                "A_vs_checkpoint_accuracy_delta": accuracy_delta,
                                "A_vs_checkpoint_prediction_mismatches": prediction_mismatches,
                                "ca_regression_pass": regression_pass,
                            }
                        )
                        ca_regression_rows.append(
                            {
                                "protocol": options.protocol,
                                "task": task,
                                "status": "PASS" if regression_pass else "FAIL",
                                "initialization_source": initialization_source,
                                "exact_original_CA_replay_candidate": exact_full_selection,
                                "weight_relative_l2": weight_relative_l2,
                                "weight_tolerance": 1e-7,
                                "seen_accuracy_delta_pp": accuracy_delta,
                                "accuracy_tolerance_pp": 1e-9,
                                "prediction_mismatches": prediction_mismatches,
                                "per_class_prediction_mismatches": per_class_prediction_mismatches,
                                "num_predictions": int(test_labels.size),
                                "numerical_fallback_count": training_result.numerical_fallback_count,
                                "classifier_parameter_layout": diagnostic_task_sizes,
                                "rng_replay": "saved_pre_CA_rng_plus_epoch_eval_dataloader_seed_draw",
                            }
                        )
                    condition_records.append(record)
                    competition_rows.append(
                        competition_row(diagnostic_base, condition, condition_metrics)
                    )
                    condition_margin = margin_rows(
                        diagnostic_base,
                        condition,
                        condition_scores,
                        test_labels,
                        class_ids,
                        known_before,
                    )
                    margin_diagnostic_rows.extend(condition_margin)

                condition_accuracy = {
                    row["condition"]: row["seen_class_accuracy"]
                    for row in condition_records
                }
                baseline = condition_accuracy["A_STRUCTURE"]
                delta_mean = condition_accuracy["B_ORACLE_MEAN"] - baseline
                delta_covariance = condition_accuracy["C_ORACLE_COVARIANCE"] - baseline
                delta_full = condition_accuracy["D_FULL_ORACLE"] - baseline
                for row in condition_records:
                    row["delta_vs_A"] = row["seen_class_accuracy"] - baseline
                    row["delta_oracle_mean"] = delta_mean
                    row["delta_oracle_cov"] = delta_covariance
                    row["delta_full_oracle"] = delta_full
                oracle_replacement_rows.extend(condition_records)

            oracle_replacement_rows.append(
                {
                    **diagnostic_base,
                    "condition": "CHECKPOINT_MAIN",
                    "is_main_cil_result": True,
                    "diagnostic_head_only": False,
                    "initialization_source": "trained_checkpoint_classifier",
                    **checkpoint_metrics,
                }
            )

        drift_lookup = {(int(row["task"]), int(row["class_id"])): row for row in drift_rows}
        covariance_lookup = {
            (int(row["task"]), int(row["class_id"])): row
            for row in covariance_rows
            if row["oracle_split"] == "train"
        }
        task_records = []
        boundary_lookup_for_correlation = {
            (int(row["task"]), int(row["class_id"])): row
            for row in boundary_rows
        }
        for class_id, record in task_primary_rows.items():
            if class_id >= known_before:
                continue
            drift_record = drift_lookup.get((task, class_id), {})
            cov_record = covariance_lookup.get((task, class_id), {})
            boundary_record = boundary_lookup_for_correlation.get((task, class_id), {})
            merged = dict(record)
            merged.update(
                {
                    "drift_angle_error_deg": drift_record.get("drift_angle_error_deg"),
                    "drift_magnitude_relative_error": drift_record.get("drift_magnitude_relative_error"),
                    "covariance_frobenius_relative_error": cov_record.get("frobenius_relative_error"),
                    "absolute_boundary_direction_error": boundary_record.get(
                        "absolute_boundary_direction_error"
                    ),
                }
            )
            task_records.append(merged)
            all_correlation_records.append(merged)
        correlation_rows.extend(_correlation_rows(options.protocol, task, task_records))

        if "train" in options.splits:
            previous_oracle_means = {key: value.copy() for key, value in task_train_means.items()}
            previous_oracle_task = task
        del checkpoint, learner, structure_covariances_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write_json(
            progress_path,
            {
                "protocol": options.protocol,
                "status": "running",
                "current_task": None,
                "completed_tasks": audited_tasks,
                "started_at_unix": started,
                "updated_at_unix": time.time(),
            },
        )

    correlation_rows.extend(_correlation_rows(options.protocol, "all", all_correlation_records))
    prototype_lookup_for_margin = {
        (int(row["task"]), int(row["class_id"])): row.get("prototype_relative_l2")
        for row in prototype_rows
        if row.get("oracle_split") == "train"
    }
    boundary_lookup_for_margin = {
        (int(row["task"]), int(row["class_id"])): row.get(
            "absolute_boundary_direction_error"
        )
        for row in boundary_rows
    }
    for row in margin_diagnostic_rows:
        key = (int(row["task"]), int(row["strongest_old_class_id"]))
        row.setdefault(
            "strongest_old_prototype_relative_l2",
            prototype_lookup_for_margin.get(key, float("nan")),
        )
        row.setdefault(
            "strongest_old_boundary_direction_error",
            boundary_lookup_for_margin.get(key, float("nan")),
        )
    outputs = {
        "prototype_oracle_audit.csv": prototype_rows,
        "apr_oracle_drift_audit.csv": drift_rows,
        "covariance_oracle_audit.csv": covariance_rows,
        "mahalanobis_oracle_audit.csv": mahalanobis_rows,
        "prototype_classifier_audit.csv": classifier_rows,
        "class_forgetting_correlation.csv": correlation_rows,
        "class_accuracy_history.csv": class_history_rows,
        "mahalanobis_nearest_old_ranking.csv": ranking_rows,
        "apr_ablation_audit.csv": apr_ablation_rows,
        "oracle_classifier_replacement.csv": oracle_replacement_rows,
        "new_class_competition_diagnostics.csv": competition_rows,
        "boundary_direction_error.csv": boundary_rows,
        "margin_diagnostics.csv": margin_diagnostic_rows,
        "gaussian_assumption_diagnostics.csv": gaussian_diagnostic_rows,
    }
    progression_rows = []
    for task_id in sorted(
        {
            int(row["task"])
            for row in oracle_replacement_rows
            if row.get("condition") in {name for name, _, _ in CONDITIONS}
        }
    ):
        by_condition = {
            row["condition"]: row
            for row in oracle_replacement_rows
            if int(row["task"]) == task_id
            and row.get("condition") in {name for name, _, _ in CONDITIONS}
        }
        if not by_condition:
            continue
        a = by_condition.get("A_STRUCTURE", {})
        a_seen = float(a.get("seen_class_accuracy", float("nan")))
        progression_rows.append(
            {
                "protocol": options.protocol,
                "task": task_id,
                "structure_seen_accuracy": a_seen,
                "oracle_mean_seen_accuracy": by_condition.get("B_ORACLE_MEAN", {}).get(
                    "seen_class_accuracy"
                ),
                "oracle_covariance_seen_accuracy": by_condition.get(
                    "C_ORACLE_COVARIANCE", {}
                ).get("seen_class_accuracy"),
                "full_oracle_seen_accuracy": by_condition.get("D_FULL_ORACLE", {}).get(
                    "seen_class_accuracy"
                ),
                "oracle_mean_delta_vs_A": by_condition.get("B_ORACLE_MEAN", {}).get(
                    "delta_vs_A"
                ),
                "oracle_covariance_delta_vs_A": by_condition.get(
                    "C_ORACLE_COVARIANCE", {}
                ).get("delta_vs_A"),
                "full_oracle_delta_vs_A": by_condition.get("D_FULL_ORACLE", {}).get(
                    "delta_vs_A"
                ),
            }
        )
    outputs["task_progression_summary.csv"] = progression_rows
    for filename, rows in outputs.items():
        write_csv(options.output_dir / filename, rows)

    regression_status = (
        "PASS"
        if ca_regression_rows
        and all(row["status"] == "PASS" for row in ca_regression_rows)
        else ("FAIL" if ca_regression_rows else "NOT_RUN")
    )
    write_json(
        options.output_dir / "ca_regression_check.json",
        {
            "protocol": options.protocol,
            "status": regression_status,
            "required_for_scientific_interpretation": True,
            "checks": ca_regression_rows,
        },
    )

    after_fingerprints = [_file_fingerprint(path) for path in checkpoints]
    if before_fingerprints != after_fingerprints:
        raise RuntimeError("checkpoint changed during read-only audit")
    manifest = {
        "format_version": 2,
        "status": "complete",
        "protocol": options.protocol,
        "dataset": protocol_config["source_dataset"],
        "expected_tasks": expected_tasks,
        "audited_tasks": audited_tasks,
        "complete_temporal_coverage": audited_tasks == list(range(1, expected_tasks + 1)),
        "data_root": str(options.data_root),
        "oracle_splits": list(options.splits),
        "primary_oracle": "oracle_train_statistics",
        "covariance_mode": options.covariance_mode,
        "oracle_head_eval_tasks": options.oracle_head_eval_tasks,
        "oracle_head_epochs_override": options.oracle_head_epochs,
        "oracle_head_samples_per_class_override": options.oracle_head_samples_per_class,
        "max_classes": options.class_cap,
        "max_samples_per_class": options.sample_cap,
        "scientific_interpretation_allowed": (
            options.class_cap is None
            and options.sample_cap is None
            and options.oracle_head_epochs is None
            and options.oracle_head_samples_per_class is None
            and regression_status == "PASS"
            and audited_tasks == list(range(1, expected_tasks + 1))
        ),
        "ca_regression_status": regression_status,
        "checkpoint_fingerprints_before_after_equal": True,
        "leakage_guards": {
            "model_eval": True,
            "torch_no_grad": True,
            "parameters_require_grad_false": True,
            "optimizer_created_for_backbone_or_adapter": False,
            "backward_called_on_backbone_or_adapter": False,
            "diagnostic_optimizer_is_head_only": True,
            "diagnostic_input_checksum_check": True,
            "audit_values_returned_to_training": False,
            "checkpoint_read_only_fingerprint_check": True,
        },
        "checkpoint_fingerprints": before_fingerprints,
        "row_counts": {filename: len(rows) for filename, rows in outputs.items()},
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_dir / "audit_manifest.json", manifest)
    code_audit = Path(__file__).resolve().parents[1] / "structure_covariance_code_audit.md"
    if code_audit.is_file():
        (options.output_dir / code_audit.name).write_text(
            code_audit.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if not options.skip_visualizations and (options.output_dir / "features").is_dir():
        from oracle_audit.visualize import build_visualizations

        build_visualizations(options.output_dir, partition, options.seed)
    write_json(
        progress_path,
        {
            "protocol": options.protocol,
            "status": "complete",
            "current_task": None,
            "completed_tasks": audited_tasks,
            "started_at_unix": started,
            "updated_at_unix": time.time(),
            "elapsed_seconds": time.time() - started,
        },
    )
    return manifest
