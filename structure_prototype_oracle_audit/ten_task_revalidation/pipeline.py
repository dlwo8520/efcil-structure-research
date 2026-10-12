from __future__ import annotations

import gc
import json
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from classifier_diagnostic.pipeline import (
    _accuracy_row,
    _assemble_bank,
    _edge_fingerprint,
    _fixed_gaussian_bank,
    _stats,
    read_csv,
    write_csv,
    write_json,
)
from classifier_procedure.pipeline import (
    _make_epoch_permutations,
    _train_cosine,
)
from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT, protocol_config
from oracle_audit.oracle_head import (
    checkpoint_classifier_weights,
    cosine_scores,
    load_classifier_initialization,
)
from oracle_audit.runtime import (
    birth_task,
    build_data_manager,
    extract_split,
    load_frozen_learner,
    release_learner,
    torch_load,
)
from pair_boundary_risk.pipeline import (
    BoundaryRiskOptions,
    _fixed_scores,
    _pair_actual_rows,
    _ranking_analysis,
    _risk_rows,
    _sample_transition_rows,
)
from representation_relation.knn import cosine_knn, purity
from representation_relation.pipeline import _dataset_identity, _validation_indices

from .metrics import (
    EPS,
    aggregate,
    directional_rows,
    normalize,
    prototype_rows_from_snapshots,
    summary,
)


TEN_PROTOCOLS = ("IA10", "CAR10")
ALL_PROTOCOLS = ("IA5", "CAR5", "IA10", "CAR10")


@dataclass
class RevalidationOptions:
    output_root: Path
    device: torch.device
    physical_gpu: str
    seed: int = 1993
    batch_size: int = 64
    num_workers: int = 4
    phases: tuple[int, ...] = (1, 2, 3)
    protocols: tuple[str, ...] = TEN_PROTOCOLS
    random_trials: int = 1000
    force: bool = False


def _checkpoint_dir(protocol: str, seed: int) -> Path:
    return ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{seed}" / "task_checkpoints"


def _scientific_dir(protocol: str, seed: int) -> Path:
    return ROOT / "scientific_audit" / protocol / f"seed_{seed}"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _required_checkpoints(protocol: str, seed: int) -> tuple[list[Path], int]:
    config = protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])
    expected = int(config["expected_tasks"])
    return [
        _checkpoint_dir(protocol, seed) / f"task_{task:02d}.pt"
        for task in range(1, expected + 1)
    ], expected


def write_extension_audit(options: RevalidationOptions) -> dict[str, Any]:
    records: dict[str, Any] = {}
    lines = [
        "# 10-Task Extension Input Audit",
        "",
        f"- Seed: `{options.seed}`",
        f"- Requested physical GPU: `{options.physical_gpu}` (process-local `cuda:0`)",
        "- Policy: read-only checkpoint diagnostics; missing checkpoints are reported and never trained automatically.",
        "",
    ]
    missing_all: list[str] = []
    for protocol in ALL_PROTOCOLS:
        paths, expected = _required_checkpoints(protocol, options.seed)
        protocol_rows = []
        lines.extend([f"## {protocol}", "", "| Task | Checkpoint | Classifier | Mean | Covariance | Frozen cache |", "|---:|---|---|---|---|---|"])
        partition: list[int] = []
        for task, path in enumerate(paths, start=1):
            row: dict[str, Any] = {"task": task, "checkpoint": str(path), "exists": path.is_file()}
            if path.is_file():
                checkpoint = torch_load(path)
                partition = list(map(int, checkpoint.get("task_sizes", [])))
                try:
                    classifier_rows = int(checkpoint_classifier_weights(checkpoint).shape[0])
                except (KeyError, RuntimeError, ValueError):
                    classifier_rows = 0
                row.update({
                    "classifier": classifier_rows >= int(checkpoint.get("total_classes", 0)),
                    "classifier_rows": classifier_rows,
                    "stored_mean": len(checkpoint.get("class_means", [])) >= int(checkpoint.get("total_classes", 0)),
                    "stored_covariance": len(checkpoint.get("class_covs", [])) >= int(checkpoint.get("total_classes", 0)),
                    "total_classes": int(checkpoint.get("total_classes", -1)),
                    "stage": checkpoint.get("stage"),
                })
                del checkpoint
            else:
                missing_all.append(str(path))
                row.update({"classifier": False, "stored_mean": False, "stored_covariance": False})
            cache = options.output_root / protocol / ".cache" / f"model_{task:02d}_on_task_{task:02d}.npz"
            scientific_cache = _scientific_dir(protocol, options.seed) / "features" / f"task_{task:02d}.npz"
            row["frozen_cache"] = cache.is_file() or scientific_cache.is_file()
            row["frozen_cache_kind"] = (
                "full train/test" if cache.is_file()
                else "scientific full-statistics + sampled-train visualization cache" if scientific_cache.is_file()
                else "none"
            )
            protocol_rows.append(row)
            mark = lambda value: "YES" if value else "NO"
            lines.append(
                f"| T{task} | {mark(row['exists'])} | {mark(row['classifier'])} | "
                f"{mark(row['stored_mean'])} | {mark(row['stored_covariance'])} | {mark(row['frozen_cache'])} |"
            )
        config = protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])
        partition = list(map(int, config["task_partition"]))
        births = [birth_task(class_id, partition) for class_id in range(sum(partition))]
        data_manager, _ = build_data_manager(
            protocol, DEFAULT_DATA_ROOTS[protocol], options.seed
        )
        original_order = list(map(
            int, data_manager.protocol_manifest["class_order_original_ids"]
        ))
        lines.extend([
            "",
            f"- Task partition: `{partition}`",
            f"- Internal class order: contiguous remapped IDs `0..{sum(partition)-1}`.",
            f"- Original dataset class order: `{original_order}`",
            f"- Class birth tasks: `{births}`",
            "",
        ])
        records[protocol] = {
            "expected_tasks": expected,
            "partition": partition,
            "internal_to_original_class": original_order,
            "birth_task_by_internal_class": births,
            "tasks": protocol_rows,
        }
    lines.extend(["## Missing checkpoints", ""])
    if missing_all:
        lines.append("MISSING CHECKPOINTS:")
        lines.extend(f"- `{path}`" for path in missing_all)
    else:
        lines.append("MISSING CHECKPOINTS: none")
    path = options.output_root / "ten_task_extension_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(options.output_root / "input_audit.json", {"protocols": records, "missing_checkpoints": missing_all})
    return {"records": records, "missing": missing_all, "path": str(path)}


def _load_snapshots(protocol: str, seed: int) -> list[dict[str, np.ndarray]]:
    directory = _scientific_dir(protocol, seed) / "features"
    paths, expected = _required_checkpoints(protocol, seed)
    snapshots: list[dict[str, np.ndarray]] = []
    for task in range(1, expected + 1):
        path = directory / f"task_{task:02d}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"validated scientific feature summary is missing: {path}; rerun the existing audit, not CIL training"
            )
        values = np.load(path, allow_pickle=False)
        snapshots.append({key: np.asarray(values[key]) for key in values.files})
    return snapshots


def _copy_existing_scientific(protocol: str, options: RevalidationOptions) -> None:
    source = _scientific_dir(protocol, options.seed)
    output = options.output_root / protocol
    tracking_source = read_csv(source / "prototype_oracle_audit.csv")
    covariance = [row for row in read_csv(source / "covariance_oracle_audit.csv") if row.get("oracle_split") == "train"]
    replacement = read_csv(source / "oracle_classifier_replacement.csv")
    expected = int(protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])["expected_tasks"])
    final_abcd = [
        row for row in replacement
        if int(row["task"]) == expected
        and row.get("condition") in {"A_STRUCTURE", "B_ORACLE_MEAN", "C_ORACLE_COVARIANCE", "D_FULL_ORACLE", "CHECKPOINT_MAIN"}
    ]
    checkpoint_accuracy = {
        int(row["task"]): row
        for row in replacement
        if row.get("condition") == "CHECKPOINT_MAIN"
    }
    competition = [row for row in read_csv(source / "new_class_competition_diagnostics.csv") if row.get("condition") == "CHECKPOINT_MAIN"]
    for row in competition:
        reference = checkpoint_accuracy.get(int(row["task"]), {})
        row["seen_class_accuracy"] = reference.get("seen_class_accuracy")
        row["old_accuracy"] = row.get("old_single_head_acc")
        row["new_accuracy"] = row.get("single_head_new_acc")
    write_csv(output / "covariance_mismatch_by_task.csv", covariance)
    write_csv(
        output / "covariance_mismatch_by_age.csv",
        aggregate(covariance, ("protocol", "class_age"), ("frobenius_relative_error", "trace_ratio")),
    )
    write_csv(output / "oracle_abcd_final_task.csv", final_abcd)
    write_csv(output / "accuracy_competition_by_task.csv", competition)
    apr = read_csv(source / "apr_oracle_drift_audit.csv")
    write_csv(output / "apr_improvement_by_task.csv", apr)
    # Preserve provenance without silently treating the sampled visualization cache as a full cache.
    write_json(output / "scientific_reuse_manifest.json", {
        "source": str(source),
        "prototype_rows_available": len(tracking_source),
        "covariance_rows_reused": len(covariance),
        "final_abcd_rows_reused": len(final_abcd),
        "competition_rows_reused": len(competition),
        "oracle_means_in_feature_summaries_are_full_train_statistics": True,
    })
    if protocol.endswith("5"):
        relation = ROOT / "representation_relation_diagnostic" / f"{protocol}_T5"
        directional = read_csv(relation / "directional_pair_confusion.csv")
        for row in directional:
            row.update({
                "protocol": protocol,
                "task": 5,
                "error_count": row.get("confusion_count"),
            })
        write_csv(output / "directional_confusion_by_task.csv", directional)
        error_counts = np.asarray([float(row["confusion_count"]) for row in directional], dtype=np.float64)
        total_errors = float(error_counts.sum())
        concentration = {
            "protocol": protocol,
            "task": 5,
            "total_errors": int(total_errors),
            "nonzero_directional_pairs": len(directional),
            "top10_error_share": float(np.sort(error_counts)[::-1][:10].sum() / max(total_errors, 1.0)),
            "top20_error_share": float(np.sort(error_counts)[::-1][:20].sum() / max(total_errors, 1.0)),
            "top50_error_share": float(np.sort(error_counts)[::-1][:50].sum() / max(total_errors, 1.0)),
        }
        write_csv(output / "error_concentration_by_task.csv", [concentration] if directional else [])
        boundary = ROOT / "pair_boundary_risk_diagnostic" / f"{protocol}_T4_T5"
        pair_rows = read_csv(boundary / "pair_confusion_boundary_metrics.csv")
        sample_rows = read_csv(boundary / "same_sample_prediction_transition.csv")
        for rows in (pair_rows, sample_rows):
            _transition_metadata(rows, protocol, 4, 5)
        write_csv(output / "pair_degradation_by_transition.csv", pair_rows)
        write_csv(output / "same_sample_transition_all_tasks.csv", sample_rows)
        counts = Counter(row["prediction_transition_type"] for row in sample_rows)
        write_csv(output / "transition_summary.csv", [{
            "protocol": protocol,
            "previous_task": 4,
            "task": 5,
            "transition": "T4->T5",
            "sample_count": len(sample_rows),
            "correct_to_wrong": counts["correct_to_wrong_B"],
            "wrong_to_correct": counts["wrong_B_to_correct"],
            "wrong_destination_change": counts["wrong_C_to_wrong_B"],
            "eligible_newly_degraded_pair_count": sum(row.get("newly_degraded_rank") not in (None, "") for row in pair_rows),
            "positive_degradation_mass": int(sum(float(row["positive_boundary_degradation_count"]) for row in pair_rows)),
            "improvement_mass": int(sum(max(float(row["boundary_violation_count_t4"]) - float(row["boundary_violation_count_t5"]), 0.0) for row in pair_rows)),
            "fixed_common_readout": "T5 Structure cosine head on T4/T5 features",
            "source": str(boundary),
        }])
        gamma = read_csv(boundary / "stored_oracle_boundary_risk.csv")
        _transition_metadata(gamma, protocol, 4, 5)
        write_csv(output / "boundary_gamma_by_transition.csv", gamma)
        ranking: list[dict[str, Any]] = []
        for filename, row_type in (
            ("risk_ranking_summary.csv", "summary"),
            ("risk_recall_at_k.csv", "recall_at_k"),
            ("captured_degradation_mass.csv", "captured_mass"),
            ("rank_correlation_summary.csv", "correlation"),
        ):
            rows = read_csv(boundary / filename)
            for row in rows:
                row.update({
                    "row_type": row_type,
                    "protocol": protocol,
                    "previous_task": 4,
                    "task": 5,
                    "transition": "T4->T5",
                    "source": str(boundary / filename),
                })
            ranking.extend(rows)
        write_csv(output / "risk_ranking_by_transition.csv", ranking)


def _write_prototype_analysis(protocol: str, options: RevalidationOptions) -> None:
    tracking, temporal, interclass = prototype_rows_from_snapshots(protocol, _load_snapshots(protocol, options.seed))
    output = options.output_root / protocol
    write_csv(output / "prototype_tracking_by_task.csv", tracking)
    write_csv(
        output / "prototype_tracking_by_age.csv",
        aggregate(tracking, ("protocol", "class_age", "is_old"), ("tracking_similarity", "tracking_relative_l2")),
    )
    write_csv(output / "prototype_temporal_drift.csv", temporal)
    write_csv(output / "interclass_prototype_similarity.csv", interclass)


def _feature_cache_path(output: Path, model_task: int, class_task: int) -> Path:
    return output / ".cache" / f"model_{model_task:02d}_on_task_{class_task:02d}.npz"


def _extract_feature_cache(
    *,
    protocol: str,
    model_task: int,
    class_task: int,
    include_train: bool,
    data_manager: Any,
    partition: list[int],
    options: RevalidationOptions,
) -> Path:
    output = options.output_root / protocol
    cache = _feature_cache_path(output, model_task, class_task)
    if cache.is_file() and not options.force:
        values = np.load(cache, allow_pickle=False)
        required = {"test_features", "test_labels", "test_sample_ids"}
        if include_train:
            required |= {"train_features", "train_labels", "train_sample_ids"}
        if required.issubset(values.files):
            return cache
    # Reuse the already validated full T5 cache instead of repeating extraction.
    legacy = (
        ROOT / "classifier_bottleneck_diagnostic" / protocol / f"seed_{options.seed}"
        / ".cache" / f"task_{class_task:02d}_features.npz"
    )
    if (
        model_task == class_task
        and include_train
        and not options.force
        and legacy.is_file()
    ):
        legacy_values = np.load(legacy, allow_pickle=False)
        total_classes = sum(partition[:class_task])
        class_ids = np.arange(total_classes, dtype=np.int64)
        train_ids, train_identity_labels = _dataset_identity(data_manager, class_ids, "train")
        test_ids, test_identity_labels = _dataset_identity(data_manager, class_ids, "test")
        if not np.array_equal(legacy_values["train_labels"], train_identity_labels):
            raise RuntimeError(f"{protocol}: reused train-cache label order mismatch")
        if not np.array_equal(legacy_values["test_labels"], test_identity_labels):
            raise RuntimeError(f"{protocol}: reused test-cache label order mismatch")
        _atomic_npz(
            cache,
            train_features=np.asarray(legacy_values["train_features"]),
            train_labels=np.asarray(legacy_values["train_labels"]),
            train_sample_ids=train_ids,
            test_features=np.asarray(legacy_values["test_features"]),
            test_labels=np.asarray(legacy_values["test_labels"]),
            test_sample_ids=test_ids,
            class_ids=class_ids,
            model_task=np.asarray(model_task),
            class_task=np.asarray(class_task),
        )
        return cache
    total_classes = sum(partition[:class_task])
    class_ids = np.arange(total_classes, dtype=np.int64)
    checkpoint_path = _checkpoint_dir(protocol, options.seed) / f"task_{model_task:02d}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"MISSING CHECKPOINTS: {checkpoint_path}")
    checkpoint = torch_load(checkpoint_path)
    learner = load_frozen_learner(checkpoint, options.device, output / ".scratch" / f"m{model_task:02d}_c{class_task:02d}")
    test_ids, expected_test_labels = _dataset_identity(data_manager, class_ids, "test")
    test_features, test_labels, _ = extract_split(
        learner, data_manager, class_ids, "test", options.batch_size,
        options.num_workers, None, options.seed + model_task * 100 + class_task,
    )
    if not np.array_equal(test_labels, expected_test_labels):
        raise RuntimeError(f"{protocol} model T{model_task}/classes T{class_task}: test identity order mismatch")
    arrays: dict[str, np.ndarray] = {
        "test_features": test_features,
        "test_labels": test_labels,
        "test_sample_ids": test_ids,
        "class_ids": class_ids,
        "model_task": np.asarray(model_task),
        "class_task": np.asarray(class_task),
    }
    if include_train:
        train_ids, expected_train_labels = _dataset_identity(data_manager, class_ids, "train")
        train_features, train_labels, _ = extract_split(
            learner, data_manager, class_ids, "train", options.batch_size,
            options.num_workers, None, options.seed + model_task * 100 + class_task + 1,
        )
        if not np.array_equal(train_labels, expected_train_labels):
            raise RuntimeError(f"{protocol} model T{model_task}/classes T{class_task}: train identity order mismatch")
        arrays.update({
            "train_features": train_features,
            "train_labels": train_labels,
            "train_sample_ids": train_ids,
        })
    _atomic_npz(cache, **arrays)
    release_learner(learner)
    del checkpoint
    return cache


def _transition_metadata(rows: list[dict[str, Any]], protocol: str, previous: int, task: int) -> None:
    for row in rows:
        row.update({
            "protocol": protocol,
            "previous_task": previous,
            "task": task,
            "transition": f"T{previous}->T{task}",
            "previous_column_alias": "t4 means previous task irrespective of numeric task",
            "current_column_alias": "t5 means current task irrespective of numeric task",
        })


def _run_phase1_protocol(protocol: str, options: RevalidationOptions) -> dict[str, Any]:
    started = time.time()
    output = options.output_root / protocol
    output.mkdir(parents=True, exist_ok=True)
    _copy_existing_scientific(protocol, options)
    _write_prototype_analysis(protocol, options)
    data_manager, config = build_data_manager(protocol, DEFAULT_DATA_ROOTS[protocol], options.seed)
    partition = list(map(int, config["task_partition"]))
    expected = len(partition)
    class_order = list(map(int, data_manager.protocol_manifest["class_order_original_ids"]))
    write_json(output / "class_order.json", {
        "protocol": protocol,
        "internal_to_original_class": class_order,
        "partition": partition,
        "birth_task_by_internal_class": [birth_task(i, partition) for i in range(sum(partition))],
    })
    sample_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    directional: list[dict[str, Any]] = []
    concentration: list[dict[str, Any]] = []
    transition_summary: list[dict[str, Any]] = []
    fingerprints_before: dict[str, dict[str, Any]] = {}
    for task in range(1, expected + 1):
        current_checkpoint = _checkpoint_dir(protocol, options.seed) / f"task_{task:02d}.pt"
        fingerprints_before[str(current_checkpoint)] = _edge_fingerprint(current_checkpoint)
        current_cache = _extract_feature_cache(
            protocol=protocol, model_task=task, class_task=task,
            include_train=(task == expected), data_manager=data_manager,
            partition=partition, options=options,
        )
        current_values = np.load(current_cache, allow_pickle=False)
        checkpoint = torch_load(current_checkpoint)
        class_count = sum(partition[:task])
        known_before = class_count - partition[task - 1]
        weights = normalize(checkpoint_classifier_weights(checkpoint).numpy()[:class_count])
        current_scores = _fixed_scores(current_values["test_features"], weights)
        current_predictions = np.argmax(current_scores, axis=1)
        task_directional, task_concentration = directional_rows(
            protocol, task, current_values["test_labels"], current_predictions, known_before
        )
        directional.extend(task_directional)
        concentration.append(task_concentration)
        if task > 1:
            previous_cache = _extract_feature_cache(
                protocol=protocol, model_task=task - 1, class_task=task,
                include_train=False, data_manager=data_manager,
                partition=partition, options=options,
            )
            previous_values = np.load(previous_cache, allow_pickle=False)
            if not np.array_equal(previous_values["test_sample_ids"], current_values["test_sample_ids"]):
                raise RuntimeError(f"{protocol} T{task-1}->T{task}: canonical sample IDs differ")
            if not np.array_equal(previous_values["test_labels"], current_values["test_labels"]):
                raise RuntimeError(f"{protocol} T{task-1}->T{task}: labels differ")
            previous_scores = _fixed_scores(previous_values["test_features"], weights)
            previous_predictions = np.argmax(previous_scores, axis=1)
            task_samples = _sample_transition_rows(
                current_values["test_sample_ids"], current_values["test_labels"],
                previous_predictions, current_predictions, partition,
            )
            metrics, margins = _pair_actual_rows(
                current_values["test_labels"], previous_scores, current_scores,
                previous_predictions, current_predictions, current_values["test_labels"],
                partition, known_before,
            )
            margin_lookup = {(row["true_class"], row["competitor_class"]): row for row in margins}
            for row in metrics:
                margin = margin_lookup[(row["true_class"], row["competitor_class"])]
                row.update({
                    "previous_margin_q05": margin["t4_q05"],
                    "current_margin_q05": margin["t5_q05"],
                    "delta_margin_q05": margin["delta_margin_q05"],
                    "true_class_birth_task": birth_task(int(row["true_class"]), partition),
                    "true_class_age": task - birth_task(int(row["true_class"]), partition),
                    "train_sample_count_i": None,
                    "train_sample_count_j": None,
                    "train_support_available": False,
                    "support_note": "phase-1 pair dynamics use held-out support; train support is added in selected phase-2 gamma runs",
                })
            _transition_metadata(task_samples, protocol, task - 1, task)
            _transition_metadata(metrics, protocol, task - 1, task)
            sample_rows.extend(task_samples)
            pair_rows.extend(metrics)
            transition_counts = Counter(row["prediction_transition_type"] for row in task_samples)
            transition_summary.append({
                "protocol": protocol,
                "previous_task": task - 1,
                "task": task,
                "transition": f"T{task-1}->T{task}",
                "sample_count": len(task_samples),
                "correct_to_wrong": transition_counts["correct_to_wrong_B"],
                "wrong_to_correct": transition_counts["wrong_B_to_correct"],
                "wrong_destination_change": transition_counts["wrong_C_to_wrong_B"],
                "eligible_newly_degraded_pair_count": sum(row.get("newly_degraded_rank") is not None for row in metrics),
                "positive_degradation_mass": int(sum(row["positive_boundary_degradation_count"] for row in metrics)),
                "improvement_mass": int(sum(max(row["boundary_violation_count_t4"] - row["boundary_violation_count_t5"], 0) for row in metrics)),
                "net_boundary_violation_change": int(sum(row["boundary_violation_count_t5"] - row["boundary_violation_count_t4"] for row in metrics)),
                "fixed_common_readout": f"T{task} Structure cosine head on T{task-1}/T{task} features",
            })
        del checkpoint
        write_csv(output / "same_sample_transition_all_tasks.csv", sample_rows)
        write_csv(output / "pair_degradation_by_transition.csv", pair_rows)
        write_csv(output / "directional_confusion_by_task.csv", directional)
        write_csv(output / "error_concentration_by_task.csv", concentration)
        write_csv(output / "transition_summary.csv", transition_summary)
        write_json(output / "phase1_progress.json", {"status": "running", "completed_through_task": task, "expected_tasks": expected})
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fingerprints_after = {path: _edge_fingerprint(Path(path)) for path in fingerprints_before}
    if fingerprints_before != fingerprints_after:
        raise RuntimeError(f"{protocol}: checkpoint fingerprint changed during phase 1")
    result = {
        "status": "complete",
        "protocol": protocol,
        "tasks": expected,
        "transitions": expected - 1,
        "class_order_recorded": True,
        "same_sample_linkage": "canonical original sample IDs",
        "fixed_common_readout": True,
        "checkpoint_fingerprints_before": fingerprints_before,
        "checkpoint_fingerprints_after": fingerprints_after,
        "checkpoints_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "phase1_manifest.json", result)
    write_json(output / "phase1_progress.json", result)
    return result


def _generalization_analysis(
    protocol: str,
    task: int,
    cache_path: Path,
    checkpoint: dict[str, Any],
    partition: list[int],
    options: RevalidationOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    values = np.load(cache_path, allow_pickle=False)
    train_features = np.asarray(values["train_features"], dtype=np.float32)
    train_labels = np.asarray(values["train_labels"], dtype=np.int64)
    test_features = np.asarray(values["test_features"], dtype=np.float32)
    test_labels = np.asarray(values["test_labels"], dtype=np.int64)
    train_ids = np.asarray(values["train_sample_ids"])
    test_ids = np.asarray(values["test_sample_ids"])
    class_ids = np.arange(sum(partition[:task]), dtype=np.int64)
    train_indices, validation_indices, _, _ = _validation_indices(
        train_labels, class_ids, options.seed + task * 1700
    )
    fit_features, fit_labels = train_features[train_indices], train_labels[train_indices]
    fit_ids = train_ids[train_indices]
    split_data = {
        "train": (fit_features, fit_labels, fit_ids),
        "validation": (
            train_features[validation_indices], train_labels[validation_indices], train_ids[validation_indices]
        ),
        "test": (test_features, test_labels, test_ids),
    }
    weights = checkpoint_classifier_weights(checkpoint).numpy()[: class_ids.size]
    normalized_weights = normalize(weights)
    oracle_means = np.stack([fit_features[fit_labels == class_id].mean(axis=0) for class_id in class_ids])
    normalized_means = normalize(oracle_means)
    margin_rows: list[dict[str, Any]] = []
    purity_rows: list[dict[str, Any]] = []
    geometry_samples: list[dict[str, Any]] = []
    for split, (features, labels, sample_ids) in split_data.items():
        if labels.size == 0:
            continue
        scores = _fixed_scores(features, normalized_weights)
        predictions = np.argmax(scores, axis=1)
        rows = np.arange(labels.size)
        rival_scores = scores.copy()
        rival_scores[rows, labels] = -np.inf
        rivals = np.argmax(rival_scores, axis=1)
        margins = scores[rows, labels] - scores[rows, rivals]
        for group_name, mask in (
            ("all", np.ones(labels.size, dtype=bool)),
            ("correct", predictions == labels),
            ("wrong", predictions != labels),
        ):
            selected = margins[mask]
            margin_rows.append({
                "protocol": protocol,
                "task": task,
                "split": split,
                "sample_group": group_name,
                **summary(selected, "margin_"),
                "negative_margin_fraction": float(np.mean(selected < 0.0)) if selected.size else float("nan"),
                "q05": float(np.quantile(selected, 0.05)) if selected.size else float("nan"),
                "q10": float(np.quantile(selected, 0.10)) if selected.size else float("nan"),
                "test_used_for_selection": False,
            })
        if split == "train":
            continue
        result = cosine_knn(
            fit_features, fit_labels, features, k=5, device=options.device,
            reference_ids=fit_ids, query_ids=sample_ids,
        )
        sample_purity = purity(result.neighbor_labels, labels)
        sample_normalized = normalize(features)
        prototype_similarity = sample_normalized @ normalized_means.T
        own_similarity = prototype_similarity[rows, labels]
        rival_proto = prototype_similarity.copy()
        rival_proto[rows, labels] = -np.inf
        nearest_rival = np.argmax(rival_proto, axis=1)
        rival_similarity = rival_proto[rows, nearest_rival]
        for group_name, mask in (
            ("all", np.ones(labels.size, dtype=bool)),
            ("correct", predictions == labels),
            ("wrong", predictions != labels),
        ):
            selected = sample_purity[mask]
            purity_rows.append({
                "protocol": protocol,
                "task": task,
                "split": split,
                "sample_group": group_name,
                **summary(selected, "purity_"),
            })
        for index in range(labels.size):
            label = int(labels[index])
            geometry_samples.append({
                "protocol": protocol,
                "task": task,
                "split": split,
                "sample_id": str(sample_ids[index]),
                "class_id": label,
                "birth_task": birth_task(label, partition),
                "class_age": task - birth_task(label, partition),
                "correct": bool(predictions[index] == label),
                "own_prototype_cosine_distance": float(1.0 - own_similarity[index]),
                "nearest_rival_class": int(nearest_rival[index]),
                "nearest_rival_prototype_cosine_distance": float(1.0 - rival_similarity[index]),
                "prototype_preference": float(own_similarity[index] - rival_similarity[index]),
                "own_class_neighborhood_purity": float(sample_purity[index]),
                "classifier_true_vs_rival_margin": float(margins[index]),
            })
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in geometry_samples:
        key = (row["protocol"], row["task"], row["split"], row["class_age"], row["correct"])
        grouped.setdefault(key, []).append(row)
    geometry_summary: list[dict[str, Any]] = []
    metrics = (
        "own_prototype_cosine_distance",
        "nearest_rival_prototype_cosine_distance",
        "prototype_preference",
        "own_class_neighborhood_purity",
        "classifier_true_vs_rival_margin",
    )
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        row = dict(zip(("protocol", "task", "split", "class_age", "correct"), key))
        for metric in metrics:
            row.update(summary((member[metric] for member in members), f"{metric}_"))
        geometry_summary.append(row)
    return margin_rows, purity_rows, geometry_summary


def _run_phase2_protocol(protocol: str, options: RevalidationOptions) -> dict[str, Any]:
    started = time.time()
    output = options.output_root / protocol
    data_manager, config = build_data_manager(protocol, DEFAULT_DATA_ROOTS[protocol], options.seed)
    partition = list(map(int, config["task_partition"]))
    expected = len(partition)
    transition_rows = read_csv(output / "transition_summary.csv")
    if protocol in TEN_PROTOCOLS and not transition_rows:
        raise RuntimeError(f"{protocol}: phase 1 transition summary is required before phase 2")
    selected_tasks: set[int] = {expected}
    if transition_rows:
        worst = max(
            transition_rows,
            key=lambda row: (float(row["positive_degradation_mass"]), int(row["task"])),
        )
        selected_tasks.add(int(worst["task"]))
    gamma_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    if protocol in TEN_PROTOCOLS:
        for task in sorted(selected_tasks):
            if task <= 1:
                continue
            current_cache = _extract_feature_cache(
                protocol=protocol, model_task=task, class_task=task,
                include_train=True, data_manager=data_manager, partition=partition, options=options,
            )
            previous_cache = _extract_feature_cache(
                protocol=protocol, model_task=task - 1, class_task=task,
                include_train=False, data_manager=data_manager, partition=partition, options=options,
            )
            current = np.load(current_cache, allow_pickle=False)
            previous = np.load(previous_cache, allow_pickle=False)
            checkpoint_path = _checkpoint_dir(protocol, options.seed) / f"task_{task:02d}.pt"
            checkpoint = torch_load(checkpoint_path)
            classes = sum(partition[:task])
            known_before = classes - partition[task - 1]
            weights = normalize(checkpoint_classifier_weights(checkpoint).numpy()[:classes])
            previous_scores = _fixed_scores(previous["test_features"], weights)
            current_scores = _fixed_scores(current["test_features"], weights)
            previous_predictions = np.argmax(previous_scores, axis=1)
            current_predictions = np.argmax(current_scores, axis=1)
            metrics, margins = _pair_actual_rows(
                current["test_labels"], previous_scores, current_scores,
                previous_predictions, current_predictions, current["train_labels"],
                partition, known_before,
            )
            risk_options = BoundaryRiskOptions(
                protocol=protocol,
                t4_checkpoint=checkpoint_path,
                t5_checkpoint=checkpoint_path,
                t4_feature_cache=previous_cache,
                t5_feature_cache=current_cache,
                output_dir=output,
                device=options.device,
                seed=options.seed,
                random_trials=options.random_trials,
            )
            risk = _risk_rows(
                risk_options,
                np.asarray(current["train_features"], dtype=np.float32),
                np.asarray(current["train_labels"], dtype=np.int64),
                weights,
                np.asarray(checkpoint["class_means"][:classes], dtype=np.float64),
                list(checkpoint["class_covs"][:classes]),
                float(checkpoint["config"].get("stats_cov_jitter", 1e-3)),
                metrics,
                margins,
            )
            _transition_metadata(risk, protocol, task - 1, task)
            for row in risk:
                row["true_class_birth_task"] = birth_task(int(row["true_class"]), partition)
                row["true_class_age"] = task - int(row["true_class_birth_task"])
                row["selection_reason"] = "final" if task == expected else "largest_phase1_positive_degradation_mass"
            gamma_rows.extend(risk)
            summary_rows, recall, captured, correlations = _ranking_analysis(
                risk, (10, 20, 50, 100), options.seed + task * 7100, options.random_trials
            )
            for row_type, rows in (
                ("summary", summary_rows), ("recall_at_k", recall),
                ("captured_mass", captured), ("correlation", correlations),
            ):
                for row in rows:
                    row.update({
                        "row_type": row_type,
                        "protocol": protocol,
                        "previous_task": task - 1,
                        "task": task,
                        "transition": f"T{task-1}->T{task}",
                        "selection_reason": "final" if task == expected else "largest_phase1_positive_degradation_mass",
                    })
                ranking_rows.extend(rows)
            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if protocol in TEN_PROTOCOLS:
        write_csv(output / "boundary_gamma_by_transition.csv", gamma_rows)
        write_csv(output / "risk_ranking_by_transition.csv", ranking_rows)

    final_cache = _extract_feature_cache(
        protocol=protocol, model_task=expected, class_task=expected,
        include_train=True, data_manager=data_manager, partition=partition, options=options,
    )
    checkpoint_path = _checkpoint_dir(protocol, options.seed) / f"task_{expected:02d}.pt"
    before = _edge_fingerprint(checkpoint_path)
    checkpoint = torch_load(checkpoint_path)
    margin_rows, purity_rows, geometry_rows = _generalization_analysis(
        protocol, expected, final_cache, checkpoint, partition, options
    )
    write_csv(output / "generalization_margin_by_task.csv", margin_rows)
    write_csv(output / "neighborhood_purity_by_task.csv", purity_rows)
    write_csv(output / "sample_geometry_summary.csv", geometry_rows)
    after = _edge_fingerprint(checkpoint_path)
    if before != after:
        raise RuntimeError(f"{protocol}: final checkpoint changed during phase 2")
    result = {
        "status": "complete",
        "protocol": protocol,
        "gamma_transition_tasks": sorted(selected_tasks) if protocol in TEN_PROTOCOLS else [],
        "gamma_policy": "final plus phase-1 transition with largest positive degradation mass",
        "generalization_task": expected,
        "knn_k": 5,
        "validation_selection": "deterministic class-stratified training-only split",
        "test_used_for_selection": False,
        "checkpoint_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "phase2_manifest.json", result)
    return result


def _run_phase3_protocol(protocol: str, options: RevalidationOptions) -> dict[str, Any]:
    if protocol not in TEN_PROTOCOLS:
        return {"status": "reused_5task", "protocol": protocol}
    started = time.time()
    output = options.output_root / protocol
    _, config = build_data_manager(protocol, DEFAULT_DATA_ROOTS[protocol], options.seed)
    partition = list(map(int, config["task_partition"]))
    task = len(partition)
    checkpoint_path = _checkpoint_dir(protocol, options.seed) / f"task_{task:02d}.pt"
    before = _edge_fingerprint(checkpoint_path)
    checkpoint = torch_load(checkpoint_path)
    cache_path = _feature_cache_path(output, task, task)
    if not cache_path.is_file():
        raise FileNotFoundError(f"phase 2 full final-task cache required: {cache_path}")
    values = np.load(cache_path, allow_pickle=False)
    train_features = np.asarray(values["train_features"], dtype=np.float32)
    train_labels = np.asarray(values["train_labels"], dtype=np.int64)
    test_features = np.asarray(values["test_features"], dtype=np.float32)
    test_labels = np.asarray(values["test_labels"], dtype=np.int64)
    classes = int(checkpoint["total_classes"])
    class_ids = np.arange(classes, dtype=np.int64)
    known_before = classes - partition[-1]
    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle_means, oracle_covariances, actual_banks = _stats(
        train_features, train_labels, class_ids, jitter
    )
    stored_means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float64)
    stored_covariances = np.stack([
        value.detach().cpu().numpy() for value in checkpoint["class_covs"][:classes]
    ])
    initial_weights, sidecar, source = load_classifier_initialization(
        checkpoint, checkpoint_path, class_ids, task
    )
    if sidecar is None or source != "exact_pre_ca_sidecar":
        raise RuntimeError(f"{protocol} T{task}: exact pre-CA classifier sidecar required")
    samples_per_class = int(sidecar["replay_samples_per_class"])
    epochs = int(sidecar["ca_epochs"])
    old_counts = {class_id: samples_per_class for class_id in range(known_before)}
    new_counts = {class_id: samples_per_class for class_id in range(known_before, classes)}
    gaussian_old = _fixed_gaussian_bank(
        oracle_means, oracle_covariances, old_counts, task - 1,
        int(sidecar["task_size_argument"]), options.device, options.seed + task * 1000 + 101,
    )
    common_new = _fixed_gaussian_bank(
        stored_means, stored_covariances, new_counts, task - 1,
        int(sidecar["task_size_argument"]), options.device, options.seed + task * 1000 + 102,
    )
    rng = np.random.RandomState(options.seed + task * 1000 + 103)
    actual_old: dict[int, np.ndarray] = {}
    for class_id in range(known_before):
        bank = actual_banks[class_id]
        actual_old[class_id] = bank[rng.randint(0, bank.shape[0], size=samples_per_class)].copy()
    gaussian_features, gaussian_labels = _assemble_bank({**gaussian_old, **common_new}, class_ids)
    actual_features, actual_labels = _assemble_bank({**actual_old, **common_new}, class_ids)
    if not np.array_equal(gaussian_labels, actual_labels):
        raise AssertionError("G256/R256 labels differ")
    permutations = _make_epoch_permutations(
        gaussian_labels, epochs, options.seed + task * 1000 + 200
    )
    common = dict(
        protocol=protocol,
        initial_weights=initial_weights,
        task_sizes=list(map(int, sidecar["task_sizes"])),
        validation_features=np.empty((0, train_features.shape[1]), dtype=np.float32),
        validation_labels=np.empty(0, dtype=np.int64),
        scale=float(sidecar["scale"]),
        init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]),
        max_epochs=epochs,
        batch_size=samples_per_class,
        device=options.device,
        permutations=permutations,
        convergence=False,
        min_epochs=epochs,
        patience=epochs,
        relative_loss_threshold=1e-4,
    )
    g256 = _train_cosine(
        condition="G256_FIXED", train_features=gaussian_features,
        train_labels=gaussian_labels, **common,
    )
    r256 = _train_cosine(
        condition="R256_FIXED", train_features=actual_features,
        train_labels=actual_labels, **common,
    )
    expected_steps = epochs * classes
    if g256.optimizer_steps != expected_steps or r256.optimizer_steps != expected_steps:
        raise AssertionError("G256/R256 optimizer-step mismatch")
    rows: list[dict[str, Any]] = []
    for condition, result in (("G256_FIXED", g256), ("R256_FIXED", r256)):
        scores = cosine_scores(test_features, result.weights, float(sidecar["scale"]))
        row = _accuracy_row(
            protocol, task, condition, scores, test_labels, class_ids, known_before,
            samples_per_class=samples_per_class,
            optimizer_steps=result.optimizer_steps,
            label_sequence_identical=True,
            current_new_bank_identical=True,
            pre_ca_initialization_source=source,
            test_used_for_selection=False,
        )
        rows.append(row)
    g_row = next(row for row in rows if row["condition"] == "G256_FIXED")
    r_row = next(row for row in rows if row["condition"] == "R256_FIXED")
    for row in rows:
        row["R256_minus_G256_seen_accuracy"] = r_row["seen_class_accuracy"] - g_row["seen_class_accuracy"]
        row["R256_minus_G256_old_accuracy"] = r_row["old_accuracy"] - g_row["old_accuracy"]
        row["R256_minus_G256_new_accuracy"] = r_row["new_accuracy"] - g_row["new_accuracy"]
    write_csv(output / "g256_r256_final_task.csv", rows)
    after = _edge_fingerprint(checkpoint_path)
    if before != after:
        raise RuntimeError(f"{protocol}: checkpoint changed during phase 3")
    result = {
        "status": "complete",
        "protocol": protocol,
        "task": task,
        "conditions": ["G256_FIXED", "R256_FIXED"],
        "same_pre_ca_head": True,
        "same_class_exposure": True,
        "same_optimizer_steps": True,
        "same_label_sequence": True,
        "same_current_new_bank": True,
        "test_used_for_selection": False,
        "checkpoint_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "phase3_manifest.json", result)
    del checkpoint, oracle_covariances, stored_covariances, gaussian_features, actual_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_revalidation(options: RevalidationOptions) -> dict[str, Any]:
    started = time.time()
    options.output_root.mkdir(parents=True, exist_ok=True)
    audit = write_extension_audit(options)
    selected = tuple(protocol.upper() for protocol in options.protocols)
    invalid = [protocol for protocol in selected if protocol not in TEN_PROTOCOLS]
    if invalid:
        raise ValueError(f"main transition execution only accepts {TEN_PROTOCOLS}, got {invalid}")
    missing_selected = [
        path for path in audit["missing"]
        if any(f"/{protocol}/" in path for protocol in selected)
    ]
    if missing_selected:
        raise FileNotFoundError("MISSING CHECKPOINTS:\n" + "\n".join(missing_selected))
    # Reuse and normalize the 5-task and 10-task scientific results before new GPU work.
    for protocol in ALL_PROTOCOLS:
        _copy_existing_scientific(protocol, options)
        _write_prototype_analysis(protocol, options)
    phase_results: dict[str, Any] = {}
    for protocol in selected:
        print(f"[{protocol}] sequential revalidation start; phases={options.phases}", flush=True)
        phase_results[protocol] = {}
        if 1 in options.phases:
            phase_results[protocol]["phase1"] = _run_phase1_protocol(protocol, options)
        if 2 in options.phases:
            phase_results[protocol]["phase2"] = _run_phase2_protocol(protocol, options)
        if 3 in options.phases:
            phase_results[protocol]["phase3"] = _run_phase3_protocol(protocol, options)
        print(f"[{protocol}] selected phases complete", flush=True)
    # Phase 2 final-task coverage for 5-task baselines is needed for direct final comparison.
    if 2 in options.phases:
        for protocol in ("IA5", "CAR5"):
            phase_results.setdefault(protocol, {})["phase2"] = _run_phase2_protocol(protocol, options)
    from .report import generate_all_reports

    reports = generate_all_reports(options.output_root, options.seed)
    manifest = {
        "status": "complete",
        "seed": options.seed,
        "physical_gpu": options.physical_gpu,
        "process_device": str(options.device),
        "protocols_executed": list(selected),
        "five_task_protocols_reused": ["IA5", "CAR5"],
        "phases": list(options.phases),
        "phase_results": phase_results,
        "reports": reports,
        "single_seed_limitation": True,
        "task_size_confound_acknowledged": True,
        "causal_accumulation_not_assumed": True,
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_root / "run_manifest.json", manifest)
    return manifest
