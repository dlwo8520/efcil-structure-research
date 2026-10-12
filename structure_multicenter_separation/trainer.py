from __future__ import annotations

import copy
import hashlib
import json
import logging
import random
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from data.data_manager import DataManager
from utils import model_factory
from utils.distribution_regions import region_storage_bytes
from utils.experiment_io import (
    capture_rng_state,
    read_csv,
    restore_rng_state,
    update_json,
    upsert_csv_rows,
    utc_now_iso,
    write_csv,
    write_json,
)


def CDS_train(args):
    seeds = copy.deepcopy(args["seed"])
    devices = copy.deepcopy(args["device"])
    for seed in seeds:
        run_args = copy.deepcopy(args)
        run_args["seed"] = int(seed)
        run_args["device"] = devices
        try:
            _train(run_args)
        except BaseException as error:
            run_dir = (
                Path(run_args["output_root"])
                / run_args["dataset"]
                / run_args["experiment_id"]
                / f"seed_{run_args['seed']}"
            ).resolve()
            update_json(
                run_dir / "run_metadata.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "ended_at_utc": utc_now_iso(),
                    "updated_at_utc": utc_now_iso(),
                    "failure_type": type(error).__name__,
                    "failure_reason": str(error),
                    "failure_traceback": traceback.format_exc(),
                },
            )
            raise


def _git_metadata():
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True
        ).splitlines()
        digest = hashlib.sha256(diff)
        for relative in sorted(untracked):
            path = root / relative
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            if path.is_file():
                digest.update(path.read_bytes())
        return {
            "commit": commit,
            "dirty_paths": dirty,
            "untracked_paths": untracked,
            "working_tree_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "dirty_paths": [],
            "untracked_paths": [],
            "working_tree_sha256": None,
        }


def _configure_logging(run_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[logging.FileHandler(run_dir / "training.log"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _train(args):
    run_dir = (
        Path(args["output_root"]) / args["dataset"] / args["experiment_id"] / f"seed_{args['seed']}"
    ).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    args["run_dir"] = str(run_dir)
    metadata_path = run_dir / "run_metadata.json"
    existing_metadata = _read_json(metadata_path, {})
    started_at_utc = existing_metadata.get("started_at_utc", utc_now_iso())
    recording_gaps = list(existing_metadata.get("recording_gaps", []))
    if (run_dir / "task_metrics.json").is_file() and not (run_dir / "task_summary.csv").is_file():
        recording_gaps.append(
            "legacy completed-task metrics predate task_summary/task_accuracy/class_accuracy correct-count CSVs"
        )
    if (run_dir / "training_diagnostics.jsonl").is_file() and not (run_dir / "epoch_metrics.csv").is_file():
        recording_gaps.append(
            "legacy epoch diagnostics predate sample-weighted component-loss epoch_metrics.csv"
        )
    recording_gaps = list(dict.fromkeys(recording_gaps))
    args["recording_gaps"] = recording_gaps
    _configure_logging(run_dir)
    _set_random(int(args["seed"]))
    _set_device(args)
    _reset_peak_cuda_stats(args["device"])
    for key, value in args.items():
        logging.info("%s: %s", key, value)
    write_json(run_dir / "resolved_config.json", args)
    source_metadata = _git_metadata()
    write_json(run_dir / "source_state.json", source_metadata)
    source_history = list(existing_metadata.get("source_history", []))
    if (
        not source_history
        or source_history[-1].get("working_tree_sha256")
        != source_metadata.get("working_tree_sha256")
    ):
        source_history.append(
            {
                **source_metadata,
                "observed_at_utc": utc_now_iso(),
                "resume_checkpoint": args.get("resume"),
            }
        )

    data_manager = DataManager(
        args["dataset"], args["shuffle"], args["seed"], args["init_cls"], args["increment"], args=args
    )
    write_json(run_dir / "protocol_manifest.json", data_manager.protocol_manifest)
    write_json(
        run_dir / "class_order.json",
        {
            "class_order_original_ids": data_manager.protocol_manifest["class_order_original_ids"],
            "tasks": data_manager.protocol_manifest["tasks"],
        },
    )
    update_json(
        metadata_path,
        {
            "recording_schema_version": 2,
            "run_id": f"{args['experiment_id']}:seed_{args['seed']}",
            "experiment_id": args["experiment_id"],
            "seed": args["seed"],
            "status": "running",
            "started_at_utc": started_at_utc,
            "updated_at_utc": utc_now_iso(),
            "ended_at_utc": None,
            "resume_checkpoint": args.get("resume"),
            "config": args,
            "source": source_metadata,
            "source_history": source_history,
            "protocol": data_manager.protocol_manifest,
            "failure_type": None,
            "failure_reason": None,
            "failure_traceback": None,
            "recording_gaps": recording_gaps,
        },
    )

    model = model_factory.get_model(args["model_name"], args)
    if args.get("resume"):
        model.load_experiment_checkpoint(args["resume"])
    metrics_path = run_dir / "task_metrics.json"
    matrix_path = run_dir / "task_accuracy_matrix.json"
    task_metrics = _read_json(metrics_path, [])
    accuracy_matrix = _read_json(matrix_path, [])
    if model._resume_stage == "task_end":
        start_task = model._cur_task + 1
    elif model._resume_stage == "train_epoch":
        start_task = model._cur_task
    else:
        start_task = 0

    for task in range(start_task, data_manager.nb_tasks):
        known_before = model._known_classes
        _reset_peak_cuda_stats(args["device"])
        started = time.perf_counter()
        model.incremental_train(data_manager)
        training_seconds = time.perf_counter() - started
        evaluation_rng_state = capture_rng_state()
        was_training = model._network.training
        try:
            y_pred, y_true = model._eval_cnn(model.test_loader)
        finally:
            restore_rng_state(evaluation_rng_state)
            model._network.train(was_training)
        predictions = y_pred[:, 0]
        num_samples = int(y_true.shape[0])
        num_correct = int(np.sum(predictions == y_true))
        top1_accuracy = num_correct * 100.0 / num_samples
        top5_correct = int(np.sum(np.any(y_pred == y_true[:, None], axis=1)))
        top5_accuracy = top5_correct * 100.0 / num_samples
        eval_stage = "task_final"

        task_row = []
        task_accuracy_rows = []
        class_start = 0
        for learned_task in range(task + 1):
            class_stop = class_start + data_manager.get_task_size(learned_task)
            task_mask = np.logical_and(y_true >= class_start, y_true < class_stop)
            task_num_samples = int(np.sum(task_mask))
            task_num_correct = int(np.sum(predictions[task_mask] == y_true[task_mask]))
            task_accuracy = task_num_correct * 100.0 / task_num_samples
            task_row.append(task_accuracy)
            task_accuracy_rows.append(
                {
                    "experiment_id": args["experiment_id"],
                    "seed": args["seed"],
                    "trained_task": task + 1,
                    "evaluated_task": learned_task + 1,
                    "eval_stage": eval_stage,
                    "num_correct": task_num_correct,
                    "num_samples": task_num_samples,
                    "accuracy": task_accuracy,
                }
            )
            class_start = class_stop
        accuracy_matrix = accuracy_matrix[:task]
        accuracy_matrix.append(task_row)

        upsert_csv_rows(
            run_dir / "task_accuracy.csv",
            task_accuracy_rows,
            ("experiment_id", "seed", "trained_task", "evaluated_task", "eval_stage"),
        )
        derived_matrix = _accuracy_matrix_from_long(
            read_csv(run_dir / "task_accuracy.csv"),
            args["experiment_id"],
            args["seed"],
            task + 1,
            eval_stage,
        )
        if derived_matrix is not None:
            accuracy_matrix = derived_matrix
        class_accuracy_rows = []
        for class_id in range(model._total_classes):
            class_mask = y_true == class_id
            class_num_samples = int(np.sum(class_mask))
            class_num_correct = int(np.sum(predictions[class_mask] == y_true[class_mask]))
            class_metadata = data_manager.get_class_metadata(class_id)
            class_accuracy_rows.append(
                {
                    "experiment_id": args["experiment_id"],
                    "seed": args["seed"],
                    "trained_task": task + 1,
                    "eval_stage": eval_stage,
                    **class_metadata,
                    "num_correct": class_num_correct,
                    "num_samples": class_num_samples,
                    "accuracy": class_num_correct * 100.0 / class_num_samples,
                }
            )
        upsert_csv_rows(
            run_dir / "class_accuracy.csv",
            class_accuracy_rows,
            ("experiment_id", "seed", "trained_task", "eval_stage", "mapped_class_id"),
        )

        old_mask = y_true < known_before
        new_mask = ~old_mask
        old_num_samples = int(np.sum(old_mask))
        new_num_samples = int(np.sum(new_mask))
        old_num_correct = int(np.sum(predictions[old_mask] == y_true[old_mask])) if old_num_samples else None
        new_num_correct = int(np.sum(predictions[new_mask] == y_true[new_mask]))
        old_accuracy = old_num_correct * 100.0 / old_num_samples if old_num_samples else None
        new_accuracy = new_num_correct * 100.0 / new_num_samples
        old_to_new = float(np.mean(predictions[old_mask] >= known_before)) if old_mask.any() else None
        new_to_old = (
            float(np.mean(predictions[new_mask] < known_before)) if new_mask.any() and known_before else 0.0
        )
        peak_memory = _peak_cuda_memory(args["device"])
        class_means = getattr(model, "_class_means", None)
        class_covariances = getattr(model, "_class_covs", None)
        class_statistics_bytes = 0
        if class_means is not None:
            class_statistics_bytes += int(class_means.nbytes)
        if class_covariances is not None:
            class_statistics_bytes += int(class_covariances.numel() * class_covariances.element_size())
        region_statistics_bytes = region_storage_bytes(
            getattr(model, "_local_stats", {}) or getattr(model, "_single_regions", {})
        )
        separation_cache = getattr(model, "_separation_cache", None)
        separation_cache_bytes = separation_cache.storage_bytes if separation_cache is not None else 0
        record = {
            "task": task,
            "task_id": task + 1,
            "eval_stage": eval_stage,
            "seen_classes": model._total_classes,
            "num_correct": num_correct,
            "num_samples": num_samples,
            "A_t": top1_accuracy,
            "top5_num_correct": top5_correct,
            "top5": top5_accuracy,
            "old_num_correct": old_num_correct,
            "old_num_samples": old_num_samples,
            "old_accuracy": old_accuracy,
            "new_num_correct": new_num_correct,
            "new_num_samples": new_num_samples,
            "new_accuracy": new_accuracy,
            "old_to_new_error_rate": old_to_new,
            "new_to_old_error_rate": new_to_old,
            "inc_accuracy_so_far": float(
                np.mean([row["A_t"] for row in task_metrics[:task]] + [top1_accuracy])
            ),
            "forgetting_so_far": _forgetting(accuracy_matrix),
            "training_seconds": training_seconds,
            "peak_cuda_allocated_bytes": peak_memory["allocated"],
            "peak_cuda_reserved_bytes": peak_memory["reserved"],
            "peak_main_process_rss_bytes": _peak_main_process_rss_bytes(),
            "class_statistics_bytes": class_statistics_bytes,
            "region_statistics_bytes": region_statistics_bytes,
            "separation_cache_bytes": separation_cache_bytes,
            "total_statistics_bytes": class_statistics_bytes + region_statistics_bytes,
            "statistics_and_cache_bytes": (
                class_statistics_bytes + region_statistics_bytes + separation_cache_bytes
            ),
            "status": "complete",
        }
        task_metrics = task_metrics[:task]
        task_metrics.append(record)
        write_json(metrics_path, task_metrics)
        write_csv(run_dir / "task_metrics.csv", task_metrics)
        write_json(matrix_path, accuracy_matrix)
        write_csv(
            run_dir / "task_accuracy_matrix.csv",
            [
                {
                    "trained_task": row_index + 1,
                    **{
                        f"eval_task_{column + 1}": (
                            row[column] if column < len(row) else None
                        )
                        for column in range(data_manager.nb_tasks)
                    },
                }
                for row_index, row in enumerate(accuracy_matrix)
            ],
        )
        upsert_csv_rows(
            run_dir / "task_summary.csv",
            [
                {
                    "experiment_id": args["experiment_id"],
                    "seed": args["seed"],
                    "task_id": task + 1,
                    "eval_stage": eval_stage,
                    "seen_classes": model._total_classes,
                    "num_correct": num_correct,
                    "num_samples": num_samples,
                    "accuracy": top1_accuracy,
                    "top5_num_correct": top5_correct,
                    "top5_accuracy": top5_accuracy,
                    "old_num_correct": old_num_correct,
                    "old_num_samples": old_num_samples,
                    "old_accuracy": old_accuracy,
                    "new_num_correct": new_num_correct,
                    "new_num_samples": new_num_samples,
                    "new_accuracy": new_accuracy,
                    "old_to_new_error_rate": old_to_new,
                    "new_to_old_error_rate": new_to_old,
                    "training_seconds": training_seconds,
                    "checkpoint_path": str(run_dir / "checkpoint_latest.pt"),
                    "checkpoint_state_id": f"task_end:task_{task + 1}",
                }
            ],
            ("experiment_id", "seed", "task_id", "eval_stage"),
        )
        logging.info("Task %d metrics: %s", task, record)
        model.after_task()
        checkpoint_state_id = model.save_experiment_checkpoint(
            run_dir / "checkpoint_latest.pt", stage="task_end"
        )
        summary = _build_summary(args, task_metrics, accuracy_matrix, data_manager.nb_tasks)
        write_json(run_dir / "summary.json", summary)
        update_json(
            metadata_path,
            {
                "status": summary["status"],
                "updated_at_utc": utc_now_iso(),
                "ended_at_utc": utc_now_iso() if summary["status"] == "complete" else None,
                "completed_tasks": len(task_metrics),
                "last_completed_task": task + 1,
                "last_checkpoint_path": str(run_dir / "checkpoint_latest.pt"),
                "last_checkpoint_state_id": checkpoint_state_id,
            },
        )

    if task_metrics:
        summary = _build_summary(args, task_metrics, accuracy_matrix, data_manager.nb_tasks)
        write_json(run_dir / "summary.json", summary)
        update_json(
            metadata_path,
            {
                "status": summary["status"],
                "updated_at_utc": utc_now_iso(),
                "ended_at_utc": utc_now_iso() if summary["status"] == "complete" else None,
                "completed_tasks": len(task_metrics),
            },
        )
        logging.info("Summary: %s", summary)


def _read_json(path, default):
    if not Path(path).is_file():
        return default
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_summary(args, task_metrics, accuracy_matrix, expected_tasks):
    complete = len(task_metrics) == expected_tasks
    return {
        "recording_schema_version": 2,
        "experiment_id": args["experiment_id"],
        "seed": args["seed"],
        "multi_center": bool(args["multi_center"]),
        "separation_covariance": (
            args["separation_covariance"] if args["old_new_separation"] else "none"
        ),
        "completed_tasks": len(task_metrics),
        "expected_tasks": expected_tasks,
        "A_by_task": [float(row["A_t"]) for row in task_metrics],
        "Inc-Acc": float(np.mean([row["A_t"] for row in task_metrics])),
        "Last-Acc": float(task_metrics[-1]["A_t"]),
        "Forgetting": _forgetting(accuracy_matrix) if complete else None,
        "forgetting_definition": (
            "mean_i(max_{l=i..T-1} a[l,i] - a[T,i]) over tasks i before final task; "
            "a[l,i] uses the single head over all classes seen at stage l"
        ),
        "accuracy_unit": "percent_0_to_100_unrounded",
        "recording_status": "partial_legacy" if args.get("recording_gaps") else "complete",
        "recording_gaps": args.get("recording_gaps", []),
        "total_training_seconds": float(sum(row["training_seconds"] for row in task_metrics)),
        "max_peak_cuda_allocated_bytes": _max_not_none(
            row.get("peak_cuda_allocated_bytes") for row in task_metrics
        ),
        "max_peak_cuda_reserved_bytes": _max_not_none(
            row.get("peak_cuda_reserved_bytes") for row in task_metrics
        ),
        "max_peak_main_process_rss_bytes": _max_not_none(
            row.get("peak_main_process_rss_bytes") for row in task_metrics
        ),
        "final_total_statistics_bytes": task_metrics[-1].get("total_statistics_bytes"),
        "max_separation_cache_bytes": _max_not_none(
            row.get("separation_cache_bytes") for row in task_metrics
        ),
        "status": "complete" if complete else "partial",
    }


def _accuracy_matrix_from_long(rows, experiment_id, seed, completed_tasks, eval_stage):
    values = {
        (int(row["trained_task"]), int(row["evaluated_task"])): float(row["accuracy"])
        for row in rows
        if row.get("experiment_id") == str(experiment_id)
        and row.get("seed") == str(seed)
        and row.get("eval_stage") == str(eval_stage)
    }
    matrix = []
    for trained_task in range(1, completed_tasks + 1):
        row = []
        for evaluated_task in range(1, trained_task + 1):
            key = (trained_task, evaluated_task)
            if key not in values:
                return None
            row.append(values[key])
        matrix.append(row)
    return matrix


def _forgetting(matrix):
    if len(matrix) < 2:
        return None
    final_task = len(matrix) - 1
    values = []
    for evaluated_task in range(final_task):
        earlier = [matrix[trained][evaluated_task] for trained in range(evaluated_task, final_task)]
        values.append(max(earlier) - matrix[final_task][evaluated_task])
    return float(np.mean(values)) if values else None


def _max_not_none(values):
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _peak_main_process_rss_bytes():
    # Linux reports ru_maxrss in KiB. This intentionally excludes DataLoader
    # worker-process peaks and is named accordingly in the machine-readable log.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _set_device(args):
    devices = []
    for device in args["device"]:
        if int(device) == -1:
            devices.append(torch.device("cpu"))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU execution requested but CUDA is unavailable")
            devices.append(torch.device(f"cuda:{int(device)}"))
    args["device"] = devices


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _cuda_index(device):
    return device.index if isinstance(device, torch.device) and device.type == "cuda" else None


def _reset_peak_cuda_stats(devices):
    if not torch.cuda.is_available():
        return
    cuda_indices = []
    for device in devices:
        index = _cuda_index(device)
        if index is not None:
            # PyTorch 2.4 raises ``Invalid device argument`` when peak stats
            # are reset before that logical CUDA device has a context.
            torch.cuda.set_device(index)
            torch.cuda.reset_peak_memory_stats(index)
            cuda_indices.append(index)
    if cuda_indices:
        torch.cuda.set_device(cuda_indices[0])


def _peak_cuda_memory(devices):
    allocated = 0
    reserved = 0
    if torch.cuda.is_available():
        for device in devices:
            index = _cuda_index(device)
            if index is not None:
                torch.cuda.synchronize(index)
                allocated = max(allocated, torch.cuda.max_memory_allocated(index))
                reserved = max(reserved, torch.cuda.max_memory_reserved(index))
    return {"allocated": allocated or None, "reserved": reserved or None}
