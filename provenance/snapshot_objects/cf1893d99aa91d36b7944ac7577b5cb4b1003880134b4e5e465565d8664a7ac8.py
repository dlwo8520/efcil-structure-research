"""Compact diagnostic exports for completed or partially completed runs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.experiment_io import read_csv, write_csv, write_json


def _number(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def export_run_diagnostics(run_dir: Path, config: dict):
    """Create spec-facing CSV tables; Parquet is optional and not faked.

    The current ``cds`` environment has no pyarrow/fastparquet. CSV is used as
    an honest lossless fallback and the storage manifest records that fact.
    """
    run_dir = Path(run_dir)
    write_json(
        run_dir / "diagnostic_storage.json",
        {
            "requested_preference": "parquet_for_large_tables",
            "actual_format": "csv",
            "reason": "no parquet engine is required by or installed in the pinned cds environment",
            "note": "CSV files have the same base names and documented schemas; no fake .parquet files are created.",
        },
    )
    _export_loss_tables(run_dir, config)
    _export_method_tables(run_dir, config)
    _export_resource(run_dir, config)
    _export_classifier(run_dir, config)
    _export_gradient(run_dir, config)


def _export_loss_tables(run_dir, config):
    rows = read_csv(run_dir / "epoch_metrics.csv")
    output, numerical = [], []
    method = config["method"]
    for row in rows:
        if row.get("stage") != "feature_training":
            continue
        sep_raw = _number(row.get("separation_raw"))
        sep_weighted = _number(row.get("separation_weighted"))
        output.append(
            {
                "protocol": config["protocol_id"],
                "run_id": config["experiment_id"],
                "method": method,
                "seed": config["seed"],
                "task": row.get("task_id"),
                "epoch": row.get("epoch"),
                "cls_loss_raw": row.get("classification_raw"),
                "scc_loss_raw": row.get("scc_raw"),
                "scc_loss_weighted": row.get("scc_weighted"),
                "sep_loss_raw": sep_raw if method in {"C0", "A", "B"} else None,
                "sep_loss_weighted": sep_weighted if method in {"C0", "A", "B"} else None,
                "pclad_loss_raw": sep_raw if method == "C" else None,
                "pclad_loss_weighted": sep_weighted if method == "C" else None,
                "total_loss": row.get("total_loss"),
                "zero_loss_ratio": None,
            }
        )
        numerical.append(
            {
                "protocol": config["protocol_id"],
                "run_id": config["experiment_id"],
                "seed": config["seed"],
                "task": row.get("task_id"),
                "epoch": row.get("epoch"),
                "nan_or_inf_detected": row.get("nan_or_inf_detected"),
                "feature_norm_min": row.get("feature_norm_min"),
                "feature_norm_max": row.get("feature_norm_max"),
                "tau_min": row.get("nearest_tau_min"),
                "tau_max": row.get("nearest_tau_max"),
                "d2_min": row.get("nearest_d2_min"),
                "d2_max": row.get("nearest_d2_max"),
                "rho_min": row.get("nearest_rho_min"),
                "rho_max": row.get("nearest_rho_max"),
            }
        )
    write_csv(run_dir / "loss_diagnostics.csv", output)
    write_csv(run_dir / "numerical_diagnostics.csv", numerical)


def _export_method_tables(run_dir, config):
    raw = read_csv(run_dir / "selective_pair_samples.csv")
    method = config["method"]
    if method == "A":
        groups = defaultdict(list)
        for row in raw:
            if row.get("selected") == "True":
                groups[(row["task"], row["epoch"], row["new_class_id"])].append(row)
        rows = []
        for (task, epoch, new_class), values in sorted(groups.items()):
            similarity = np.asarray([float(row["similarity"]) for row in values])
            d2 = np.asarray([float(row["D2"]) for row in values])
            loss = np.asarray([float(row["relation_pair_loss"]) for row in values])
            rows.append(
                {
                    "protocol": config["protocol_id"], "task": task, "epoch": epoch,
                    "new_class_id": new_class,
                    "selected_old_count": int(values[0]["selected_old_count"]),
                    "mean_selected_similarity": float(similarity.mean()),
                    "max_selected_similarity": float(similarity.max()),
                    "min_selected_similarity": float(similarity.min()),
                    "active_sep_ratio": float(np.mean([row["intrusion"] == "True" for row in values])),
                    "mean_D2": float(d2.mean()), "p10_D2": float(np.quantile(d2, 0.10)),
                    "p50_D2": float(np.quantile(d2, 0.50)), "p90_D2": float(np.quantile(d2, 0.90)),
                    "mean_tau_drift": float(np.mean([float(row["tau_drift"]) for row in values])),
                    "mean_sep_loss": float(loss.mean()),
                    "weighted_sep_loss": float(loss.mean()) * float(config["sep_lambda"]),
                }
            )
        write_csv(run_dir / "selective_am_diagnostics.csv", rows)
    elif method == "B":
        rows = []
        for row in raw:
            if row.get("selected") != "True":
                continue
            rows.append(
                {
                    "protocol": config["protocol_id"], "task": row["task"], "epoch": row["epoch"],
                    "new_class_id": row["new_class_id"], "old_class_id": row["old_class_id"],
                    "similarity": row["similarity"], "cutoff_beta": row["cutoff_beta"],
                    "r_similarity_excess": row["r_similarity_excess"], "alpha": config["alpha"],
                    "tau_drift": row["tau_drift"], "tau_enlarged": row["tau_effective"],
                    "tau_scale": row["tau_scale"], "D2": row["D2"], "rho": row["rho"],
                    "intrusion": row["intrusion"], "sep_loss": row["relation_pair_loss"],
                }
            )
        write_csv(run_dir / "enlarged_boundary_diagnostics.csv", rows)
    elif method == "C":
        rows = []
        for row in raw:
            loss = float(row["relation_pair_loss"])
            rows.append(
                {
                    "protocol": config["protocol_id"], "task": row["task"], "epoch": row["epoch"],
                    "new_class_id": row["new_class_id"], "old_class_id": row["old_class_id"],
                    "similarity": row["similarity"], "selected": row["selected"],
                    "cosine_new_old": row["pair_cosine"], "proto_clad_raw_loss": loss,
                    "lambda_clad": config["lambda_clad"],
                    "proto_clad_weighted_loss": loss * float(config["lambda_clad"]),
                }
            )
        write_csv(run_dir / "proto_clad_diagnostics.csv", rows)
        geometry = {
            (row.get("task"), row.get("epoch")): row
            for row in read_csv(run_dir / "geometry_diagnostics.csv")
        }
        gradients = {
            (row.get("task"), row.get("epoch"), row.get("batch")): row
            for row in read_csv(run_dir / "gradient_diagnostics.csv")
        }
        groups = defaultdict(list)
        for row in raw:
            groups[(row["task"], row["epoch"], row["batch"])].append(row)
        summaries = []
        for (task, epoch, batch), values in sorted(groups.items()):
            selected = [float(row["pair_cosine"]) for row in values if row["selected"] == "True"]
            unselected = [float(row["pair_cosine"]) for row in values if row["selected"] != "True"]
            selected_values = np.asarray(selected, dtype=np.float64)
            grad = gradients.get((task, epoch, batch), {})
            geom = geometry.get((task, epoch), {})
            summaries.append(
                {
                    "protocol": config["protocol_id"], "task": task, "epoch": epoch, "batch": batch,
                    "mean_selected_cosine": float(selected_values.mean()),
                    "p10_selected_cosine": float(np.quantile(selected_values, 0.10)),
                    "p50_selected_cosine": float(np.quantile(selected_values, 0.50)),
                    "p90_selected_cosine": float(np.quantile(selected_values, 0.90)),
                    "mean_unselected_cosine": float(np.mean(unselected)) if unselected else None,
                    "correct_new_cosine": geom.get("correct_new_cosine"),
                    "cosine_gap": geom.get("cosine_gap"),
                    "grad_norm_cls": grad.get("grad_norm_cls"),
                    "grad_norm_scc": grad.get("grad_norm_scc"),
                    "grad_norm_pclad": grad.get("grad_norm_abs"),
                    "cos_grad_cls_pclad": grad.get("cos_cls_abs"),
                    "cos_grad_scc_pclad": grad.get("grad_cosine_scc_abs"),
                }
            )
        write_csv(run_dir / "proto_clad_batch_summary.csv", summaries)


def _export_resource(run_dir, config):
    rows = []
    for row in read_csv(run_dir / "task_metrics.csv"):
        rows.append(
            {
                "protocol": config["protocol_id"], "run_id": config["experiment_id"],
                "method": config["method"], "seed": config["seed"], "task": row.get("task_id"),
                "training_seconds": row.get("training_seconds"),
                "peak_cuda_allocated_bytes": row.get("peak_cuda_allocated_bytes"),
                "peak_cuda_reserved_bytes": row.get("peak_cuda_reserved_bytes"),
                "peak_main_process_rss_bytes": row.get("peak_main_process_rss_bytes"),
                "statistics_and_cache_bytes": row.get("statistics_and_cache_bytes"),
            }
        )
    write_csv(run_dir / "resource_diagnostics.csv", rows)


def _export_classifier(run_dir, config):
    rows = []
    for row in read_csv(run_dir / "epoch_metrics.csv"):
        if row.get("stage") == "feature_training":
            continue
        rows.append({"protocol": config["protocol_id"], "run_id": config["experiment_id"], **row})
    write_csv(run_dir / "classifier_diagnostics.csv", rows)


def _export_gradient(run_dir, config):
    rows = []
    for row in read_csv(run_dir / "gradient_diagnostics.csv"):
        rows.append(
            {
                "protocol": config["protocol_id"], "run_id": config["experiment_id"],
                "method": config["method"], "seed": config["seed"],
                "task": row.get("task"), "epoch": row.get("epoch"), "batch": row.get("batch"),
                "grad_norm_cls": row.get("grad_norm_cls"), "grad_norm_scc": row.get("grad_norm_scc"),
                "grad_norm_sep": row.get("grad_norm_abs") if config["method"] in {"C0", "A", "B"} else None,
                "grad_norm_pclad": row.get("grad_norm_abs") if config["method"] == "C" else None,
                "cos_grad_cls_sep": row.get("cos_cls_abs") if config["method"] in {"C0", "A", "B"} else None,
                "cos_grad_cls_pclad": row.get("cos_cls_abs") if config["method"] == "C" else None,
            }
        )
    write_csv(run_dir / "selective_gradient_diagnostics.csv", rows)
