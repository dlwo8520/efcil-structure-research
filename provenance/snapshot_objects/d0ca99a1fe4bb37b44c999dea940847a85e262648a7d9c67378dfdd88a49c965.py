"""RP-specific post-processing built from diagnostics already collected in training."""

from __future__ import annotations

from pathlib import Path

from utils.experiment_io import read_csv, write_csv


def export_rp_run_diagnostics(run_dir: Path, config: dict) -> None:
    run_dir = Path(run_dir)
    gradient_rows = []
    for row in read_csv(run_dir / "gradient_metrics.csv"):
        gradient_rows.append(
            {
                "protocol": config["protocol_id"],
                "run_id": config["experiment_id"],
                "seed": config["seed"],
                "task": row.get("task_id"),
                "epoch": row.get("epoch"),
                "batch": row.get("batch"),
                "grad_norm_cls": row.get("grad_classification_norm"),
                "grad_norm_scc": row.get("grad_scc_norm"),
                "grad_norm_rp_clad": row.get("grad_abs_norm"),
                "cos_grad_cls_rp_clad": row.get("grad_cosine_classification_abs"),
                "cos_grad_scc_rp_clad": row.get("grad_cosine_scc_abs"),
            }
        )
    write_csv(run_dir / "rp_gradient_diagnostics.csv", gradient_rows)

    loss_rows = []
    for row in read_csv(run_dir / "epoch_metrics.csv"):
        if row.get("stage") != "feature_training":
            continue
        loss_rows.append(
            {
                "protocol": config["protocol_id"],
                "run_id": config["experiment_id"],
                "seed": config["seed"],
                "task": row.get("task_id"),
                "epoch": row.get("epoch"),
                "classification_loss": row.get("classification_raw"),
                "scc_loss": row.get("scc_raw"),
                "rp_proto_clad_raw_loss": row.get("separation_raw"),
                "rp_proto_clad_weighted_loss": row.get("separation_weighted"),
                "lambda_clad": config["lambda_clad"],
                "total_loss": row.get("total_loss"),
            }
        )
    write_csv(run_dir / "rp_loss_diagnostics.csv", loss_rows)

