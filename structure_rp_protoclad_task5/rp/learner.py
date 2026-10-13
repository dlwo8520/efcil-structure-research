"""SelectiveSimilarityLearner with Proto-CLAD geometry replaced by frozen RP."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from selective.learner import SelectiveSimilarityLearner
from selective.losses import top_p_selection
from utils.experiment_io import (
    capture_rng_state,
    upsert_csv_rows,
    utc_now_iso,
    write_json,
)

from .diagnostics import export_rp_run_diagnostics
from .losses import rp_proto_clad_relation
from .projection import make_projection, project_relu, projection_metadata
from .snapshot import create_rp_snapshot


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.size < 2 or right.size != left.size or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _summary(values: torch.Tensor, prefix: str) -> dict:
    values = torch.as_tensor(values).detach().cpu().float().flatten()
    if values.numel() == 0:
        return {f"{prefix}_{name}": None for name in ("mean", "std", "p50", "p90", "p95", "max")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(unbiased=False)),
        f"{prefix}_p50": float(torch.quantile(values, 0.50)),
        f"{prefix}_p90": float(torch.quantile(values, 0.90)),
        f"{prefix}_p95": float(torch.quantile(values, 0.95)),
        f"{prefix}_max": float(values.max()),
    }


class RPProtoCLADLearner(SelectiveSimilarityLearner):
    """RP changes only the cosine used by Proto-CLAD's auxiliary loss."""

    RP_CHECKPOINT_KEYS = (
        "rp_enabled",
        "rp_dim",
        "rp_seed",
        "rp_input_dim",
        "rp_activation",
        "rp_bias",
        "rp_trainable",
        "rp_matrix_shape",
        "rp_matrix_checksum",
    )

    def __init__(self, args):
        self._capture_rp_features = False
        self._captured_rp_features = None
        self._captured_rp_labels = None
        self._rp_logged_tasks = set()
        super().__init__(args)
        if str(args.get("selective_method")) != "rp_proto_clad":
            raise ValueError("RPProtoCLADLearner requires selective_method=rp_proto_clad")
        feature_dim = int(self.feature_dim)
        if feature_dim != int(args["rp_input_dim"]):
            raise ValueError(
                f"configured RP input dim {args['rp_input_dim']} != network feature dim {feature_dim}"
            )
        self._rp_matrix_cpu = make_projection(feature_dim, args["rp_dim"], args["rp_seed"])
        metadata = projection_metadata(self._rp_matrix_cpu, args["rp_seed"])
        if metadata["shape"] != list(args["rp_matrix_shape"]):
            raise ValueError("RP matrix shape differs from the validated launch config")
        if metadata["checksum_sha256"] != args["rp_matrix_checksum"]:
            raise ValueError("RP matrix checksum differs from the validated launch config")
        if self._rp_matrix_cpu.requires_grad:
            raise AssertionError("RP matrix unexpectedly requires gradients")
        self._rp_device_cache: dict[str, torch.Tensor] = {}
        write_json(self.run_dir / "rp_projection.json", metadata)
        create_rp_snapshot(self.run_dir, args["rp_extension_root"])

    @property
    def rp_matrix(self) -> torch.Tensor:
        return self._rp_matrix_cpu

    def _rp_on(self, device) -> torch.Tensor:
        key = str(torch.device(device))
        if key not in self._rp_device_cache:
            value = self._rp_matrix_cpu.to(device=device, dtype=torch.float32)
            value.requires_grad_(False)
            self._rp_device_cache[key] = value
        return self._rp_device_cache[key]

    def save_experiment_checkpoint(self, path, stage, next_epoch=0, optimizer=None, scheduler=None):
        """Preserve the parent checkpoint schema and add explicit RP metadata."""
        state_id = (
            f"train_epoch:task_{self._cur_task + 1}:next_epoch_{int(next_epoch)}"
            if stage == "train_epoch"
            else f"{stage}:task_{self._cur_task + 1}"
        )
        payload = {
            "format_version": 1,
            "state_id": state_id,
            "stage": stage,
            "next_epoch": int(next_epoch),
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "task_sizes": list(self.task_sizes),
            "network_state_dict": self._checkpoint_network().state_dict(),
            "class_means": getattr(self, "_class_means", None),
            "class_covs": getattr(self, "_class_covs", None),
            "radius": getattr(self, "radius", None),
            "local_stats": self._local_stats,
            "single_regions": self._single_regions,
            "task_source_means": self._task_source_means,
            "task_source_covs": self._task_source_covs,
            "task_source_local_stats": self._task_source_local_stats,
            "sep_reference_means": self._sep_reference_means,
            "sep_reference_covs": self._sep_reference_covs,
            "now_task_mean": getattr(self, "now_task_mean", None),
            "now_task_covs": getattr(self, "now_task_covs", None),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": capture_rng_state(),
            "config": self.args,
            "rp_projection": dict(self.args["rp_projection"]),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        write_json(
            self.run_dir / "checkpoint_manifest.json",
            {
                "checkpoint_path": str(target.resolve()),
                "checkpoint_state_id": state_id,
                "stage": stage,
                "task_id": self._cur_task + 1,
                "next_epoch": int(next_epoch),
                "updated_at_utc": utc_now_iso(),
                "rp_projection": self.args["rp_projection"],
                "note": "checkpoint_latest.pt is atomically replaced and includes frozen RP metadata",
            },
        )
        return state_id

    def load_experiment_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("config", {})
        mismatches = {
            key: {"checkpoint": stored.get(key), "current": self.args.get(key)}
            for key in self.RP_CHECKPOINT_KEYS
            if stored.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"RP checkpoint/config mismatch: {mismatches}")
        if checkpoint.get("rp_projection") != self.args.get("rp_projection"):
            raise ValueError("checkpoint top-level RP projection metadata mismatch")
        return super().load_experiment_checkpoint(path)

    def _collect_current_features(self, feature_key="sfeatures", return_sample_ids=False):
        result = super()._collect_current_features(feature_key, return_sample_ids)
        if self._capture_rp_features and feature_key == "sfeatures":
            features, labels = result[:2]
            self._captured_rp_features = torch.as_tensor(features).detach().cpu().float()
            self._captured_rp_labels = torch.as_tensor(labels).detach().cpu().long()
        return result

    def _refresh_followup_cache(self, epoch):
        should_capture = self._cur_task > 0 and self._cur_task not in self._rp_logged_tasks
        self._capture_rp_features = should_capture
        self._captured_rp_features = None
        self._captured_rp_labels = None
        try:
            super()._refresh_followup_cache(epoch)
        finally:
            self._capture_rp_features = False
        self._write_rp_pair_geometry(epoch)
        if should_capture:
            if self._captured_rp_features is None or self._captured_rp_labels is None:
                raise RuntimeError("failed to reuse the current-feature cache for RP diagnostics")
            self._write_rp_prototype_approximation(
                epoch, self._captured_rp_features, self._captured_rp_labels
            )
            self._rp_logged_tasks.add(self._cur_task)
            self._captured_rp_features = None
            self._captured_rp_labels = None

    def _relation_loss(self, features, targets):
        if str(self.args.get("selective_method")) != "rp_proto_clad":
            return super()._relation_loss(features, targets)
        return rp_proto_clad_relation(
            features=features,
            targets=targets,
            cache=self._separation_cache,
            apr_old_prototypes=self._followup_old_prototypes,
            current_prototypes=self._followup_current_prototypes,
            similarities=self._similarity_matrix,
            known_classes=self._known_classes,
            top_p=float(self.args["top_p"]),
            rp_matrix=self._rp_on(features.device),
        )

    def _write_rp_pair_geometry(self, epoch):
        if self._cur_task <= 0:
            return
        current = torch.as_tensor(self._followup_current_prototypes).cpu().float()
        old = torch.as_tensor(self._followup_old_prototypes).cpu().float()
        original = F.normalize(current, dim=1) @ F.normalize(old, dim=1).T
        current_rp = project_relu(current, self._rp_matrix_cpu)
        old_rp = project_relu(old, self._rp_matrix_cpu)
        projected = F.normalize(current_rp, dim=1) @ F.normalize(old_rp, dim=1).T
        selected, ranks, _, _ = top_p_selection(
            self._similarity_matrix, float(self.args["top_p"])
        )
        rows = []
        for new_offset in range(current.shape[0]):
            for old_class in range(old.shape[0]):
                if not bool(selected[new_offset, old_class]):
                    continue
                rank = int(ranks[new_offset, old_class])
                rows.append(
                    {
                        "protocol": self.args["protocol_id"],
                        "run_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "new_class": self._known_classes + new_offset,
                        "old_class": old_class,
                        "original_cosine": float(original[new_offset, old_class]),
                        "rp_cosine": float(projected[new_offset, old_class]),
                        "delta_cosine": float(
                            projected[new_offset, old_class] - original[new_offset, old_class]
                        ),
                        "selection_similarity_original": float(
                            self._similarity_matrix[new_offset, old_class]
                        ),
                        "similarity_rank": rank,
                        "similarity_percentile": 1.0 - (rank - 1) / self._known_classes,
                    }
                )
        upsert_csv_rows(
            self.run_dir / "rp_pair_geometry_diagnostics.csv",
            rows,
            ("protocol", "run_id", "seed", "task", "epoch", "new_class", "old_class"),
        )
        left = original.numpy().ravel()
        right = projected.numpy().ravel()
        upsert_csv_rows(
            self.run_dir / "rp_similarity_structure_diagnostics.csv",
            [
                {
                    "protocol": self.args["protocol_id"],
                    "run_id": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task": self._cur_task + 1,
                    "epoch": epoch + 1,
                    "pair_count": int(left.size),
                    "pearson_original_vs_rp": _pearson(left, right),
                    "spearman_original_vs_rp": _spearman(left, right),
                }
            ],
            ("protocol", "run_id", "seed", "task", "epoch"),
        )

    def _project_feature_chunks(self, features: torch.Tensor) -> torch.Tensor:
        device = self._device
        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        projected = []
        matrix = self._rp_on(device)
        with torch.no_grad():
            for chunk in features.split(256):
                projected.append(project_relu(chunk.to(device), matrix).cpu())
        return torch.cat(projected, dim=0)

    def _write_rp_prototype_approximation(self, epoch, features, labels):
        projected = self._project_feature_chunks(features)
        rows = []
        for class_id in sorted(int(value) for value in labels.unique().tolist()):
            mask = labels == class_id
            mean_original = features[mask].mean(dim=0, keepdim=True)
            project_after_mean = project_relu(mean_original, self._rp_matrix_cpu).squeeze(0)
            mean_after_project = projected[mask].mean(dim=0)
            difference = project_after_mean - mean_after_project
            rows.append(
                {
                    "protocol": self.args["protocol_id"],
                    "run_id": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task": self._cur_task + 1,
                    "epoch": epoch + 1,
                    "class_id": class_id,
                    "sample_count": int(mask.sum()),
                    "cos_a_b": float(
                        F.cosine_similarity(
                            project_after_mean[None, :], mean_after_project[None, :], dim=1
                        )[0]
                    ),
                    "l2_a_b": float(difference.norm()),
                    "relative_l2_a_b": float(
                        difference.norm() / mean_after_project.norm().clamp_min(1e-8)
                    ),
                }
            )
        upsert_csv_rows(
            self.run_dir / "rp_prototype_approximation.csv",
            rows,
            ("protocol", "run_id", "seed", "task", "class_id"),
        )
        original_norm = features.norm(dim=1)
        rp_norm = projected.norm(dim=1)
        zero_ratio = float(projected.eq(0).float().mean())
        feature_row = {
            "protocol": self.args["protocol_id"],
            "run_id": self.args["experiment_id"],
            "seed": self.args["seed"],
            "task": self._cur_task + 1,
            "epoch": epoch + 1,
            "sample_count": int(features.shape[0]),
            **_summary(original_norm, "original_norm"),
            **_summary(rp_norm, "rp_norm"),
            "rp_zero_ratio": zero_ratio,
            "rp_nonzero_ratio": 1.0 - zero_ratio,
        }
        upsert_csv_rows(
            self.run_dir / "rp_feature_diagnostics.csv",
            [feature_row],
            ("protocol", "run_id", "seed", "task"),
        )

    def _sample_relation_rows(self, relation, epoch, batch_index, sample_ids, targets):
        interval = max(1, int(self.args.get("diagnostic_pair_interval", 20)))
        if batch_index % interval:
            return [], []
        if self.args.get("diagnostic_pair_epochs", "last") == "last" and epoch + 1 != self.tuned_epochs:
            return [], []
        diag = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in relation.diagnostics.items()
        }
        rows, gap_rows = [], []
        old_count = self._known_classes
        for sample_index in range(len(targets)):
            for old_class in range(old_count):
                if not bool(diag["selected_mask"][sample_index, old_class]):
                    continue
                pair_slot = sample_index * old_count + old_class
                common = {
                    "protocol": self.args["protocol_id"],
                    "method": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task": self._cur_task + 1,
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "sample_id": int(sample_ids[sample_index]),
                    "sample_in_batch": pair_slot,
                    "actual_sample_in_batch": sample_index,
                    "new_class_id": int(targets[sample_index]),
                    "old_class_id": old_class,
                    "selected": True,
                    "selected_old_count": int(diag["selected_mask"][sample_index].sum()),
                    "similarity": float(diag["similarity"][sample_index, old_class]),
                    "similarity_rank": int(diag["similarity_rank"][sample_index, old_class]),
                    "cutoff_beta": float(diag["cutoff_beta"][sample_index]),
                    "D2": float(diag["all_d2"][sample_index, old_class]),
                    "tau_drift": float(self._separation_cache.taus.cpu()[old_class]),
                    "tau_effective": float(diag["all_pair_taus"][sample_index, old_class]),
                    "tau_scale": 1.0,
                    "r_similarity_excess": 0.0,
                    "rho": float(diag["all_rho"][sample_index, old_class]),
                    "intrusion": bool(diag["all_rho"][sample_index, old_class] < 1.0),
                    "relation_pair_loss": float(diag["pair_loss"][sample_index, old_class]),
                    "raw_hinge": float(diag["raw_hinge"][sample_index, old_class]),
                    "pair_cosine": float(diag["rp_cosine"][sample_index, old_class]),
                    "original_pair_cosine": float(
                        diag["original_cosine"][sample_index, old_class]
                    ),
                }
                rows.append(common)
                gap_rows.append(
                    {
                        **common,
                        "original_cosine": common["original_pair_cosine"],
                        "rp_cosine": common["pair_cosine"],
                        "delta_cosine": float(diag["delta_cosine"][sample_index, old_class]),
                        "correct_new_cosine_original": float(
                            diag["correct_new_cosine_original"][sample_index]
                        ),
                        "selected_old_cosine_original": common["original_pair_cosine"],
                        "gap_original": float(diag["gap_original"][sample_index, old_class]),
                        "correct_new_cosine_rp": float(
                            diag["correct_new_cosine_rp"][sample_index]
                        ),
                        "selected_old_cosine_rp": common["pair_cosine"],
                        "gap_rp": float(diag["gap_rp"][sample_index, old_class]),
                        "delta_gap": float(diag["delta_gap"][sample_index, old_class]),
                        "original_feature_norm": float(
                            diag["original_feature_norm"][sample_index]
                        ),
                        "rp_feature_norm": float(diag["rp_feature_norm"][sample_index]),
                        "rp_zero_ratio": float(diag["rp_sample_zero_ratio"][sample_index]),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "rp_gap_diagnostics.csv",
            gap_rows,
            ("protocol", "method", "seed", "task", "epoch", "batch", "sample_in_batch"),
        )
        return rows, []

    def _init_train(self, *args, **kwargs):
        super()._init_train(*args, **kwargs)
        export_rp_run_diagnostics(self.run_dir, self.args)

    def _write_evaluation_diagnostics(self, y_pred, y_true):
        super()._write_evaluation_diagnostics(y_pred, y_true)
        predictions = y_pred[:, 0].astype(int)
        truths = y_true.astype(int)
        task = self._cur_task + 1
        old_true = truths < self._known_classes
        old_pred = predictions < self._known_classes
        rows = [
            {
                "protocol": self.args["protocol_id"],
                "run_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task": task,
                "old_to_new_count": int(np.sum(old_true & ~old_pred)),
                "old_to_new_rate": float(np.mean(~old_pred[old_true])) if old_true.any() else None,
                "old_to_old_count": int(np.sum(old_true & old_pred)),
                "old_to_old_rate": float(np.mean(old_pred[old_true])) if old_true.any() else None,
                "new_to_old_count": int(np.sum(~old_true & old_pred)),
                "new_to_old_rate": float(np.mean(old_pred[~old_true])) if (~old_true).any() else None,
                "new_to_new_count": int(np.sum(~old_true & ~old_pred)),
                "new_to_new_rate": float(np.mean(~old_pred[~old_true])) if (~old_true).any() else None,
                "is_final_task": task == int(self.args["expected_tasks"]),
            }
        ]
        upsert_csv_rows(
            self.run_dir / "rp_confusion_diagnostics.csv",
            rows,
            ("protocol", "run_id", "seed", "task"),
        )
        if self._known_classes == 0:
            return
        selected, ranks, _, _ = top_p_selection(
            self._similarity_matrix, float(self.args["top_p"])
        )
        pair_rows = []
        for new_offset in range(selected.shape[0]):
            new_class = self._known_classes + new_offset
            new_total = int(np.sum(truths == new_class))
            for old_class in range(selected.shape[1]):
                if not bool(selected[new_offset, old_class]):
                    continue
                old_total = int(np.sum(truths == old_class))
                pair_rows.append(
                    {
                        "protocol": self.args["protocol_id"],
                        "run_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": task,
                        "new_class": new_class,
                        "old_class": old_class,
                        "similarity_rank": int(ranks[new_offset, old_class]),
                        "new_to_selected_old_count": int(
                            np.sum((truths == new_class) & (predictions == old_class))
                        ),
                        "new_to_selected_old_rate": (
                            float(np.sum((truths == new_class) & (predictions == old_class)))
                            / new_total if new_total else None
                        ),
                        "selected_old_to_new_count": int(
                            np.sum((truths == old_class) & (predictions == new_class))
                        ),
                        "selected_old_to_new_rate": (
                            float(np.sum((truths == old_class) & (predictions == new_class)))
                            / old_total if old_total else None
                        ),
                        "is_final_task": task == int(self.args["expected_tasks"]),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "rp_selected_confusion.csv",
            pair_rows,
            ("protocol", "run_id", "seed", "task", "new_class", "old_class"),
        )
