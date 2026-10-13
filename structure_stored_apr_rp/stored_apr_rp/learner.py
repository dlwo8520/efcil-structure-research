"""Independent StoredAPR subclass; the original RP learner is never changed."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rp.learner import RPProtoCLADLearner, _pearson, _spearman
from rp.projection import project_relu
from selective.losses import top_p_selection
from utils.experiment_io import capture_rng_state, upsert_csv_rows, utc_now_iso

from .losses import stored_apr_rp_relation
from .prototypes import apr_corrected_prototypes, birth_prototypes, comparison
from .snapshot import create_snapshot


class StoredAPRRPProtoCLADLearner(RPProtoCLADLearner):
    """Use birth-time sample-wise RP means plus APR-derived projected drift."""

    def __init__(self, args):
        super().__init__(args)
        if not args.get("stored_apr_rp_enabled", False):
            raise ValueError("StoredAPR learner requires stored_apr_rp_enabled=true")
        self._birth_records: dict[int, dict] = {}
        self._capture_birth_features = False
        self._birth_features = None
        self._birth_labels = None
        create_snapshot(self.run_dir, args["stored_apr_rp_root"])

    def _collect_current_features(self, feature_key="sfeatures", return_sample_ids=False):
        result = super()._collect_current_features(feature_key, return_sample_ids)
        if self._capture_birth_features and feature_key == "sfeatures" and return_sample_ids:
            self._birth_features = torch.as_tensor(result[0]).detach().cpu().float()
            self._birth_labels = torch.as_tensor(result[1]).detach().cpu().long()
        return result

    def _update_region_statistics(self, data_manager, new_means_after, new_covariances_after):
        # Parent already performs a task-final pass over current (not old) train
        # samples here. Reuse that pass, before classifier adaptation, for B.
        self._capture_birth_features = True
        self._birth_features = self._birth_labels = None
        try:
            super()._update_region_statistics(data_manager, new_means_after, new_covariances_after)
        finally:
            self._capture_birth_features = False
        if self._birth_features is None or self._birth_labels is None:
            raise RuntimeError("task-final current sample features were not captured")
        class_ids = range(self._known_classes, self._total_classes)
        if any(class_id in self._birth_records for class_id in class_ids):
            raise RuntimeError("birth RP prototype would overwrite an existing class")
        projected = self._project_feature_chunks(self._birth_features)
        new_records = birth_prototypes(
            self._birth_features, self._birth_labels, class_ids,
            self._rp_matrix_cpu, projected=projected,
        )
        rows = []
        for class_id, entry in new_records.items():
            entry.update({
                "class_id": class_id,
                "birth_task": self._cur_task + 1,
                "rp_seed": int(self.args["rp_seed"]),
                "rp_shape": list(self.args["rp_matrix_shape"]),
                "rp_checksum": str(self.args["rp_matrix_checksum"]),
                "feature_dim": int(self.args["rp_input_dim"]),
                "rp_dim": int(self.args["rp_dim"]),
                "feature_space": self._feature_space_id("task_final"),
            })
            self._birth_records[class_id] = entry
            oldstyle = project_relu(entry["mu_birth"][None], self._rp_matrix_cpu)[0]
            metrics = comparison(oldstyle, entry["q_birth"])
            rows.append({
                "protocol": self.args["protocol_id"],
                "run_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "stage": "birth",
                "task": self._cur_task + 1,
                "epoch": 0,
                "class_id": class_id,
                "sample_count": entry["sample_count"],
                "birth_cosine": metrics["cosine"],
                "birth_l2": metrics["l2"],
                "birth_relative_l2": metrics["relative_l2"],
                "q_birth_norm": float(entry["q_birth"].norm()),
                "q_birth_oldstyle_norm": float(oldstyle.norm()),
                "rp_checksum": entry["rp_checksum"],
            })
        upsert_csv_rows(
            self.run_dir / "stored_rp_prototype_diagnostics.csv", rows,
            ("protocol", "run_id", "seed", "stage", "task", "epoch", "class_id"),
        )
        self._save_birth_sidecar()
        # A class-level statistic is retained; sample features are immediately discarded.
        self._birth_features = self._birth_labels = None

    def _birth_state(self):
        return {
            "schema_version": 1,
            "rp_seed": int(self.args["rp_seed"]),
            "rp_shape": list(self.args["rp_matrix_shape"]),
            "rp_checksum": str(self.args["rp_matrix_checksum"]),
            "records": self._birth_records,
        }

    def _validate_birth_state(self, state, expected_old_classes):
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("missing or unsupported stored RP prototype state")
        for key, expected in (
            ("rp_seed", int(self.args["rp_seed"])),
            ("rp_shape", list(self.args["rp_matrix_shape"])),
            ("rp_checksum", str(self.args["rp_matrix_checksum"])),
        ):
            if state.get(key) != expected:
                raise ValueError(f"stored RP {key} mismatch")
        records = state.get("records")
        if not isinstance(records, dict) or sorted(records) != list(range(expected_old_classes)):
            raise ValueError("stored RP prototypes do not cover exactly the old classes")
        for class_id, entry in records.items():
            if int(entry.get("class_id", -1)) != class_id:
                raise ValueError("stored RP class ID mismatch")
            if not 1 <= int(entry.get("birth_task", 0)) <= int(self.args["expected_tasks"]):
                raise ValueError("stored RP birth task is invalid")
            for key, expected in (
                ("rp_seed", int(self.args["rp_seed"])),
                ("rp_shape", list(self.args["rp_matrix_shape"])),
                ("feature_dim", int(self.args["rp_input_dim"])),
                ("rp_dim", int(self.args["rp_dim"])),
            ):
                if entry.get(key) != expected:
                    raise ValueError(f"stored RP class {key} mismatch")
            if tuple(entry["mu_birth"].shape) != (int(self.args["rp_input_dim"]),):
                raise ValueError("stored RP birth mean dimension mismatch")
            if tuple(entry["q_birth"].shape) != (int(self.args["rp_dim"]),):
                raise ValueError("stored RP birth projection dimension mismatch")
            if entry.get("rp_checksum") != self.args["rp_matrix_checksum"]:
                raise ValueError("stored RP class matrix checksum mismatch")
            if not torch.isfinite(entry["mu_birth"]).all() or not torch.isfinite(entry["q_birth"]).all():
                raise ValueError("stored RP class statistic contains non-finite values")
        return records

    def _save_birth_sidecar(self):
        target = self.run_dir / "stored_rp_birth_prototypes.pt"
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(self._birth_state(), temporary)
        os.replace(temporary, target)

    def _corrected_old_q(self, device):
        apr = torch.as_tensor(self._followup_old_prototypes, device=device).float()
        return apr_corrected_prototypes(
            self._birth_records, apr, self._rp_on(device), device=device,
        )

    def _relation_loss(self, features, targets):
        corrected, _, _, _ = self._corrected_old_q(features.device)
        return stored_apr_rp_relation(
            features=features,
            targets=targets,
            cache=self._separation_cache,
            apr_old_prototypes=self._followup_old_prototypes,
            current_prototypes=self._followup_current_prototypes,
            stored_old_projected=corrected,
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
        corrected, oldstyle, birth_q, projected_birth_mean = self._corrected_old_q("cpu")
        original = F.normalize(current, dim=1) @ F.normalize(old, dim=1).T
        current_rp = project_relu(current, self._rp_matrix_cpu)
        projected = F.normalize(current_rp, dim=1) @ F.normalize(corrected, dim=1).T
        selected, ranks, _, _ = top_p_selection(self._similarity_matrix, float(self.args["top_p"]))
        pair_rows = []
        for new_offset in range(current.shape[0]):
            for old_class in range(old.shape[0]):
                if not bool(selected[new_offset, old_class]):
                    continue
                rank = int(ranks[new_offset, old_class])
                pair_rows.append({
                    "protocol": self.args["protocol_id"],
                    "run_id": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task": self._cur_task + 1,
                    "epoch": epoch + 1,
                    "new_class": self._known_classes + new_offset,
                    "old_class": old_class,
                    "original_cosine": float(original[new_offset, old_class]),
                    "rp_cosine": float(projected[new_offset, old_class]),
                    "delta_cosine": float(projected[new_offset, old_class] - original[new_offset, old_class]),
                    "selection_similarity_original": float(self._similarity_matrix[new_offset, old_class]),
                    "similarity_rank": rank,
                    "similarity_percentile": 1.0 - (rank - 1) / self._known_classes,
                })
        upsert_csv_rows(
            self.run_dir / "rp_pair_geometry_diagnostics.csv", pair_rows,
            ("protocol", "run_id", "seed", "task", "epoch", "new_class", "old_class"),
        )
        left, right = original.numpy().ravel(), projected.numpy().ravel()
        upsert_csv_rows(
            self.run_dir / "rp_similarity_structure_diagnostics.csv",
            [{
                "protocol": self.args["protocol_id"],
                "run_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task": self._cur_task + 1,
                "epoch": epoch + 1,
                "pair_count": int(left.size),
                "pearson_original_vs_rp": _pearson(left, right),
                "spearman_original_vs_rp": _spearman(left, right),
            }],
            ("protocol", "run_id", "seed", "task", "epoch"),
        )
        old_rows = []
        for class_id in range(self._known_classes):
            metrics = comparison(corrected[class_id], oldstyle[class_id])
            old_rows.append({
                "protocol": self.args["protocol_id"],
                "run_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "stage": "old_epoch",
                "task": self._cur_task + 1,
                "epoch": epoch + 1,
                "class_id": class_id,
                "birth_task": self._birth_records[class_id]["birth_task"],
                "stored_vs_oldstyle_cosine": metrics["cosine"],
                "stored_vs_oldstyle_l2": metrics["l2"],
                "stored_vs_oldstyle_relative_l2": metrics["relative_l2"],
                "apr_delta_norm": float((oldstyle[class_id] - projected_birth_mean[class_id]).norm()),
                "q_current_norm": float(corrected[class_id].norm()),
                "q_current_negative_ratio": float((corrected[class_id] < 0).float().mean()),
                "q_birth_norm": float(birth_q[class_id].norm()),
            })
        upsert_csv_rows(
            self.run_dir / "stored_rp_prototype_diagnostics.csv", old_rows,
            ("protocol", "run_id", "seed", "stage", "task", "epoch", "class_id"),
        )

    def save_experiment_checkpoint(self, path, stage, next_epoch=0, optimizer=None, scheduler=None):
        """Parent checkpoint schema plus immutable birth-time class statistics."""
        self._validate_birth_state(self._birth_state(), self._known_classes)
        state_id = (
            f"train_epoch:task_{self._cur_task + 1}:next_epoch_{int(next_epoch)}"
            if stage == "train_epoch" else f"{stage}:task_{self._cur_task + 1}"
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
            "stored_apr_rp": self._birth_state(),
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, target)
        from utils.experiment_io import write_json
        write_json(self.run_dir / "checkpoint_manifest.json", {
            "checkpoint_path": str(target.resolve()),
            "checkpoint_state_id": state_id,
            "stage": stage,
            "task_id": self._cur_task + 1,
            "next_epoch": int(next_epoch),
            "updated_at_utc": utc_now_iso(),
            "rp_projection": self.args["rp_projection"],
            "stored_rp_class_count": len(self._birth_records),
            "stored_rp_schema_version": 1,
        })
        return state_id

    def load_experiment_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        records = self._validate_birth_state(
            payload.get("stored_apr_rp"), int(payload["known_classes"]),
        )
        result = super().load_experiment_checkpoint(path)
        self._birth_records = records
        return result
