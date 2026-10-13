"""Three-method learner; all parent repositories remain read-only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from similarity.learner import SimilarityAwareLearner
from similarity.losses import legacy_absolute_relation
from utils.experiment_io import read_csv, upsert_csv_rows, write_json

from .diagnostics import export_run_diagnostics
from .losses import (
    enlarged_boundary_relation,
    proto_clad_relation,
    selective_am_relation,
    top_p_selection,
)
from .snapshot import create_sweep_snapshot


class SelectiveSimilarityLearner(SimilarityAwareLearner):
    """Vanilla Structure, C0, and similarity-selective A/B/C methods."""

    CHECKPOINT_KEYS = (
        "protocol_id", "task_partition", "class_order_seed", "selective_method",
        "top_p", "alpha", "lambda_clad", "diagnostic_pair_interval",
    )

    def __init__(self, args):
        super().__init__(args)
        self._selection_logged_task = None
        self._selective_epoch_stats = {}
        create_sweep_snapshot(
            self.run_dir,
            {
                "structure_base": args["base_root"],
                "e3_followup": args["e3_root"],
                "similarity_extension": args["similarity_extension_root"],
                "selective_sweep": args["selective_sweep_root"],
            },
        )

    def load_experiment_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("config", {})
        mismatches = {
            key: {"checkpoint": stored.get(key), "current": self.args.get(key)}
            for key in self.CHECKPOINT_KEYS
            if stored.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"selective-sweep checkpoint/config mismatch: {mismatches}")
        return super().load_experiment_checkpoint(path)

    def _relation_loss(self, features, targets):
        method = str(self.args["selective_method"])
        if method == "baseline_am_drift":
            return legacy_absolute_relation(features, self._separation_cache)
        common = dict(
            features=features,
            targets=targets,
            cache=self._separation_cache,
            similarities=self._similarity_matrix,
            known_classes=self._known_classes,
            top_p=float(self.args["top_p"]),
        )
        if method == "selective_am_drift":
            return selective_am_relation(**common)
        if method == "enlarged_boundary":
            return enlarged_boundary_relation(**common, alpha=float(self.args["alpha"]))
        if method == "proto_clad":
            return proto_clad_relation(
                **common, apr_old_prototypes=self._followup_old_prototypes
            )
        raise ValueError(f"unknown selective_method: {method}")

    def _refresh_followup_cache(self, epoch):
        super()._refresh_followup_cache(epoch)
        self._write_conflict_selection()
        self._write_apr_drift(epoch)

    def _write_conflict_selection(self):
        if self._cur_task <= 0 or self._selection_logged_task == self._cur_task:
            return
        fraction = self.args.get("top_p")
        if fraction is None:
            self._selection_logged_task = self._cur_task
            return
        selected, ranks, cutoffs, _ = top_p_selection(self._similarity_matrix, float(fraction))
        rows = []
        old_count = self._known_classes
        for new_offset in range(self._similarity_matrix.shape[0]):
            new_class = old_count + new_offset
            for old_class in range(old_count):
                rank = int(ranks[new_offset, old_class])
                rows.append(
                    {
                        "protocol": self.args["protocol_id"],
                        "dataset": self.args["source_dataset"],
                        "num_tasks": self.args["expected_tasks"],
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"], "task": self._cur_task + 1,
                        "new_class_id": new_class, "old_class_id": old_class,
                        "similarity": float(self._similarity_matrix[new_offset, old_class]),
                        "similarity_rank": rank,
                        "similarity_percentile": 1.0 - (rank - 1) / old_count,
                        "P": fraction, "selected": bool(selected[new_offset, old_class]),
                        "cutoff_beta": float(cutoffs[new_offset]),
                        "old_birth_task": self._birth_task(old_class),
                        "old_age": self._cur_task + 1 - self._birth_task(old_class),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "conflict_selection.csv", rows,
            ("protocol", "method", "seed", "task", "new_class_id", "old_class_id"),
        )
        distribution_path = self.run_dir / "similarity_distribution.json"
        if distribution_path.is_file():
            record = json.loads(distribution_path.read_text(encoding="utf-8"))
        else:
            record = {"protocol": self.args["protocol_id"], "tasks": {}}
        values = self._similarity_matrix.detach().cpu().numpy().ravel()
        record["tasks"][str(self._cur_task + 1)] = {
            "pair_count": int(values.size), "old_class_count": old_count,
            "new_class_count": int(self._similarity_matrix.shape[0]), "P": fraction,
            "selected_per_new_class": int(selected[0].sum()),
            "mean": float(values.mean()), "std": float(values.std()),
            "min": float(values.min()), "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)), "p90": float(np.quantile(values, 0.90)),
            "max": float(values.max()),
        }
        write_json(distribution_path, record)
        self._selection_logged_task = self._cur_task

    def _write_apr_drift(self, epoch):
        if self._cur_task <= 0 or self._followup_old_prototypes is None:
            return
        source = torch.as_tensor(self._task_source_means).cpu().float()
        corrected = torch.as_tensor(self._followup_old_prototypes).cpu().float()
        shift = corrected - source
        adaptive = {int(row["class_id"]): row for row in self._adaptive_rows}
        variances = self._separation_cache.factors.detach().cpu().float()
        rows = []
        for class_id in range(self._known_classes):
            entry = adaptive[class_id]
            direction_cos = torch.nn.functional.cosine_similarity(
                shift[class_id : class_id + 1], source[class_id : class_id + 1], dim=1
            )[0]
            rows.append(
                {
                    "protocol": self.args["protocol_id"], "seed": self.args["seed"],
                    "task": self._cur_task + 1, "epoch": epoch + 1,
                    "old_class_id": class_id, "birth_task": self._birth_task(class_id),
                    "age": self._cur_task + 1 - self._birth_task(class_id),
                    "apr_shift_norm": float(torch.linalg.vector_norm(shift[class_id])),
                    "apr_shift_l1": float(shift[class_id].abs().sum()),
                    "apr_shift_cosine_direction_if_available": float(direction_cos),
                    "base_tau": entry["tau_birth"], "drift_scale": entry["adaptive_scale"],
                    "tau_drift": entry["tau"],
                    "mean_diag_variance": float(variances[class_id].mean()),
                    "median_diag_variance": float(variances[class_id].median()),
                    "trace_cov": float(variances[class_id].sum()),
                }
            )
        upsert_csv_rows(
            self.run_dir / "apr_drift_diagnostics.csv", rows,
            ("protocol", "seed", "task", "epoch", "old_class_id"),
        )

    def _sample_relation_rows(self, relation, epoch, batch_index, sample_ids, targets):
        method = self.args["method"]
        if method in {"Structure", "C0"}:
            return [], []
        device_diag = relation.diagnostics
        if method == "A":
            selected_mask = device_diag["selected_mask"].detach().cpu()
            similarities = device_diag["similarity"].detach().cpu()
            d2_values = device_diag["all_d2"].detach().cpu()
            tau_values = device_diag["all_pair_taus"].detach().cpu()
            pair_losses = device_diag["pair_loss"].detach().cpu()
            rho_values = device_diag["all_rho"].detach().cpu()
            for sample_index, target in enumerate(targets):
                mask = selected_mask[sample_index]
                key = (self._cur_task + 1, epoch + 1, int(target))
                state = self._selective_epoch_stats.setdefault(
                    key,
                    {
                        "selected_old_count": int(mask.sum()), "count": 0, "similarity_sum": 0.0,
                        "similarity_min": float("inf"), "similarity_max": float("-inf"),
                        "intrusion_count": 0, "d2_sum": 0.0, "d2_reservoir": [],
                        "tau_sum": 0.0, "loss_sum": 0.0,
                    },
                )
                sim = similarities[sample_index][mask]
                d2 = d2_values[sample_index][mask]
                tau = tau_values[sample_index][mask]
                loss = pair_losses[sample_index][mask]
                rho = rho_values[sample_index][mask]
                count = int(mask.sum())
                state["count"] += count
                state["similarity_sum"] += float(sim.sum())
                state["similarity_min"] = min(state["similarity_min"], float(sim.min()))
                state["similarity_max"] = max(state["similarity_max"], float(sim.max()))
                state["intrusion_count"] += int(rho.lt(1.0).sum())
                state["d2_sum"] += float(d2.sum())
                remaining = max(0, 512 - len(state["d2_reservoir"]))
                if remaining:
                    state["d2_reservoir"].extend(d2[:remaining].tolist())
                state["tau_sum"] += float(tau.sum())
                state["loss_sum"] += float(loss.sum())
        interval = max(1, int(self.args.get("diagnostic_pair_interval", 20)))
        if batch_index % interval:
            return [], []
        if self.args.get("diagnostic_pair_epochs", "last") == "last" and epoch + 1 != self.tuned_epochs:
            return [], []
        diag = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in relation.diagnostics.items()}
        rows = []
        old_count = self._known_classes
        for sample_index in range(len(targets)):
            for old_class in range(old_count):
                selected = bool(diag["selected_mask"][sample_index, old_class])
                if method != "C" and not selected:
                    continue
                pair_slot = sample_index * old_count + old_class
                row = {
                    "protocol": self.args["protocol_id"], "method": self.args["experiment_id"],
                    "seed": self.args["seed"], "task": self._cur_task + 1, "epoch": epoch + 1,
                    "batch": batch_index, "sample_id": int(sample_ids[sample_index]),
                    # Parent upsert key includes sample_in_batch, so encode the pair.
                    "sample_in_batch": pair_slot, "actual_sample_in_batch": sample_index,
                    "new_class_id": int(targets[sample_index]), "old_class_id": old_class,
                    "selected": selected, "selected_old_count": int(diag["selected_mask"][sample_index].sum()),
                    "similarity": float(diag["similarity"][sample_index, old_class]),
                    "similarity_rank": int(diag["similarity_rank"][sample_index, old_class]),
                    "cutoff_beta": float(diag["cutoff_beta"][sample_index]),
                    "D2": float(diag["all_d2"][sample_index, old_class]),
                    "tau_drift": float(self._separation_cache.taus.cpu()[old_class]),
                    "tau_effective": float(diag["all_pair_taus"][sample_index, old_class]),
                    "tau_scale": float(diag.get("tau_scale", torch.ones_like(diag["all_d2"]))[sample_index, old_class]),
                    "r_similarity_excess": float(diag.get("similarity_excess", torch.zeros_like(diag["all_d2"]))[sample_index, old_class]),
                    "rho": float(diag["all_rho"][sample_index, old_class]),
                    "intrusion": bool(diag["all_rho"][sample_index, old_class] < 1.0),
                    "relation_pair_loss": float(diag["pair_loss"][sample_index, old_class]),
                    "raw_hinge": float(diag["raw_hinge"][sample_index, old_class]),
                    "pair_cosine": (
                        float(diag["pair_cosine"][sample_index, old_class])
                        if "pair_cosine" in diag else None
                    ),
                }
                rows.append(row)
        # The parent writes this list after each epoch. We use a neutral staging
        # filename there, then copy it to the explicitly named table below.
        return rows, []

    def _init_train(self, *args, **kwargs):
        super()._init_train(*args, **kwargs)
        source = self.run_dir / "relative_margin_diagnostics.csv"
        target = self.run_dir / "selective_pair_samples.csv"
        if source.is_file():
            # Rows in this project are our neutral pair schema. Preserve them
            # under a stable name expected by post-processing.
            target.write_bytes(source.read_bytes())
        export_run_diagnostics(self.run_dir, self.args)
        self._write_selective_epoch_summary()

    def _write_selective_epoch_summary(self):
        if self.args["method"] != "A" or not self._selective_epoch_stats:
            return
        rows = []
        for (task, epoch, new_class), state in sorted(self._selective_epoch_stats.items()):
            count = state["count"]
            samples = np.asarray(state["d2_reservoir"], dtype=np.float64)
            rows.append(
                {
                    "protocol": self.args["protocol_id"], "task": task, "epoch": epoch,
                    "new_class_id": new_class, "selected_old_count": state["selected_old_count"],
                    "mean_selected_similarity": state["similarity_sum"] / count,
                    "max_selected_similarity": state["similarity_max"],
                    "min_selected_similarity": state["similarity_min"],
                    "active_sep_ratio": state["intrusion_count"] / count,
                    "mean_D2": state["d2_sum"] / count,
                    "p10_D2": float(np.quantile(samples, 0.10)),
                    "p50_D2": float(np.quantile(samples, 0.50)),
                    "p90_D2": float(np.quantile(samples, 0.90)),
                    "D2_quantile_sample_cap": 512,
                    "mean_tau_drift": state["tau_sum"] / count,
                    "mean_sep_loss": state["loss_sum"] / count,
                    "weighted_sep_loss": state["loss_sum"] / count * float(self.args["sep_lambda"]),
                }
            )
        upsert_csv_rows(
            self.run_dir / "selective_am_diagnostics.csv", rows,
            ("protocol", "task", "epoch", "new_class_id"),
        )

    def _write_evaluation_diagnostics(self, y_pred, y_true):
        super()._write_evaluation_diagnostics(y_pred, y_true)
        predictions = y_pred[:, 0]
        task = self._cur_task + 1
        matrix = np.zeros((self._total_classes, self._total_classes), dtype=np.int64)
        np.add.at(matrix, (y_true.astype(int), predictions.astype(int)), 1)
        np.savez_compressed(
            self.run_dir / f"confusion_task_{task:02d}.npz",
            matrix=matrix,
            true_class_ids=np.arange(self._total_classes),
            pred_class_ids=np.arange(self._total_classes),
        )
        if self._known_classes == 0 or self.args["method"] in {"Structure", "C0"}:
            return
        selected, ranks, cutoffs, _ = top_p_selection(
            self._similarity_matrix, float(self.args["top_p"])
        )
        rows = []
        for new_offset in range(self._similarity_matrix.shape[0]):
            true_class = self._known_classes + new_offset
            total = int((y_true == true_class).sum())
            if not total:
                continue
            for old_class in range(self._known_classes):
                count = int(matrix[true_class, old_class])
                rows.append(
                    {
                        "protocol": self.args["protocol_id"], "method": self.args["experiment_id"],
                        "seed": self.args["seed"], "task": task, "new_class_id": true_class,
                        "old_class_id": old_class, "count": count, "new_class_total": total,
                        "normalized_rate": count / total, "similarity": float(self._similarity_matrix[new_offset, old_class]),
                        "similarity_rank": int(ranks[new_offset, old_class]),
                        "selected": bool(selected[new_offset, old_class]),
                        "cutoff_beta": float(cutoffs[new_offset]),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "confusion_pairs.csv", rows,
            ("protocol", "method", "seed", "task", "new_class_id", "old_class_id"),
        )
