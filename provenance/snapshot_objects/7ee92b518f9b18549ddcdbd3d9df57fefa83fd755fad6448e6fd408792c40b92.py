"""C0--C4 learner layered on the read-only similarity-aware extension."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from similarity.learner import SimilarityAwareLearner
from utils.experiment_io import read_csv, upsert_csv_rows

from .losses import selective_control_relation
from .snapshot import create_control_snapshot


class ControlLearner(SimilarityAwareLearner):
    """AM-Drift with exact-count high/low/random pair controls."""

    CONTROL_COMPATIBILITY_KEYS = (
        "control_selection_mode",
        "control_fraction",
        "control_weak_weight",
    )

    def __init__(self, args):
        super().__init__(args)
        self._control_state_task = None
        self._control_high_mask = None
        self._control_low_mask = None
        self._control_random_mask = None
        self._control_pair_weights = None
        self._control_rank = None
        self._control_train_context = None
        create_control_snapshot(
            self.run_dir,
            base_root=self.args["base_root"],
            e3_root=self.args["e3_root"],
            similarity_root=self.args["similarity_extension_root"],
            control_root=self.args["control_extension_root"],
        )

    def load_experiment_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("config", {})
        mismatches = {
            key: {"checkpoint": stored.get(key), "current": self.args.get(key)}
            for key in self.CONTROL_COMPATIBILITY_KEYS
            if stored.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"control checkpoint/config mismatch: {mismatches}")
        return super().load_experiment_checkpoint(path)

    @staticmethod
    def _stable_order(values, descending):
        """Sort by similarity with old class id as deterministic tie-break."""
        array = torch.as_tensor(values).detach().cpu().numpy()
        class_ids = np.arange(array.shape[0])
        primary = -array if descending else array
        return torch.from_numpy(np.lexsort((class_ids, primary))).long()

    def _random_mask(self, new_count, old_count, selected_count):
        mask = torch.zeros((new_count, old_count), dtype=torch.bool)
        seed = int(self.args["seed"])
        task = int(self._cur_task + 1)
        for new_offset in range(new_count):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed * 1_000_003 + task * 10_007 + new_offset)
            chosen = torch.randperm(old_count, generator=generator)[:selected_count]
            mask[new_offset, chosen] = True
        return mask

    def _ensure_similarity_state(self):
        super()._ensure_similarity_state()
        if self._cur_task <= 0 or self._control_state_task == self._cur_task:
            return
        similarities = torch.as_tensor(self._similarity_matrix).detach().cpu().float()
        new_count, old_count = similarities.shape
        selected_count = max(
            1, min(old_count, int(math.ceil(old_count * float(self.args["control_fraction"]))))
        )
        high = torch.zeros_like(similarities, dtype=torch.bool)
        low = torch.zeros_like(similarities, dtype=torch.bool)
        ranks = torch.empty_like(similarities, dtype=torch.long)
        for new_offset in range(new_count):
            descending = self._stable_order(similarities[new_offset], descending=True)
            ascending = self._stable_order(similarities[new_offset], descending=False)
            high[new_offset, descending[:selected_count]] = True
            low[new_offset, ascending[:selected_count]] = True
            ranks[new_offset, descending] = torch.arange(1, old_count + 1)
        random = self._random_mask(new_count, old_count, selected_count)

        mode = str(self.args["control_selection_mode"])
        weights = torch.ones_like(similarities)
        if mode == "high_off":
            weights[high] = 0.0
        elif mode == "low_off":
            weights[low] = 0.0
        elif mode == "random_off":
            weights[random] = 0.0
        elif mode == "high_weak":
            weights[high] = float(self.args["control_weak_weight"])

        self._control_high_mask = high
        self._control_low_mask = low
        self._control_random_mask = random
        self._control_pair_weights = weights
        self._control_rank = ranks
        self._control_state_task = self._cur_task
        self._write_control_pair_selection()

    def _pair_group(self, new_offset, old_class_id):
        if (
            str(self.args["control_selection_mode"]) == "random_off"
            and bool(self._control_random_mask[new_offset, old_class_id])
        ):
            return "random"
        if bool(self._control_high_mask[new_offset, old_class_id]):
            return "high"
        if bool(self._control_low_mask[new_offset, old_class_id]):
            return "low"
        return "middle"

    def _write_control_pair_selection(self):
        mode = str(self.args["control_selection_mode"])
        new_count, old_count = self._similarity_matrix.shape
        rows = []
        for new_offset in range(new_count):
            new_class_id = self._known_classes + new_offset
            for old_class_id in range(old_count):
                rank = int(self._control_rank[new_offset, old_class_id])
                quantile = 1.0 if old_count == 1 else 1.0 - (rank - 1) / (old_count - 1)
                selected_off = bool(
                    (mode == "high_off" and self._control_high_mask[new_offset, old_class_id])
                    or (mode == "low_off" and self._control_low_mask[new_offset, old_class_id])
                    or (mode == "random_off" and self._control_random_mask[new_offset, old_class_id])
                )
                selected_weak = bool(
                    mode == "high_weak" and self._control_high_mask[new_offset, old_class_id]
                )
                rows.append(
                    {
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "new_class_id": new_class_id,
                        "old_class_id": old_class_id,
                        "similarity": float(self._similarity_matrix[new_offset, old_class_id]),
                        "similarity_rank": rank,
                        "similarity_quantile": quantile,
                        "group": self._pair_group(new_offset, old_class_id),
                        "selected_for_off": selected_off,
                        "selected_for_weak": selected_weak,
                        "weak_weight": (
                            float(self.args["control_weak_weight"]) if selected_weak else 1.0
                        ),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "control_pair_selection.csv",
            rows,
            ("method", "seed", "task", "new_class_id", "old_class_id"),
        )

    def _new_epoch_accumulator(self):
        new_count, old_count = self._control_pair_weights.shape
        return {
            "counts": torch.zeros(new_count, dtype=torch.long),
            "d2": torch.zeros((new_count, old_count), dtype=torch.float64),
            "rho": torch.zeros((new_count, old_count), dtype=torch.float64),
            "raw": torch.zeros((new_count, old_count), dtype=torch.float64),
            "weighted": torch.zeros((new_count, old_count), dtype=torch.float64),
            "intrusion": torch.zeros((new_count, old_count), dtype=torch.float64),
        }

    def _accumulate_control_pairs(self, relation, targets):
        context = self._control_train_context
        if context is None:
            return
        call_index = context["call_index"]
        batch_count = context["batch_count"]
        epoch = context["start_epoch"] + call_index // batch_count
        batch_index = call_index % batch_count
        accumulator = context.get("accumulator")
        if accumulator is None or context.get("accumulator_epoch") != epoch:
            accumulator = self._new_epoch_accumulator()
            context["accumulator"] = accumulator
            context["accumulator_epoch"] = epoch
        offsets = targets.detach().cpu().long() - int(self._known_classes)
        diag = relation.diagnostics
        d2 = diag["all_d2"].detach().cpu().double()
        rho = diag["all_rho"].detach().cpu().double()
        raw = diag["all_raw_sep_loss"].detach().cpu().double()
        weighted = diag["all_weighted_sep_loss"].detach().cpu().double()
        for new_offset in offsets.unique(sorted=True).tolist():
            mask = offsets == int(new_offset)
            count = int(mask.sum())
            accumulator["counts"][new_offset] += count
            accumulator["d2"][new_offset] += d2[mask].sum(dim=0)
            accumulator["rho"][new_offset] += rho[mask].sum(dim=0)
            accumulator["raw"][new_offset] += raw[mask].sum(dim=0)
            accumulator["weighted"][new_offset] += weighted[mask].sum(dim=0)
            accumulator["intrusion"][new_offset] += rho[mask].lt(1.0).double().sum(dim=0)
        context["call_index"] += 1
        if batch_index + 1 == batch_count:
            self._flush_control_epoch(epoch, accumulator)
            context["accumulator"] = None

    def _flush_control_epoch(self, epoch, accumulator):
        adaptive = self._adaptive_lookup()
        new_count, old_count = self._control_pair_weights.shape
        rows = []
        for new_offset in range(new_count):
            count = int(accumulator["counts"][new_offset])
            if count <= 0:
                raise RuntimeError(f"no training samples for new-class offset {new_offset}")
            new_class_id = self._known_classes + new_offset
            for old_class_id in range(old_count):
                margin = adaptive[old_class_id]
                rows.append(
                    {
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "new_class_id": new_class_id,
                        "old_class_id": old_class_id,
                        "similarity": float(self._similarity_matrix[new_offset, old_class_id]),
                        "base_tau": float(margin["tau_birth"]),
                        "drift_scale": float(margin["adaptive_scale"]),
                        "final_tau": float(self._separation_cache.taus[old_class_id]),
                        "D2": float(accumulator["d2"][new_offset, old_class_id] / count),
                        "rho": float(accumulator["rho"][new_offset, old_class_id] / count),
                        "raw_sep_loss": float(accumulator["raw"][new_offset, old_class_id] / count),
                        "applied_sep_weight": float(
                            self._control_pair_weights[new_offset, old_class_id]
                        ),
                        "weighted_sep_loss": float(
                            accumulator["weighted"][new_offset, old_class_id] / count
                        ),
                        "intrusion": float(
                            accumulator["intrusion"][new_offset, old_class_id] / count
                        ),
                    }
                )
        upsert_csv_rows(
            self.run_dir / "control_pair_separation.csv",
            rows,
            ("method", "seed", "task", "epoch", "new_class_id", "old_class_id"),
        )

        counts = accumulator["counts"].double()[:, None].expand(new_count, old_count)
        intrusion = accumulator["intrusion"] / counts.clamp_min(1)
        effective = intrusion * self._control_pair_weights.double()
        summary = {
            "method": self.args["experiment_id"],
            "seed": self.args["seed"],
            "task": self._cur_task + 1,
            "epoch": epoch + 1,
            "overall_intrusion": float(intrusion.mean()),
            "high_sim_intrusion": float(intrusion[self._control_high_mask].mean()),
            "low_sim_intrusion": float(intrusion[self._control_low_mask].mean()),
            "random_intrusion": float(intrusion[self._control_random_mask].mean()),
            "overall_effective_intrusion": float(effective.mean()),
            "high_sim_effective_intrusion": float(effective[self._control_high_mask].mean()),
            "low_sim_effective_intrusion": float(effective[self._control_low_mask].mean()),
            "random_effective_intrusion": float(effective[self._control_random_mask].mean()),
        }
        upsert_csv_rows(
            self.run_dir / "control_intrusion.csv",
            [summary],
            ("method", "seed", "task", "epoch"),
        )

    def _relation_loss(self, features, targets):
        if self._control_pair_weights is None:
            raise RuntimeError("control pair selection was not initialized")
        offsets = targets.long() - int(self._known_classes)
        pair_weights = self._control_pair_weights.to(features.device)[offsets]
        result = selective_control_relation(features, self._separation_cache, pair_weights)
        self._accumulate_control_pairs(result, targets)
        return result

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        batch_count = len(train_loader)
        if batch_count <= 0:
            raise ValueError("training loader is empty")
        self._control_train_context = {
            "start_epoch": int(start_epoch),
            "batch_count": int(batch_count),
            "call_index": 0,
            "accumulator": None,
        }
        try:
            return super()._init_train(
                train_loader, test_loader, optimizer, scheduler, start_epoch=start_epoch
            )
        finally:
            self._control_train_context = None

    def _group_union(self, mask):
        return set(torch.nonzero(mask.any(dim=0), as_tuple=False).flatten().tolist())

    def _write_control_group_metrics(self, predictions, y_true):
        task = self._cur_task + 1
        old_total = int(np.sum(y_true < self._known_classes))
        new_total = int(np.sum(y_true >= self._known_classes))
        forgetting = {
            int(row["old_class_id"]): float(row["forgetting_increment"])
            for row in read_csv(self.run_dir / "class_forgetting.csv")
            if int(row["task"]) == task
        }
        metrics = {
            "method": self.args["experiment_id"],
            "seed": self.args["seed"],
            "task": task,
        }
        for label, mask in (
            ("high_sim", self._control_high_mask),
            ("low_sim", self._control_low_mask),
            ("random", self._control_random_mask),
        ):
            old_classes = self._group_union(mask)
            old_mask = np.isin(y_true, list(old_classes))
            old_correct = int(np.sum(predictions[old_mask] == y_true[old_mask]))
            metrics[f"{label}_old_accuracy"] = (
                100.0 * old_correct / int(old_mask.sum()) if old_mask.any() else None
            )
            current_new = y_true >= self._known_classes
            new_correct = int(np.sum(predictions[current_new] == y_true[current_new]))
            metrics[f"{label}_new_accuracy"] = (
                100.0 * new_correct / new_total if new_total else None
            )
            new_to_old = 0
            old_to_new = 0
            for truth, prediction in zip(y_true.tolist(), predictions.tolist()):
                if truth >= self._known_classes and prediction < self._known_classes:
                    new_offset = truth - self._known_classes
                    new_to_old += int(bool(mask[new_offset, prediction]))
                elif truth < self._known_classes and prediction >= self._known_classes:
                    new_offset = prediction - self._known_classes
                    if 0 <= new_offset < mask.shape[0]:
                        old_to_new += int(bool(mask[new_offset, truth]))
            metrics[f"{label}_new_to_old_confusion"] = (
                new_to_old / new_total if new_total else None
            )
            metrics[f"{label}_old_to_new_confusion"] = (
                old_to_new / old_total if old_total else None
            )
            values = [forgetting[class_id] for class_id in old_classes if class_id in forgetting]
            metrics[f"{label}_forgetting"] = float(np.mean(values)) if values else None
        upsert_csv_rows(
            self.run_dir / "control_group_metrics.csv",
            [metrics],
            ("method", "seed", "task"),
        )

    def _write_evaluation_diagnostics(self, y_pred, y_true):
        super()._write_evaluation_diagnostics(y_pred, y_true)
        if self._known_classes == 0 or self._similarity_matrix is None:
            return
        predictions = y_pred[:, 0]
        rows = []
        for true_class in range(self._total_classes):
            true_mask = y_true == true_class
            total = int(true_mask.sum())
            if not total:
                continue
            values, counts = np.unique(predictions[true_mask], return_counts=True)
            for pred_class, count in zip(values.tolist(), counts.tolist()):
                similarity = None
                group = "not_current_old_new_pair"
                if true_class >= self._known_classes and pred_class < self._known_classes:
                    new_offset, old_class = true_class - self._known_classes, pred_class
                    similarity = float(self._similarity_matrix[new_offset, old_class])
                    group = self._pair_group(new_offset, old_class)
                elif true_class < self._known_classes and pred_class >= self._known_classes:
                    new_offset, old_class = pred_class - self._known_classes, true_class
                    if 0 <= new_offset < self._similarity_matrix.shape[0]:
                        similarity = float(self._similarity_matrix[new_offset, old_class])
                        group = self._pair_group(new_offset, old_class)
                rows.append(
                    {
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "true_class": true_class,
                        "pred_class": int(pred_class),
                        "true_old_or_new": (
                            "old" if true_class < self._known_classes else "new"
                        ),
                        "pred_old_or_new": (
                            "old" if pred_class < self._known_classes else "new"
                        ),
                        "pair_similarity": similarity,
                        "pair_group": group,
                        "count": int(count),
                        "normalized_rate": float(count) / total,
                    }
                )
        upsert_csv_rows(
            self.run_dir / "control_confusion.csv",
            rows,
            ("method", "seed", "task", "true_class", "pred_class"),
        )
        self._write_control_group_metrics(predictions, y_true)
