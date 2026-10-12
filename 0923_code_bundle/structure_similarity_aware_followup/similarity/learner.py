"""Similarity-aware learner layered on the read-only E3 extension."""

from __future__ import annotations

import csv
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from followup.diagnostics import component_gradient_metrics, mean_numeric_rows
from followup.learner import FollowupLearner
from followup.losses import prototype_geometry, scalar_tensor_summary
from utils.distill_loss import (
    LobachevskyRelativeGeomLoss,
    features_distill_loss,
    logits_distill_loss,
)
from utils.experiment_io import (
    capture_rng_state,
    read_csv,
    restore_rng_state,
    update_json,
    upsert_csv_rows,
    utc_now_iso,
)
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy

from .losses import (
    hard_relative_relation,
    legacy_absolute_relation,
    similarity_tau_relation,
    similarity_weighted_scc,
    soft_hybrid_relation,
    cosine_similarity_matrix,
)
from .snapshot import create_similarity_snapshot


class SimilarityAwareLearner(FollowupLearner):
    """B0/B1/S0-S4 implementation with frozen pre-training similarities."""

    NEW_COMPATIBILITY_KEYS = (
        "similarity_relation_mode",
        "similarity_alpha",
        "relative_margin",
        "similarity_beta_source",
        "similarity_beta_quantile",
        "similarity_beta_fixed",
        "similarity_scc_enabled",
        "similarity_scc_alpha",
        "similarity_scc_min_weight",
        "soft_gate_temperature",
    )

    def __init__(self, args):
        super().__init__(args)
        self._similarity_state_task = None
        self._similarity_matrix = None
        self._similarity_new_old_prototypes = None
        self._similarity_beta = None
        self._pair_similarity_logged_task = None
        create_similarity_snapshot(
            self.run_dir,
            base_root=self.args["base_root"],
            e3_root=self.args["e3_root"],
            extension_root=self.args["similarity_extension_root"],
        )

    def load_experiment_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        stored = checkpoint.get("config", {})
        mismatches = {
            key: {"checkpoint": stored.get(key), "current": self.args.get(key)}
            for key in self.NEW_COMPATIBILITY_KEYS
            if stored.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"similarity-aware checkpoint/config mismatch: {mismatches}")
        return super().load_experiment_checkpoint(path)

    def _birth_task(self, class_id):
        start = 0
        for task_index, size in enumerate(self.task_sizes):
            if start <= int(class_id) < start + int(size):
                return task_index + 1
            start += int(size)
        increment = int(self.args.get("increment", 40))
        initial = int(self.args.get("init_cls", increment))
        return 1 if class_id < initial else 2 + (class_id - initial) // increment

    def _beta_from_s0(self):
        path = (
            Path(self.args["output_root"])
            / self.args["dataset"]
            / "S0"
            / f"seed_{self.args['seed']}"
            / "pair_similarity.csv"
        )
        rows = [
            row
            for row in read_csv(path)
            if int(row["task"]) == self._cur_task + 1
        ]
        if not rows:
            raise RuntimeError(
                f"S0 beta source is missing for seed={self.args['seed']} "
                f"task={self._cur_task + 1}: {path}. Run S0 first or choose task_quantile."
            )
        values = torch.tensor([float(row["similarity"]) for row in rows])
        return float(torch.quantile(values, float(self.args["similarity_beta_quantile"])))

    def _resolve_beta(self):
        source = str(self.args.get("similarity_beta_source", "S0"))
        if source == "fixed":
            return float(self.args["similarity_beta_fixed"])
        if source == "S0" and self.args["experiment_id"] not in {"B0", "B1", "S0", "S1"}:
            return self._beta_from_s0()
        return float(
            torch.quantile(
                self._similarity_matrix.flatten(),
                float(self.args.get("similarity_beta_quantile", 0.75)),
            )
        )

    def _ensure_similarity_state(self):
        """Freeze q_n^old and s_nk before any optimizer step of this task."""
        if self._cur_task <= 0 or self._similarity_state_task == self._cur_task:
            return
        rng_state = capture_rng_state()
        try:
            old_model_features, labels = self._collect_current_features("tfeatures")
        finally:
            restore_rng_state(rng_state)
        q_old, _ = self._statistics_from_features(old_model_features, labels)
        old_prototypes = torch.as_tensor(self._task_source_means).detach().cpu().float()
        self._similarity_new_old_prototypes = torch.as_tensor(q_old).detach().cpu().float()
        self._similarity_matrix = cosine_similarity_matrix(
            self._similarity_new_old_prototypes, old_prototypes
        ).cpu()
        self._similarity_beta = self._resolve_beta()
        self._similarity_state_task = self._cur_task
        self._pair_similarity_logged_task = None
        logging.info(
            "Frozen old-model similarity task=%d shape=%s beta=%.6f source=%s",
            self._cur_task + 1,
            tuple(self._similarity_matrix.shape),
            self._similarity_beta,
            self.args.get("similarity_beta_source"),
        )

    def _adaptive_lookup(self):
        return {int(row["class_id"]): row for row in self._adaptive_rows}

    def _write_pair_similarity(self):
        if self._pair_similarity_logged_task == self._cur_task:
            return
        adaptive = self._adaptive_lookup()
        corrected = torch.as_tensor(self._followup_old_prototypes).cpu().float()
        source = torch.as_tensor(self._task_source_means).cpu().float()
        drift = torch.linalg.vector_norm(corrected - source, dim=1)
        top_k = min(int(self.args.get("top_k_similar_old", 5)), self._known_classes)
        rows = []
        for new_offset in range(self._similarity_matrix.shape[0]):
            new_class_id = self._known_classes + new_offset
            order = torch.argsort(self._similarity_matrix[new_offset], descending=True)
            ranks = {int(class_id): rank + 1 for rank, class_id in enumerate(order.tolist())}
            for old_class_id in range(self._known_classes):
                rank = ranks[old_class_id]
                margin = adaptive.get(old_class_id, {})
                rows.append(
                    {
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "new_class_id": new_class_id,
                        "old_class_id": old_class_id,
                        "similarity": float(self._similarity_matrix[new_offset, old_class_id]),
                        "similarity_rank_within_new": rank,
                        "is_top1": rank == 1,
                        "is_top5": rank <= top_k,
                        "old_class_birth_task": self._birth_task(old_class_id),
                        "old_class_age": self._cur_task + 1 - self._birth_task(old_class_id),
                        "apr_drift_norm": float(drift[old_class_id]),
                        "tau_before": margin.get("tau_birth"),
                        "tau_after_if_adaptive": margin.get("tau"),
                        "beta": self._similarity_beta,
                        "beta_source": self.args.get("similarity_beta_source"),
                        "similarity_feature_space": "frozen_previous_task_teacher",
                    }
                )
        upsert_csv_rows(
            self.run_dir / "pair_similarity.csv",
            rows,
            ("method", "seed", "task", "new_class_id", "old_class_id"),
        )
        self._pair_similarity_logged_task = self._cur_task

    def _write_similarity_tau_diagnostics(self, epoch):
        if self.args.get("similarity_relation_mode") != "similarity_tau":
            return
        adaptive = self._adaptive_lookup()
        rows = []
        alpha = float(self.args["similarity_alpha"])
        for new_offset in range(self._similarity_matrix.shape[0]):
            new_class_id = self._known_classes + new_offset
            for old_class_id in range(self._known_classes):
                similarity = float(self._similarity_matrix[new_offset, old_class_id])
                similarity_scale = 1.0 - alpha * min(1.0, max(0.0, (similarity + 1.0) * 0.5))
                margin = adaptive[old_class_id]
                rows.append(
                    {
                        "method": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "new_class_id": new_class_id,
                        "old_class_id": old_class_id,
                        "similarity": similarity,
                        "apr_drift_norm": margin["apr_shift_norm"],
                        "drift_scale": margin["adaptive_scale"],
                        "similarity_scale": similarity_scale,
                        "base_tau": margin["tau_birth"],
                        "drift_tau": margin["tau"],
                        "final_tau": float(margin["tau"]) * similarity_scale,
                    }
                )
        upsert_csv_rows(
            self.run_dir / "similarity_tau_diagnostics.csv",
            rows,
            ("method", "seed", "task", "epoch", "new_class_id", "old_class_id"),
        )

    def _refresh_followup_cache(self, epoch):
        self._ensure_similarity_state()
        super()._refresh_followup_cache(epoch)
        self._write_pair_similarity()
        self._write_similarity_tau_diagnostics(epoch)

    def _relation_loss(self, features, targets):
        mode = str(self.args.get("similarity_relation_mode", "none"))
        if mode == "legacy_absolute":
            return legacy_absolute_relation(features, self._separation_cache)
        if mode == "similarity_tau":
            return similarity_tau_relation(
                features,
                targets,
                self._separation_cache,
                self._similarity_matrix,
                self._known_classes,
                alpha=float(self.args["similarity_alpha"]),
            )
        common = dict(
            features=features,
            targets=targets,
            cache=self._separation_cache,
            old_prototypes=self._followup_old_prototypes,
            current_prototypes=self._followup_current_prototypes,
            similarities=self._similarity_matrix,
            known_classes=self._known_classes,
            beta=self._similarity_beta,
            margin=float(self.args["relative_margin"]),
        )
        if mode == "hard_relative":
            return hard_relative_relation(**common)
        if mode == "soft_hybrid":
            return soft_hybrid_relation(
                **common, temperature=float(self.args["soft_gate_temperature"])
            )
        raise ValueError(f"relation loss requested for inactive/unknown mode: {mode}")

    def _sample_relation_rows(self, relation, epoch, batch_index, sample_ids, targets):
        mode = str(self.args["similarity_relation_mode"])
        if mode not in {"hard_relative", "soft_hybrid"}:
            return [], []
        diag = relation.diagnostics
        rows = []
        gates = []
        for index in range(targets.shape[0]):
            new_class = int(targets[index])
            hard_old = int(diag["hard_old_index"][index])
            common = {
                "method": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task": self._cur_task + 1,
                "epoch": epoch + 1,
                "batch": batch_index,
                "sample_id": int(sample_ids[index]),
                "sample_in_batch": index,
                "new_class_id": new_class,
                "hard_old_class_id": hard_old,
                "class_similarity": float(diag["hard_class_similarity"][index]),
                "s_new": float(diag["s_new"][index]),
                "s_old": float(diag["s_old"][index]),
                "cosine_gap": float(diag["cosine_gap"][index]),
                "margin": float(self.args["relative_margin"]),
                "margin_violation": bool(diag["margin_violation"][index]),
                "relative_loss": float(diag["relative_sample_loss"][index]),
                "absolute_loss": float(diag["absolute_sample_loss"][index]),
                "used_absolute_or_relative": (
                    "relative" if bool(diag["used_relative"][index]) else "absolute"
                ),
                "beta": self._similarity_beta,
            }
            rows.append(common)
            if mode == "soft_hybrid":
                gates.append(
                    {
                        **common,
                        "old_class_id": hard_old,
                        "similarity": common["class_similarity"],
                        "gate_weight": float(diag["gate_weight"][index]),
                        "weighted_absolute": float(diag["weighted_absolute"][index]),
                        "weighted_relative": float(diag["weighted_relative"][index]),
                    }
                )
        return rows, gates

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        progress = tqdm(range(start_epoch, self.tuned_epochs))
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            eps=1e-7,
            s=self.args["scale"],
            m=self.args["margin"],
        )
        for epoch in progress:
            epoch_started = time.perf_counter()
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            if self._needs_followup_cache() and epoch % int(
                self.args.get("prototype_refresh_epochs", self.args["sep_refresh_epochs"])
            ) == 0:
                self._refresh_followup_cache(epoch)
            self._network.train()

            sums = defaultdict(float)
            vectors = defaultdict(list)
            gradient_rows = []
            gradient_export_rows = []
            relative_rows = []
            gate_rows = []
            scc_rows = []
            class_diag = defaultdict(lambda: defaultdict(list))
            correct, processed_samples, optimizer_steps = 0, 0, 0
            scc_active = self._cur_task > 0 and bool(self.args["scc"])
            feature_active = self._cur_task > 0 and not scc_active and bool(self.args["feature"])
            logit_active = (
                self._cur_task > 0
                and not scc_active
                and not feature_active
                and bool(self.args["logit"])
            )
            relation_active = self._separation_is_active()
            similarity_scc = scc_active and bool(self.args.get("similarity_scc_enabled", False))

            for batch_index, (sample_ids, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                sample_ids = torch.as_tensor(sample_ids).cpu().long()
                batch_size = int(targets.shape[0])
                outputs = self._network(inputs, label=targets)
                logits = outputs["logits"]
                features = outputs["sfeatures"]
                loss_cls = loss_cos(
                    logits[:, self._known_classes :], targets - self._known_classes
                )
                loss_scc_raw = loss_scc_effective = loss_feature = loss_logit = None
                scc_result = None
                if self._cur_task == 0:
                    original_loss = loss_cls
                else:
                    if similarity_scc:
                        scc_result = similarity_weighted_scc(
                            outputs,
                            self._class_means,
                            targets,
                            self._similarity_matrix,
                            self._known_classes,
                            alpha=float(self.args["similarity_scc_alpha"]),
                            minimum_weight=float(self.args["similarity_scc_min_weight"]),
                        )
                        loss_scc_raw = scc_result.raw_loss
                        loss_scc_effective = scc_result.weighted_loss
                    else:
                        loss_scc_raw = LobachevskyRelativeGeomLoss(
                            outputs, self._class_means, self._device
                        )
                        loss_scc_effective = loss_scc_raw
                    loss_feature = features_distill_loss(outputs)
                    loss_logit = logits_distill_loss(
                        outputs, self._class_means, self._network, T=4.0
                    )
                    if scc_active:
                        original_loss = loss_cls + float(self.args["scc_value"]) * loss_scc_effective
                    elif feature_active:
                        original_loss = loss_cls + 0.1 * loss_feature
                    elif logit_active:
                        original_loss = loss_cls + 5.0 * loss_logit
                    else:
                        original_loss = loss_cls

                zero = features.sum() * 0.0
                relation = None
                relation_loss = absolute_component = relative_component = zero
                if relation_active:
                    if self._separation_cache is None:
                        raise RuntimeError("active similarity relation has no separation cache")
                    relation = self._relation_loss(features, targets)
                    relation_loss = relation.loss
                    absolute_component = relation.absolute_component
                    relative_component = relation.relative_component
                weighted_relation = float(self.args["sep_lambda"]) * relation_loss
                loss = original_loss + weighted_relation
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss task={self._cur_task + 1} "
                        f"epoch={epoch + 1} batch={batch_index}"
                    )

                grad_interval = int(self.args.get("diagnostic_grad_interval", 20))
                if grad_interval > 0 and batch_index % grad_interval == 0:
                    components = {
                        "classification": loss_cls,
                        "scc": loss_scc_effective if scc_active else None,
                        "abs": absolute_component if relation_active else None,
                        "rel": (
                            relative_component
                            if self.args["similarity_relation_mode"] in {"hard_relative", "soft_hybrid"}
                            else None
                        ),
                    }
                    metrics = component_gradient_metrics(components, self._network.parameters())
                    row = {
                        "experiment_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task_id": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "batch": batch_index,
                        **metrics,
                    }
                    gradient_rows.append(row)
                    gradient_export_rows.append(
                        {
                            "method": self.args["experiment_id"],
                            "seed": self.args["seed"],
                            "task": self._cur_task + 1,
                            "epoch": epoch + 1,
                            "batch": batch_index,
                            "grad_norm_cls": metrics.get("grad_classification_norm"),
                            "grad_norm_scc": metrics.get("grad_scc_norm"),
                            "grad_norm_abs": metrics.get("grad_abs_norm"),
                            "grad_norm_rel": metrics.get("grad_rel_norm"),
                            "cos_cls_scc": metrics.get("grad_cosine_classification_scc"),
                            "cos_cls_abs": metrics.get("grad_cosine_classification_abs"),
                            "cos_cls_rel": metrics.get("grad_cosine_classification_rel"),
                            "cos_scc_rel": metrics.get("grad_cosine_scc_rel"),
                        }
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sums["classification"] += float(loss_cls.item()) * batch_size
                if scc_active:
                    sums["scc_raw"] += float(loss_scc_raw.item()) * batch_size
                    sums["scc_effective"] += float(loss_scc_effective.item()) * batch_size
                if feature_active:
                    sums["feature_distillation"] += float(loss_feature.item()) * batch_size
                if logit_active:
                    sums["logit_distillation"] += float(loss_logit.item()) * batch_size
                if relation_active:
                    sums["relation"] += float(relation_loss.item()) * batch_size
                    sums["absolute_component"] += float(absolute_component.item()) * batch_size
                    sums["relative_component"] += float(relative_component.item()) * batch_size
                    sums["weighted_relation"] += float(weighted_relation.item()) * batch_size
                sums["original_structure"] += float(original_loss.item()) * batch_size
                sums["total"] += float(loss.item()) * batch_size
                vectors["feature_norm"].append(features.detach().norm(dim=1).cpu())

                geometry = None
                if self._cur_task > 0 and self._followup_old_prototypes is not None:
                    geometry = prototype_geometry(
                        features,
                        targets,
                        self._followup_old_prototypes,
                        self._followup_current_prototypes,
                        self._known_classes,
                    )
                    for key in (
                        "nearest_old_cosine",
                        "correct_new_cosine",
                        "cosine_gap",
                        "nearest_old_euclidean",
                    ):
                        vectors[key].append(geometry[key].cpu())
                if relation is not None:
                    diag = relation.diagnostics
                    for key in ("nearest_d2", "nearest_rho", "nearest_tau"):
                        vectors[key].append(diag[key].cpu())
                    vectors["intrusion"].append(diag["intrusion"].float().cpu())
                    if "margin_violation" in diag:
                        vectors["relative_margin_violation"].append(
                            diag["margin_violation"].float().cpu()
                        )
                    vectors["relation_gate"].append(diag["gate_weight"].float().cpu())
                    all_d2 = diag["all_d2"].cpu()
                    all_rho = diag["all_rho"].cpu()
                    if "all_pair_taus" in diag:
                        all_taus = diag["all_pair_taus"].cpu()
                    else:
                        all_taus = self._separation_cache.taus.cpu()[None, :].expand_as(all_d2)
                    for old_class_id in range(self._known_classes):
                        d2 = all_d2[:, old_class_id]
                        rho = all_rho[:, old_class_id]
                        tau = all_taus[:, old_class_id]
                        class_diag[old_class_id]["d2"].extend(d2.tolist())
                        class_diag[old_class_id]["rho"].extend(rho.tolist())
                        class_diag[old_class_id]["tau"].extend(tau.tolist())
                        class_diag[old_class_id]["intrusion"].extend(rho.lt(1.0).float().tolist())

                    if bool(self.args.get("diagnostic_sample_rows", True)):
                        batch_relative, batch_gates = self._sample_relation_rows(
                            relation, epoch, batch_index, sample_ids, targets.detach().cpu()
                        )
                        relative_rows.extend(batch_relative)
                        gate_rows.extend(batch_gates)
                if scc_result is not None and bool(self.args.get("diagnostic_sample_rows", True)):
                    diag = scc_result.diagnostics
                    for index in range(batch_size):
                        scc_rows.append(
                            {
                                "method": self.args["experiment_id"],
                                "seed": self.args["seed"],
                                "task": self._cur_task + 1,
                                "epoch": epoch + 1,
                                "batch": batch_index,
                                "sample_id": int(sample_ids[index]),
                                "sample_in_batch": index,
                                "new_class_id": int(targets[index]),
                                "max_old_similarity": float(diag["max_old_similarity"][index]),
                                "scc_weight": float(diag["scc_weight"][index]),
                                "raw_scc_loss": float(diag["raw_scc_loss"][index]),
                                "weighted_scc_loss": float(diag["weighted_scc_loss"][index]),
                            }
                        )

                predictions = logits.max(dim=1).indices
                correct += predictions.eq(targets).cpu().sum()
                processed_samples += batch_size
                optimizer_steps += 1

            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_started
            checkpoint_enabled = bool(self.args.get("checkpoint_every_epoch", True))
            checkpoint_state_id = (
                f"train_epoch:task_{self._cur_task + 1}:next_epoch_{epoch + 1}"
                if checkpoint_enabled
                else None
            )
            diagnostics = {
                "experiment_id": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task_id": self._cur_task + 1,
                "stage": "feature_training",
                "epoch": epoch + 1,
                "classification_active": True,
                "classification_raw": sums["classification"] / processed_samples,
                "classification_coefficient": 1.0,
                "classification_weighted": sums["classification"] / processed_samples,
                "scc_active": scc_active,
                "scc_similarity_aware": similarity_scc,
                "scc_raw": sums["scc_raw"] / processed_samples if scc_active else None,
                "scc_effective_raw": sums["scc_effective"] / processed_samples if scc_active else None,
                "scc_coefficient": float(self.args["scc_value"]) if scc_active else None,
                "scc_weighted": (
                    sums["scc_effective"] * float(self.args["scc_value"]) / processed_samples
                    if scc_active
                    else None
                ),
                "feature_distillation_active": feature_active,
                "feature_distillation_raw": sums["feature_distillation"] / processed_samples if feature_active else None,
                "feature_distillation_coefficient": 0.1 if feature_active else None,
                "feature_distillation_weighted": sums["feature_distillation"] * 0.1 / processed_samples if feature_active else None,
                "logit_distillation_active": logit_active,
                "logit_distillation_raw": sums["logit_distillation"] / processed_samples if logit_active else None,
                "logit_distillation_coefficient": 5.0 if logit_active else None,
                "logit_distillation_weighted": sums["logit_distillation"] * 5.0 / processed_samples if logit_active else None,
                "separation_active": relation_active,
                "separation_raw": sums["relation"] / processed_samples if relation_active else None,
                "separation_coefficient": float(self.args["sep_lambda"]) if relation_active else None,
                "separation_weighted": sums["weighted_relation"] / processed_samples if relation_active else None,
                "similarity_relation_mode": self.args["similarity_relation_mode"],
                "absolute_component_raw": sums["absolute_component"] / processed_samples if relation_active else None,
                "relative_component_raw": sums["relative_component"] / processed_samples if relation_active else None,
                "similarity_beta": self._similarity_beta if self._cur_task > 0 else None,
                "original_structure_loss": sums["original_structure"] / processed_samples,
                "total_loss": sums["total"] / processed_samples,
                "learning_rate": learning_rates[0],
                "learning_rates": learning_rates,
                "processed_samples": processed_samples,
                "optimizer_steps": optimizer_steps,
                "epoch_seconds": epoch_seconds,
                "nan_or_inf_detected": False,
                "checkpoint_path": str(self.run_dir / "checkpoint_latest.pt") if checkpoint_enabled else None,
                "checkpoint_state_id": checkpoint_state_id,
            }
            for key, chunks in vectors.items():
                diagnostics.update(scalar_tensor_summary(torch.cat(chunks), key))
            diagnostics["student_feature_norm"] = diagnostics.get("feature_norm_mean")
            diagnostics["intrusion_ratio"] = diagnostics.get("intrusion_mean")
            if gradient_rows:
                averaged = mean_numeric_rows(
                    [
                        {key: value for key, value in row.items() if key.startswith("grad_")}
                        for row in gradient_rows
                    ]
                )
                diagnostics.update({f"epoch_{key}": value for key, value in averaged.items()})
                upsert_csv_rows(
                    self.run_dir / "gradient_metrics.csv",
                    gradient_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "batch"),
                )
                upsert_csv_rows(
                    self.run_dir / "gradient_diagnostics.csv",
                    gradient_export_rows,
                    ("method", "seed", "task", "epoch", "batch"),
                )
            upsert_csv_rows(
                self.run_dir / "epoch_metrics.csv",
                [diagnostics],
                ("experiment_id", "seed", "task_id", "stage", "epoch"),
            )

            if relative_rows:
                upsert_csv_rows(
                    self.run_dir / "relative_margin_diagnostics.csv",
                    relative_rows,
                    ("method", "seed", "task", "epoch", "batch", "sample_in_batch"),
                )
            if gate_rows:
                upsert_csv_rows(
                    self.run_dir / "soft_gate_diagnostics.csv",
                    gate_rows,
                    ("method", "seed", "task", "epoch", "batch", "sample_in_batch"),
                )
            if scc_rows:
                upsert_csv_rows(
                    self.run_dir / "scc_similarity_diagnostics.csv",
                    scc_rows,
                    ("method", "seed", "task", "epoch", "batch", "sample_in_batch"),
                )

            feature_row = {
                "method": self.args["experiment_id"],
                "seed": self.args["seed"],
                "task": self._cur_task + 1,
                "epoch": epoch + 1,
                "mean": diagnostics.get("feature_norm_mean"),
                "std": diagnostics.get("feature_norm_std"),
                "p50": diagnostics.get("feature_norm_p50"),
                "p90": diagnostics.get("feature_norm_p90"),
                "p95": diagnostics.get("feature_norm_p95"),
                "max": diagnostics.get("feature_norm_max"),
            }
            upsert_csv_rows(
                self.run_dir / "feature_norm.csv",
                [feature_row],
                ("method", "seed", "task", "epoch"),
            )
            if self._cur_task > 0:
                geometry_row = {
                    "method": self.args["experiment_id"],
                    "seed": self.args["seed"],
                    "task": self._cur_task + 1,
                    "epoch": epoch + 1,
                    "nearest_old_cosine": diagnostics.get("nearest_old_cosine_mean"),
                    "correct_new_cosine": diagnostics.get("correct_new_cosine_mean"),
                    "cosine_gap": diagnostics.get("cosine_gap_mean"),
                    "diagonal_D2": diagnostics.get("nearest_d2_mean"),
                    "tau": diagnostics.get("nearest_tau_mean"),
                    "rho": diagnostics.get("nearest_rho_mean"),
                    "intrusion": diagnostics.get("intrusion_mean"),
                }
                upsert_csv_rows(
                    self.run_dir / "geometry_diagnostics.csv",
                    [geometry_row],
                    ("method", "seed", "task", "epoch"),
                )

            if self._adaptive_rows and class_diag:
                adaptive = self._adaptive_lookup()
                class_rows = []
                for class_id, values in sorted(class_diag.items()):
                    row = dict(adaptive.get(class_id, {}))
                    row.update(
                        {
                            "experiment_id": self.args["experiment_id"],
                            "seed": self.args["seed"],
                            "task_id": self._cur_task + 1,
                            "epoch": epoch + 1,
                            "class_id": class_id,
                            "evaluated_new_sample_count": len(values["rho"]),
                            "intrusion_ratio": float(np.mean(values["intrusion"])),
                            "active_separation_ratio": float(np.mean(values["intrusion"])),
                            "mean_separation_penalty": float(
                                np.mean(np.maximum(0.0, 1.0 - np.asarray(values["rho"])))
                            ),
                            "current_d2_mean": float(np.mean(values["d2"])),
                            "current_tau_mean": float(np.mean(values["tau"])),
                        }
                    )
                    class_rows.append(row)
                upsert_csv_rows(
                    self.run_dir / "adaptive_class_diagnostics.csv",
                    class_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "class_id"),
                )

            train_acc = np.around(tensor2numpy(correct) * 100 / processed_samples, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                diagnostics["total_loss"],
                train_acc,
                test_acc,
            )
            progress.set_description(info)
            if checkpoint_enabled:
                self.save_experiment_checkpoint(
                    self.run_dir / "checkpoint_latest.pt",
                    stage="train_epoch",
                    next_epoch=epoch + 1,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
            metadata = {
                "updated_at_utc": utc_now_iso(),
                "last_recorded_stage": "feature_training",
                "last_recorded_task": self._cur_task + 1,
                "last_recorded_epoch": epoch + 1,
            }
            if checkpoint_enabled:
                metadata.update(
                    {
                        "last_checkpoint_state_id": checkpoint_state_id,
                        "last_checkpoint_path": str(self.run_dir / "checkpoint_latest.pt"),
                    }
                )
            update_json(self.run_dir / "run_metadata.json", metadata)

        if start_epoch >= self.tuned_epochs:
            logging.info("Training epochs already complete; continuing task finalization")

    def _write_evaluation_diagnostics(self, y_pred, y_true):
        predictions = y_pred[:, 0]
        task = self._cur_task + 1
        method = self.args["experiment_id"]
        seed = self.args["seed"]

        # Store only non-zero cells; absent cells are exact zeros. Rates are
        # normalized within each true class and remain single-head predictions.
        confusion_rows = []
        for true_class in range(self._total_classes):
            mask = y_true == true_class
            total = int(mask.sum())
            if not total:
                continue
            values, counts = np.unique(predictions[mask], return_counts=True)
            for pred_class, count in zip(values.tolist(), counts.tolist()):
                confusion_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "task": task,
                        "true_class": true_class,
                        "pred_class": int(pred_class),
                        "true_is_old_or_new": "old" if true_class < self._known_classes else "new",
                        "pred_is_old_or_new": "old" if pred_class < self._known_classes else "new",
                        "count": int(count),
                        "true_class_total": total,
                        "normalized_rate": float(count) / total,
                    }
                )
        upsert_csv_rows(
            self.run_dir / "cross_task_confusion.csv",
            confusion_rows,
            ("method", "seed", "task", "true_class", "pred_class"),
        )

        if self._known_classes == 0:
            return
        previous = {
            int(row["mapped_class_id"]): float(row["accuracy"])
            for row in read_csv(self.run_dir / "class_accuracy.csv")
            if int(row["trained_task"]) == task - 1 and row.get("eval_stage") == "task_final"
        }
        rows = []
        for class_id in range(self._known_classes):
            mask = y_true == class_id
            total = int(mask.sum())
            correct = int(np.sum(predictions[mask] == y_true[mask]))
            after = 100.0 * correct / total
            before = previous.get(class_id)
            if before is None:
                raise RuntimeError(
                    f"missing previous class accuracy for task={task} class={class_id}"
                )
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "task": task,
                    "old_class_id": class_id,
                    "accuracy_before_current_task": before,
                    "accuracy_after_current_task": after,
                    "delta_accuracy": after - before,
                    "forgetting_increment": before - after,
                    "num_correct_after": correct,
                    "num_samples": total,
                }
            )
        upsert_csv_rows(
            self.run_dir / "class_forgetting.csv",
            rows,
            ("method", "seed", "task", "old_class_id"),
        )

    def _eval_cnn(self, loader):
        y_pred, y_true = super()._eval_cnn(loader)
        if loader is getattr(self, "test_loader", None):
            self._write_evaluation_diagnostics(y_pred, y_true)
        return y_pred, y_true

