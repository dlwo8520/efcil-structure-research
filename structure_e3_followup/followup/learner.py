"""Structure learner subclass containing only follow-up experiment changes."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from models.CDS_adapter import Learner as StructureLearner
from utils.adjust_distribution import LobachevskyPrototypeCorrector
from utils.distribution_regions import SeparationCache, flatten_regions
from utils.distill_loss import (
    LobachevskyRelativeGeomLoss,
    features_distill_loss,
    logits_distill_loss,
)
from utils.experiment_io import (
    append_jsonl,
    capture_rng_state,
    restore_rng_state,
    update_json,
    upsert_csv_rows,
    utc_now_iso,
)
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import tensor2numpy

from .adaptive_margin import recalibrate_taus
from .diagnostics import component_gradient_metrics, mean_numeric_rows
from .losses import (
    angular_hard_negative_loss,
    diagonal_region_loss,
    pic_loss,
    prototype_geometry,
    scalar_tensor_summary,
)
from .snapshot import create_run_snapshot


class FollowupLearner(StructureLearner):
    """Non-invasive extension of the pinned Structure learner."""

    def __init__(self, args):
        super().__init__(args)
        self._followup_old_prototypes = None
        self._followup_current_prototypes = None
        self._adaptive_rows: List[Dict] = []
        create_run_snapshot(
            self.run_dir,
            config=self.args,
            base_root=self.args["base_root"],
            extension_root=self.args["extension_root"],
            launch_command=self.args.get("launch_command", "unknown"),
        )

    def load_experiment_checkpoint(self, path):
        """Reject resuming a checkpoint under a different follow-up method."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("config", {})
        keys = (
            "pic_enabled",
            "pic_lambda",
            "pic_temperature",
            "ahn_enabled",
            "ahn_lambda",
            "ahn_margin",
            "ahn_top_k",
            "normalized_diagonal",
            "adaptive_margin_mode",
            "adaptive_margin_strength",
            "ceos_replay_fraction",
            "ceos_lambda_min",
            "ceos_enemy_top_k",
        )
        mismatches = {
            key: {"checkpoint": checkpoint_config.get(key), "current": self.args.get(key)}
            for key in keys
            if checkpoint_config.get(key) != self.args.get(key)
        }
        if mismatches:
            raise ValueError(f"follow-up checkpoint/config mismatch: {mismatches}")
        return super().load_experiment_checkpoint(path)

    def _needs_followup_cache(self) -> bool:
        # Refresh for every method so baseline/CEOS receive the same common
        # cosine and Euclidean geometry diagnostics as loss-based variants.
        return self._cur_task > 0

    def _refresh_followup_cache(self, epoch: int) -> None:
        """One current-task pass supplies PIC/AHN prototypes and E3 cache.

        Only current-task data, stored statistics, and APR corrections are
        used.  No old raw exemplar is loaded or reconstructed.
        """
        if self.args.get("multi_center", False):
            raise ValueError("follow-up experiments intentionally require multi_center=false")
        if self._task_source_means is None or self._sep_reference_means is None:
            raise RuntimeError("APR source/reference statistics are missing")
        rng_state = capture_rng_state()
        started = time.perf_counter()
        try:
            target_features, target_labels = self._collect_current_features("sfeatures")
            target_means, target_covariances = self._statistics_from_features(
                target_features, target_labels
            )
            if self.args.get("apr", True):
                corrected_means, corrected_covariances = self.adjust_distribution.correct_prototypes(
                    self._task_source_means,
                    self._task_source_covs,
                    self._sep_reference_means,
                    self._sep_reference_covs,
                    target_means,
                    target_covariances,
                )
            else:
                corrected_means = self._task_source_means.copy()
                corrected_covariances = self._task_source_covs.detach().cpu().clone()

            self._followup_old_prototypes = torch.as_tensor(corrected_means).detach().cpu().float()
            self._followup_current_prototypes = torch.as_tensor(target_means).detach().cpu().float()
            self._adaptive_rows = []

            if self._separation_is_active():
                feature_space = self._feature_space_id(f"epoch_{epoch}_followup_cache")
                regions = []
                birth_tasks = []
                for class_id in range(self._known_classes):
                    boundary = self._single_regions[class_id][0]
                    region = dict(boundary)
                    region["mean"] = torch.as_tensor(corrected_means[class_id]).cpu().float()
                    region["covariance"] = torch.as_tensor(
                        corrected_covariances[class_id]
                    ).cpu().float()
                    region["tau_by_covariance"] = dict(boundary["tau_by_covariance"])
                    region["regularization"] = dict(boundary.get("regularization", {}))
                    region["feature_space"] = feature_space
                    regions.append(region)
                    birth_tasks.append(int(boundary["birth_task"]))

                self._separation_cache = SeparationCache(
                    regions,
                    self.args["separation_covariance"],
                    feature_space,
                    self.args,
                )
                mode = str(self.args.get("adaptive_margin_mode", "none"))
                drift = torch.linalg.vector_norm(
                    torch.as_tensor(corrected_means).cpu().float()
                    - torch.as_tensor(self._task_source_means).cpu().float(),
                    dim=1,
                )
                adaptive = recalibrate_taus(
                    birth_taus=self._separation_cache.taus.cpu(),
                    variances=self._separation_cache.factors.cpu(),
                    class_ids=self._separation_cache.class_ids.cpu(),
                    birth_tasks=torch.tensor(birth_tasks, dtype=torch.long),
                    current_task=self._cur_task,
                    drift_norms=drift,
                    mode=mode,
                    strength=float(self.args.get("adaptive_margin_strength", 0.25)),
                    min_scale=float(self.args.get("adaptive_margin_min_scale", 0.5)),
                    max_scale=float(self.args.get("adaptive_margin_max_scale", 2.0)),
                )
                self._separation_cache.taus = adaptive.taus.to(self._separation_cache.taus.device)
                source_trace = self._task_source_covs.detach().cpu().float().diagonal(
                    dim1=-2, dim2=-1
                ).sum(dim=1)
                corrected_trace = self._separation_cache.factors.detach().cpu().float().sum(dim=1)
                expansion = corrected_trace / source_trace.clamp_min(1e-8)
                self._adaptive_rows = [
                    {
                        "experiment_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task_id": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "adaptive_margin_mode": mode,
                        "covariance_expansion_ratio": float(expansion[index]),
                        **row,
                    }
                    for index, row in enumerate(adaptive.rows)
                ]
                upsert_csv_rows(
                    self.run_dir / "adaptive_margin.csv",
                    self._adaptive_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "class_id"),
                )
                cache_record = {
                    "task": self._cur_task,
                    "epoch": epoch,
                    "feature_space": feature_space,
                    "covariance_view": self.args["separation_covariance"],
                    "normalized_diagonal": bool(self.args.get("normalized_diagonal", False)),
                    "adaptive_margin_mode": mode,
                    "region_count": self._separation_cache.region_count,
                    "cache_storage_bytes": self._separation_cache.storage_bytes,
                    "refresh_seconds": time.perf_counter() - started,
                }
                append_jsonl(self.run_dir / "followup_cache.jsonl", cache_record)
                logging.info("Follow-up cache: %s", cache_record)
        finally:
            restore_rng_state(rng_state)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, start_epoch=0):
        prog_bar = tqdm(range(start_epoch, self.tuned_epochs))
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface",
            eps=1e-7,
            s=self.args["scale"],
            m=self.args["margin"],
        )
        for epoch in prog_bar:
            epoch_started = time.perf_counter()
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            if (
                self._needs_followup_cache()
                and epoch % int(self.args.get("prototype_refresh_epochs", self.args["sep_refresh_epochs"])) == 0
            ):
                self._refresh_followup_cache(epoch)
            self._network.train()

            sums = defaultdict(float)
            vectors = defaultdict(list)
            gradient_rows = []
            hard_histogram = Counter()
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
            separation_active = self._separation_is_active()
            pic_active = self._cur_task > 0 and bool(self.args.get("pic_enabled", False))
            ahn_active = self._cur_task > 0 and bool(self.args.get("ahn_enabled", False))

            for batch_index, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                batch_size = int(targets.shape[0])
                outputs = self._network(inputs, label=targets)
                logits = outputs["logits"]
                features = outputs["sfeatures"]
                loss_cls = loss_cos(
                    logits[:, self._known_classes :], targets - self._known_classes
                )
                loss_scc = loss_feature = loss_logit = None
                if self._cur_task == 0:
                    original_loss = loss_cls
                else:
                    loss_scc = LobachevskyRelativeGeomLoss(
                        outputs, self._class_means, self._device
                    )
                    loss_feature = features_distill_loss(outputs)
                    loss_logit = logits_distill_loss(
                        outputs, self._class_means, self._network, T=4.0
                    )
                    if scc_active:
                        original_loss = loss_cls + float(self.args["scc_value"]) * loss_scc
                    elif feature_active:
                        original_loss = loss_cls + 0.1 * loss_feature
                    elif logit_active:
                        original_loss = loss_cls + 5.0 * loss_logit
                    else:
                        original_loss = loss_cls

                zero = features.sum() * 0.0
                separation_loss = pic_value = ahn_value = zero
                separation_result = pic_result = ahn_result = None
                if separation_active:
                    if self._separation_cache is None:
                        raise RuntimeError("active diagonal separation has no cache")
                    separation_result = diagonal_region_loss(
                        features,
                        self._separation_cache,
                        normalize=bool(self.args.get("normalized_diagonal", False)),
                        return_all=True,
                    )
                    separation_loss = separation_result.loss
                if pic_active:
                    pic_result = pic_loss(
                        features,
                        targets,
                        self._followup_old_prototypes,
                        self._followup_current_prototypes,
                        self._known_classes,
                        temperature=float(self.args.get("pic_temperature", 0.07)),
                    )
                    pic_value = pic_result.loss
                if ahn_active:
                    ahn_result = angular_hard_negative_loss(
                        features,
                        targets,
                        self._followup_old_prototypes,
                        self._followup_current_prototypes,
                        self._known_classes,
                        margin=float(self.args.get("ahn_margin", 0.5)),
                        top_k=int(self.args.get("ahn_top_k", 2)),
                    )
                    ahn_value = ahn_result.loss

                weighted_sep = float(self.args["sep_lambda"]) * separation_loss
                weighted_pic = float(self.args.get("pic_lambda", 1.0)) * pic_value
                weighted_ahn = float(self.args.get("ahn_lambda", 1.0)) * ahn_value
                loss = original_loss + weighted_sep + weighted_pic + weighted_ahn
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at task={self._cur_task + 1}, "
                        f"epoch={epoch + 1}, batch={batch_index}"
                    )

                grad_interval = int(self.args.get("diagnostic_grad_interval", 20))
                if grad_interval > 0 and batch_index % grad_interval == 0:
                    grad_metrics = component_gradient_metrics(
                        {
                            "classification": loss_cls,
                            "scc": loss_scc if scc_active else None,
                            "separation": separation_loss if separation_active else None,
                            "pic": pic_value if pic_active else None,
                            "ahn": ahn_value if ahn_active else None,
                        },
                        self._network.parameters(),
                    )
                    gradient_rows.append(
                        {
                            "experiment_id": self.args["experiment_id"],
                            "seed": self.args["seed"],
                            "task_id": self._cur_task + 1,
                            "epoch": epoch + 1,
                            "batch": batch_index,
                            **grad_metrics,
                        }
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sums["classification"] += float(loss_cls.item()) * batch_size
                if scc_active:
                    sums["scc"] += float(loss_scc.item()) * batch_size
                if feature_active:
                    sums["feature_distillation"] += float(loss_feature.item()) * batch_size
                if logit_active:
                    sums["logit_distillation"] += float(loss_logit.item()) * batch_size
                if separation_active:
                    sums["separation"] += float(separation_loss.item()) * batch_size
                    sums["weighted_separation"] += float(weighted_sep.item()) * batch_size
                if pic_active:
                    sums["pic"] += float(pic_value.item()) * batch_size
                    sums["weighted_pic"] += float(weighted_pic.item()) * batch_size
                if ahn_active:
                    sums["ahn"] += float(ahn_value.item()) * batch_size
                    sums["weighted_ahn"] += float(weighted_ahn.item()) * batch_size
                sums["original_structure"] += float(original_loss.item()) * batch_size
                sums["total"] += float(loss.item()) * batch_size
                vectors["feature_norm"].append(features.detach().norm(dim=1).cpu())

                geometry = None
                for candidate in (ahn_result, pic_result):
                    if candidate is not None:
                        geometry = candidate.diagnostics
                        break
                if geometry is None and self._cur_task > 0 and self._followup_old_prototypes is not None:
                    geometry = prototype_geometry(
                        features,
                        targets,
                        self._followup_old_prototypes,
                        self._followup_current_prototypes,
                        self._known_classes,
                    )
                if geometry is not None:
                    for key in (
                        "nearest_old_cosine",
                        "correct_new_cosine",
                        "cosine_gap",
                        "nearest_old_euclidean",
                    ):
                        vectors[key].append(geometry[key].cpu())
                if pic_result is not None:
                    vectors["pic_positive_similarity"].append(
                        pic_result.diagnostics["positive_similarity"].cpu()
                    )
                    vectors["pic_margin_violation"].append(
                        pic_result.diagnostics["margin_violation"].float().cpu()
                    )
                if ahn_result is not None:
                    vectors["ahn_margin_violation"].append(
                        ahn_result.diagnostics["margin_violation"].float().cpu()
                    )
                    vectors["ahn_margin_violation_amount"].append(
                        ahn_result.diagnostics["margin_violation_amount"].cpu()
                    )
                    for class_id in ahn_result.diagnostics["hard_negative_classes"].cpu().flatten().tolist():
                        hard_histogram[int(class_id)] += 1
                if separation_result is not None:
                    for key in (
                        "nearest_d2",
                        "nearest_rho",
                        "nearest_tau",
                        "margin_violation",
                    ):
                        vectors[key].append(separation_result.diagnostics[key].cpu())
                    vectors["intrusion"].append(
                        separation_result.diagnostics["intrusion"].float().cpu()
                    )
                    diag = separation_result.diagnostics
                    all_d2 = diag["all_d2"].cpu()
                    all_rho = diag["all_rho"].cpu()
                    region_taus = diag["region_taus"].cpu()
                    for region_index, class_id in enumerate(
                        diag["region_class_ids"].cpu().tolist()
                    ):
                        d2_values = all_d2[:, region_index]
                        rho_values = all_rho[:, region_index]
                        violations = torch.relu(region_taus[region_index] - d2_values)
                        class_diag[int(class_id)]["nearest_d2"].extend(d2_values.tolist())
                        class_diag[int(class_id)]["nearest_rho"].extend(rho_values.tolist())
                        class_diag[int(class_id)]["nearest_tau"].extend(
                            [float(region_taus[region_index])] * d2_values.numel()
                        )
                        class_diag[int(class_id)]["margin_violation"].extend(
                            violations.tolist()
                        )
                        class_diag[int(class_id)]["intrusion"].extend(
                            rho_values.lt(1.0).float().tolist()
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
                "scc_raw": sums["scc"] / processed_samples if scc_active else None,
                "scc_coefficient": float(self.args["scc_value"]) if scc_active else None,
                "scc_weighted": (
                    sums["scc"] * float(self.args["scc_value"]) / processed_samples
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
                "separation_active": separation_active,
                "separation_raw": sums["separation"] / processed_samples if separation_active else None,
                "separation_coefficient": float(self.args["sep_lambda"]) if separation_active else None,
                "separation_weighted": sums["weighted_separation"] / processed_samples if separation_active else None,
                "normalized_diagonal": bool(self.args.get("normalized_diagonal", False)),
                "pic_active": pic_active,
                "pic_raw": sums["pic"] / processed_samples if pic_active else None,
                "pic_coefficient": float(self.args.get("pic_lambda", 1.0)) if pic_active else None,
                "pic_weighted": sums["weighted_pic"] / processed_samples if pic_active else None,
                "ahn_active": ahn_active,
                "ahn_raw": sums["ahn"] / processed_samples if ahn_active else None,
                "ahn_coefficient": float(self.args.get("ahn_lambda", 1.0)) if ahn_active else None,
                "ahn_weighted": sums["weighted_ahn"] / processed_samples if ahn_active else None,
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
            # Backward-compatible names expected by the base aggregation code.
            diagnostics["student_feature_norm"] = diagnostics.get("feature_norm_mean")
            diagnostics["intrusion_ratio"] = diagnostics.get("intrusion_mean")
            if gradient_rows:
                averaged_gradients = mean_numeric_rows(
                    [
                        {
                            key: value
                            for key, value in row.items()
                            if key.startswith("grad_")
                        }
                        for row in gradient_rows
                    ]
                )
                diagnostics.update({f"epoch_{key}": value for key, value in averaged_gradients.items()})
                upsert_csv_rows(
                    self.run_dir / "gradient_metrics.csv",
                    gradient_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "batch"),
                )
            append_jsonl(self.run_dir / "training_diagnostics.jsonl", diagnostics)
            upsert_csv_rows(
                self.run_dir / "epoch_metrics.csv",
                [diagnostics],
                ("experiment_id", "seed", "task_id", "stage", "epoch"),
            )

            if hard_histogram:
                histogram_rows = [
                    {
                        "experiment_id": self.args["experiment_id"],
                        "seed": self.args["seed"],
                        "task_id": self._cur_task + 1,
                        "epoch": epoch + 1,
                        "old_class_id": class_id,
                        "selection_count": count,
                    }
                    for class_id, count in sorted(hard_histogram.items())
                ]
                upsert_csv_rows(
                    self.run_dir / "hard_negative_histogram.csv",
                    histogram_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "old_class_id"),
                )
            if self._adaptive_rows and class_diag:
                adaptive_lookup = {int(row["class_id"]): row for row in self._adaptive_rows}
                class_rows = []
                for class_id, values in sorted(class_diag.items()):
                    base = adaptive_lookup.get(class_id, {})
                    row = dict(base)
                    row.update(
                        {
                            "experiment_id": self.args["experiment_id"],
                            "seed": self.args["seed"],
                            "task_id": self._cur_task + 1,
                            "epoch": epoch + 1,
                            "class_id": class_id,
                            "evaluated_new_sample_count": len(values["nearest_rho"]),
                            "intrusion_ratio": float(np.mean(values["intrusion"])),
                            "active_separation_ratio": float(np.mean(values["intrusion"])),
                            "mean_separation_penalty": float(np.mean(values["margin_violation"])),
                            "current_d2_mean": float(np.mean(values["nearest_d2"])),
                        }
                    )
                    class_rows.append(row)
                upsert_csv_rows(
                    self.run_dir / "adaptive_class_diagnostics.csv",
                    class_rows,
                    ("experiment_id", "seed", "task_id", "epoch", "class_id"),
                )

            train_acc = np.around(
                tensor2numpy(correct) * 100 / processed_samples, decimals=2
            )
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, "
                "Test_accy {:.2f}"
            ).format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                diagnostics["total_loss"],
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)
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

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        if self._cur_task > 0 and float(self.args.get("ceos_replay_fraction", 0.0)) > 0.0:
            from .stage2 import run_ceos_classifier_retraining

            return run_ceos_classifier_retraining(self, task_size, ca_epochs)
        return super()._stage2_compact_classifier(task_size, ca_epochs)

    def _update_region_statistics(self, data_manager, new_means_after, new_covariances_after):
        """Keep base region storage, then add method-correct task-end E3 metrics."""
        super()._update_region_statistics(data_manager, new_means_after, new_covariances_after)
        if not self._separation_is_active():
            return
        old_regions = [
            self._single_regions[class_id][0] for class_id in range(self._known_classes)
        ]
        feature_space = old_regions[0]["feature_space"]
        cache = SeparationCache(
            old_regions,
            self.args["separation_covariance"],
            feature_space,
            self.args,
        )
        birth_tasks = torch.tensor(
            [int(region["birth_task"]) for region in old_regions], dtype=torch.long
        )
        corrected_means = torch.stack(
            [region["mean"].detach().cpu().float() for region in old_regions]
        )
        source_means = torch.as_tensor(self._task_source_means).cpu().float()
        adaptive = recalibrate_taus(
            birth_taus=cache.taus.cpu(),
            variances=cache.factors.cpu(),
            class_ids=cache.class_ids.cpu(),
            birth_tasks=birth_tasks,
            current_task=self._cur_task,
            drift_norms=torch.linalg.vector_norm(corrected_means - source_means, dim=1),
            mode=str(self.args.get("adaptive_margin_mode", "none")),
            strength=float(self.args.get("adaptive_margin_strength", 0.25)),
            min_scale=float(self.args.get("adaptive_margin_min_scale", 0.5)),
            max_scale=float(self.args.get("adaptive_margin_max_scale", 2.0)),
        )
        cache.taus = adaptive.taus.to(cache.taus.device)
        rng_state = capture_rng_state()
        try:
            features, _ = self._collect_current_features("sfeatures")
        finally:
            restore_rng_state(rng_state)
        chunks = defaultdict(list)
        with torch.no_grad():
            batch_size = int(self.args["stats_batch_size"])
            for start in range(0, features.shape[0], batch_size):
                result = diagonal_region_loss(
                    features[start : start + batch_size].to(self._device),
                    cache,
                    normalize=bool(self.args.get("normalized_diagonal", False)),
                )
                for key, value in result.diagnostics.items():
                    if key != "nearest_old_class":
                        chunks[key].append(value.float().cpu())
        record = {
            "experiment_id": self.args["experiment_id"],
            "seed": self.args["seed"],
            "task_id": self._cur_task + 1,
            "stage": "task_end_followup_separation",
            "feature_space": feature_space,
            "covariance_view": self.args["separation_covariance"],
            "normalized_diagonal": bool(self.args.get("normalized_diagonal", False)),
            "adaptive_margin_mode": self.args.get("adaptive_margin_mode", "none"),
            "region_count": cache.region_count,
        }
        for key, values in chunks.items():
            record.update(scalar_tensor_summary(torch.cat(values), key))
        record["intrusion_ratio"] = record.get("intrusion_mean")
        append_jsonl(self.run_dir / "followup_task_end_diagnostics.jsonl", record)
        self._separation_cache = cache
