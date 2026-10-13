"""Structure classifier retraining with a controlled CEOS replay fraction."""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F

from utils.experiment_io import update_json, upsert_csv_rows, utc_now_iso

from .ceos import sample_ceos
from .losses import scalar_tensor_summary


CEOS_SAMPLE_COLUMNS = (
    "experiment_id",
    "seed",
    "task_id",
    "epoch",
    "sample_id",
    "source_old_class",
    "nearest_new_enemy_class",
    "interpolation_coefficient",
    "distance_old_to_generated",
    "distance_generated_to_enemy",
    "cosine_to_old_prototype",
    "cosine_to_enemy",
    "old_dominance_constraint",
    "old_class_logit",
    "enemy_class_logit",
    "old_new_logit_margin",
    "boundary_proximity_abs_logit_gap",
    "predicted_as_source_old",
)


def _write_sample_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CEOS_SAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _gaussian_samples(learner, class_id: int, count: int, mean_scale: float) -> torch.Tensor:
    if count == 0:
        return torch.empty((0, learner._class_covs.shape[-1]), device=learner._device)
    mean = torch.as_tensor(
        learner._class_means[class_id], dtype=torch.float32, device=learner._device
    ) * float(mean_scale)
    covariance = learner._class_covs[class_id].to(learner._device).float()
    return MultivariateNormal(mean, covariance).sample((int(count),))


def run_ceos_classifier_retraining(learner, task_size: int, ca_epochs: int = 5):
    """Replace a fixed fraction of old-class Gaussian samples with CEOS.

    Current classes remain Gaussian.  For every class and epoch the total is
    exactly ``replay_samples_per_class``; CEOS50/CEOS100 therefore do not gain
    extra optimization samples over the Structure baseline.
    """
    fraction = float(learner.args["ceos_replay_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("ceos_replay_fraction must be in (0, 1]")
    model = learner._network.module if isinstance(learner._network, nn.DataParallel) else learner._network
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    parameters = [parameter for parameter in model.fc.parameters() if parameter.requires_grad]
    optimizer = optim.SGD(
        parameters,
        lr=learner.init_lr,
        momentum=0.9,
        weight_decay=learner.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ca_epochs)
    model.to(learner._device)
    model.eval()

    current_features, current_labels = learner._collect_current_features("sfeatures")
    current_features = current_features.to(learner._device).float()
    current_labels = current_labels.to(learner._device).long()
    class_count = learner._total_classes
    samples_per_class = int(learner.args["replay_samples_per_class"])
    ceos_per_old_class = int(round(samples_per_class * fraction))
    gaussian_per_old_class = samples_per_class - ceos_per_old_class

    for epoch in range(ca_epochs):
        epoch_started = time.perf_counter()
        learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
        sampled_data = []
        sampled_labels = []
        sample_rows = []
        sample_id = 0
        ceos_gap_values = []
        ceos_constraint_values = []
        ceos_predicted_values = []

        for class_id in range(class_count):
            task_id = class_id // task_size
            mean_scale = 0.9 + (task_id + 1) / (learner._cur_task + 1) * 0.1
            is_old = class_id < learner._known_classes
            gaussian_count = gaussian_per_old_class if is_old else samples_per_class
            gaussian = _gaussian_samples(learner, class_id, gaussian_count, mean_scale)
            sampled_data.append(gaussian)
            sampled_labels.extend([class_id] * gaussian_count)

            if not is_old:
                continue
            old_prototype = torch.as_tensor(
                learner._class_means[class_id],
                dtype=torch.float32,
                device=learner._device,
            ) * float(mean_scale)
            ceos = sample_ceos(
                old_prototype,
                current_features,
                current_labels,
                ceos_per_old_class,
                lambda_min=float(learner.args.get("ceos_lambda_min", 0.7)),
                enemy_top_k=int(learner.args.get("ceos_enemy_top_k", 1)),
            )
            sampled_data.append(ceos.features)
            sampled_labels.extend([class_id] * ceos_per_old_class)
            with torch.no_grad():
                logits = float(learner.args["scale"]) * model.ca_forward(ceos.features)["logits"]
                row_indices = torch.arange(ceos_per_old_class, device=learner._device)
                old_logits = logits[:, class_id]
                enemy_logits = logits[row_indices, ceos.enemy_labels]
                gaps = old_logits - enemy_logits
                predicted = logits[:, :class_count].argmax(dim=1).eq(class_id)
            ceos_gap_values.append(gaps.detach().cpu())
            ceos_constraint_values.append(ceos.old_dominance.float().detach().cpu())
            ceos_predicted_values.append(predicted.float().detach().cpu())
            if bool(learner.args.get("ceos_log_samples", True)):
                for index in range(ceos_per_old_class):
                    sample_rows.append(
                        {
                            "experiment_id": learner.args["experiment_id"],
                            "seed": learner.args["seed"],
                            "task_id": learner._cur_task + 1,
                            "epoch": epoch + 1,
                            "sample_id": sample_id,
                            "source_old_class": class_id,
                            "nearest_new_enemy_class": int(ceos.enemy_labels[index]),
                            "interpolation_coefficient": float(ceos.lambdas[index]),
                            "distance_old_to_generated": float(ceos.distance_from_old[index]),
                            "distance_generated_to_enemy": float(ceos.distance_to_enemy[index]),
                            "cosine_to_old_prototype": float(ceos.cosine_to_old[index]),
                            "cosine_to_enemy": float(ceos.cosine_to_enemy[index]),
                            "old_dominance_constraint": bool(ceos.old_dominance[index]),
                            "old_class_logit": float(old_logits[index]),
                            "enemy_class_logit": float(enemy_logits[index]),
                            "old_new_logit_margin": float(gaps[index]),
                            "boundary_proximity_abs_logit_gap": float(gaps[index].abs()),
                            "predicted_as_source_old": bool(predicted[index]),
                        }
                    )
                    sample_id += 1

        features = torch.cat(sampled_data, dim=0).float().to(learner._device)
        labels = torch.tensor(sampled_labels, dtype=torch.long, device=learner._device)
        expected = class_count * samples_per_class
        if features.shape[0] != expected or labels.shape[0] != expected:
            raise AssertionError("CEOS altered the controlled replay sample total")
        permutation = torch.randperm(expected, device=learner._device)
        features, labels = features[permutation], labels[permutation]

        loss_sample_sum = 0.0
        processed_samples = 0
        for step in range(class_count):
            start = step * samples_per_class
            stop = (step + 1) * samples_per_class
            inputs = features[start:stop]
            targets = labels[start:stop]
            logits = float(learner.args["scale"]) * model.ca_forward(inputs)["logits"]
            if learner.logit_norm is not None:
                per_task_norm = []
                begin = 0
                end = 0
                for task_index in range(learner._cur_task + 1):
                    end += learner.task_sizes[task_index]
                    per_task_norm.append(
                        torch.norm(logits[:, begin:end], p=2, dim=-1, keepdim=True) + 1e-7
                    )
                    begin = end
                normalization = torch.cat(per_task_norm, dim=-1).mean(dim=-1, keepdim=True)
                used_logits = logits[:, :class_count] / normalization / learner.logit_norm
            else:
                used_logits = logits[:, :class_count]
            loss = F.cross_entropy(used_logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite CEOS classifier loss task={learner._cur_task + 1}, "
                    f"epoch={epoch + 1}, step={step}"
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(targets.shape[0])
            loss_sample_sum += float(loss.item()) * batch_size
            processed_samples += batch_size

        scheduler.step()
        mean_loss = loss_sample_sum / processed_samples
        gap_tensor = torch.cat(ceos_gap_values)
        constraint_tensor = torch.cat(ceos_constraint_values)
        predicted_tensor = torch.cat(ceos_predicted_values)
        row = {
            "experiment_id": learner.args["experiment_id"],
            "seed": learner.args["seed"],
            "task_id": learner._cur_task + 1,
            "stage": "classifier_retraining",
            "epoch": epoch + 1,
            "classification_active": True,
            "classification_raw": mean_loss,
            "classification_coefficient": 1.0,
            "classification_weighted": mean_loss,
            "scc_active": False,
            "separation_active": False,
            "pic_active": False,
            "ahn_active": False,
            "original_structure_loss": mean_loss,
            "total_loss": mean_loss,
            "ceos_active": True,
            "ceos_replay_fraction": fraction,
            "ceos_samples_per_old_class": ceos_per_old_class,
            "gaussian_samples_per_old_class": gaussian_per_old_class,
            "total_samples_per_class": samples_per_class,
            "ceos_old_dominance_rate": float(constraint_tensor.mean()),
            "ceos_predicted_as_source_old_rate": float(predicted_tensor.mean()),
            "learning_rate": learning_rates[0],
            "learning_rates": learning_rates,
            "processed_samples": processed_samples,
            "optimizer_steps": class_count,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "nan_or_inf_detected": False,
            "checkpoint_path": None,
            "checkpoint_state_id": None,
            "pending_checkpoint_state_id": f"task_end:task_{learner._cur_task + 1}",
            **scalar_tensor_summary(gap_tensor, "ceos_old_new_logit_margin"),
            **scalar_tensor_summary(gap_tensor.abs(), "ceos_boundary_proximity"),
        }
        upsert_csv_rows(
            learner.run_dir / "epoch_metrics.csv",
            [row],
            ("experiment_id", "seed", "task_id", "stage", "epoch"),
        )
        upsert_csv_rows(
            learner.run_dir / "ceos_summary.csv",
            [row],
            ("experiment_id", "seed", "task_id", "epoch"),
        )
        if sample_rows:
            _write_sample_rows(
                learner.run_dir
                / "ceos_samples"
                / f"task_{learner._cur_task + 1}_epoch_{epoch + 1}.csv",
                sample_rows,
            )
        update_json(
            learner.run_dir / "run_metadata.json",
            {
                "updated_at_utc": utc_now_iso(),
                "last_recorded_stage": "classifier_retraining",
                "last_recorded_task": learner._cur_task + 1,
                "last_recorded_epoch": epoch + 1,
                "pending_checkpoint_state_id": f"task_end:task_{learner._cur_task + 1}",
            },
        )
        test_accuracy = learner._compute_accuracy(learner._network, learner.test_loader)
        logging.info(
            "CEOS CA Task %s Epoch %s => Loss %.3f, Test_accy %.3f",
            learner._cur_task,
            epoch + 1,
            mean_loss,
            test_accuracy,
        )
