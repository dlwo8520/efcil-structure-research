from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[3])))

import copy
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from .data import Cars196IncrementalData, CIFAR100IncrementalData, make_loader
from .experiment import write_csv
from .metrics import build_class_rows, evaluate_seen, summarize_task_metrics
from .pcds import (
    FeatureStatisticsAccumulator,
    PrototypeMemory,
    adaptive_prototype_rectification,
    stable_covariance_factor,
    structural_consistency_loss,
)
from .pcds_model import PCDSContinualViT
from .utils import (
    assert_physical_gpu,
    collect_environment,
    count_parameters,
    dump_json,
    dump_yaml,
    load_config,
    resolve_class_order,
    set_seed,
    setup_logger,
    tensor_l2_norm,
)


METHOD_FLAGS = {
    "baseline": {"apr": False, "scc": False, "ca": False},
    "apr_only": {"apr": True, "scc": False, "ca": False},
    "scc_only": {"apr": False, "scc": True, "ca": False},
    "apr_scc": {"apr": True, "scc": True, "ca": False},
    "pcds_full": {"apr": True, "scc": True, "ca": True},
    "pcds_structure_sgds": {"apr": True, "scc": True, "ca": True},
}
METHODS = set(METHOD_FLAGS)


def resolve_task_sizes(config: dict[str, Any]) -> list[int]:
    if "task_sizes" in config:
        sizes = [int(value) for value in config["task_sizes"]]
    else:
        sizes = [int(config["classes_per_task"])] * int(config["num_tasks"])
    if len(sizes) != int(config["num_tasks"]) or any(size <= 0 for size in sizes):
        raise ValueError("task_sizes must contain one positive entry per task")
    expected_classes = int(config.get("num_classes", sum(sizes)))
    if sum(sizes) != expected_classes:
        raise ValueError(
            f"task_sizes cover {sum(sizes)} classes, expected {expected_classes}"
        )
    return sizes


def epochs_for_task(config: dict[str, Any], task_id: int) -> int:
    if "initial_epochs" in config or "incremental_epochs" in config:
        epochs = int(
            config["initial_epochs"] if task_id == 0 else config["incremental_epochs"]
        )
    else:
        epochs = int(config["epochs_per_task"])
    if epochs <= 0:
        raise ValueError("task epochs must be positive")
    return epochs


def validate_protocol(
    config: dict[str, Any], repo_root: Path, require_gpu: bool
) -> dict[str, Any]:
    errors: list[str] = []
    dataset = str(config.get("dataset"))
    if dataset not in {"cifar100", "cars196"}:
        errors.append("dataset must be cifar100 or cars196")
    if int(config.get("num_tasks", -1)) != 10:
        errors.append("num_tasks must be 10")
    if int(config.get("seed", -1)) != 0:
        errors.append("primary PCDS comparison must use seed 0")
    if bool(config.get("raw_replay", False)):
        errors.append("raw replay must be disabled")
    if bool(config.get("task_id_masking", False)):
        errors.append("Task-ID masking must be disabled")
    if str(config.get("model_name")) != "vit_base_patch16_224.augreg_in21k":
        errors.append("backbone must match the KeepLoRA ViT-B/16-IN21K checkpoint")

    if dataset == "cifar100":
        if int(config.get("classes_per_task", -1)) != 10:
            errors.append("classes_per_task must be 10")
        if int(config.get("num_tasks", 0)) * int(
            config.get("classes_per_task", 0)
        ) != 100:
            errors.append("task split must cover exactly 100 classes")
        if int(config.get("epochs_per_task", -1)) != 10:
            errors.append("primary PCDS comparison must use 10 epochs per task")
        if str(config.get("class_order")) != "god_seed1993":
            errors.append("class_order must reuse god_seed1993")

        keep_config_path = repo_root / str(config["keep_lora_config"])
        if not keep_config_path.exists():
            errors.append(f"KeepLoRA comparison config does not exist: {keep_config_path}")
        else:
            keep_config = load_config(keep_config_path)
            comparisons = {
                "data_root": (str(config["data_root"]), str(keep_config["data_root"])),
                "model_name": (str(config["model_name"]), str(keep_config["model_name"])),
                "num_tasks": (int(config["num_tasks"]), int(keep_config["num_tasks"])),
                "classes_per_task": (
                    int(config["classes_per_task"]),
                    int(keep_config["classes_per_task"]),
                ),
                "seed": (int(config["seed"]), int(keep_config["seed"])),
                "class_order": (str(config["class_order"]), str(keep_config["class_order"])),
            }
            errors.extend(
                f"{key} differs from KeepLoRA: {left!r} != {right!r}"
                for key, (left, right) in comparisons.items()
                if left != right
            )

        class_order = resolve_class_order(config["class_order"], num_classes=100)
        keep_order_path = repo_root / str(config["keep_lora_class_order"])
        if keep_order_path.exists():
            with keep_order_path.open("r", encoding="utf-8") as handle:
                keep_order = [int(value) for value in json.load(handle)]
            if class_order != keep_order:
                errors.append("resolved class order differs from the completed KeepLoRA run")
        else:
            errors.append(f"completed KeepLoRA class order is missing: {keep_order_path}")
        task_sizes = [10] * 10
        dataset_label = "CIFAR-100"
        data_description = "canonical CIFAR-100 train/test"
    else:
        try:
            task_sizes = resolve_task_sizes(config)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            task_sizes = []
        if task_sizes and task_sizes != [20] * 9 + [16]:
            errors.append("Cars-196 CDS 10-task split must be 20x9 + 16")
        if int(config.get("num_classes", -1)) != 196:
            errors.append("Cars-196 must contain 196 classes")
        if str(config.get("class_order")) != "cds_cars196_seed1993":
            errors.append("Cars class order must use released CDS shuffle seed 1993")
        if int(config.get("class_order_seed", -1)) != 1993:
            errors.append("class_order_seed must be 1993 and separate from training seed")
        if int(config.get("initial_epochs", -1)) != 70:
            errors.append("Cars official config requires 70 initial epochs")
        if int(config.get("incremental_epochs", -1)) != 40:
            errors.append("Cars official config requires 40 incremental epochs")
        official_values = {
            "batch_size": 32,
            "learning_rate": 0.005,
            "weight_decay": 0.0005,
            "scc_weight": 5.0,
            "apr_gamma": 10.0,
            "apr_rho": 10.0,
            "classifier_margin": 0.3,
            "adapter_placement": "attention",
        }
        for key, expected in official_values.items():
            if config.get(key) != expected:
                errors.append(
                    f"Cars official setting {key} must be {expected!r}, "
                    f"found {config.get(key)!r}"
                )
        try:
            resolve_class_order(config["class_order"], num_classes=196)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
        dataset_label = "Stanford Cars-196"
        data_description = "official train/test ImageFolder split"

    data_root = Path(config["data_root"])
    if not ((data_root / "train").is_dir() and (data_root / "test").is_dir()):
        errors.append(f"local {dataset_label} train/test directories are missing: {data_root}")
    if errors:
        raise RuntimeError("PCDS protocol validation failed:\n- " + "\n- ".join(errors))
    physical_gpu = int(config.get("physical_gpu", 1))
    expected_uuid = str(
        config.get(
            "physical_gpu_uuid",
            "afb08c1c-8a66-104a-01d1-046df70f0552",
        )
    )
    gpu_info = (
        assert_physical_gpu(physical_gpu, expected_uuid) if require_gpu else {}
    )
    return {
        "dataset": dataset_label,
        "tasks": int(config["num_tasks"]),
        "task_sizes": task_sizes,
        "classes": sum(task_sizes),
        "class_order": str(config["class_order"]),
        "class_order_seed": config.get("class_order_seed"),
        "data_protocol": data_description,
        "class_order_matches_keep_lora": dataset == "cifar100",
        "single_head_all_seen_classes": True,
        "raw_replay": False,
        "old_train_data_access": False,
        "synthetic_features_train_classifier_only": True,
        "sgds_activation_guidance": bool(config.get("use_sgds", False)),
        "gpu": gpu_info,
    }


@torch.no_grad()
def collect_current_task_statistics(
    feature_function: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
    amp: bool,
    begin_class: int,
    num_classes: int,
    feature_dim: int,
    covariance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulator = FeatureStatisticsAccumulator(
        begin_class, num_classes, feature_dim, device
    )
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            features = feature_function(images)
        accumulator.update(features, labels)
    return accumulator.finalize(covariance_epsilon)


@torch.no_grad()
def collect_current_task_means(
    feature_function: Callable[[torch.Tensor], torch.Tensor],
    loader,
    device: torch.device,
    amp: bool,
    begin_class: int,
    num_classes: int,
    feature_dim: int,
) -> torch.Tensor:
    """Current-task means without retaining images or individual features."""
    sums = torch.zeros(num_classes, feature_dim, device=device, dtype=torch.float64)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        local_labels = labels.to(device, non_blocking=True).long() - begin_class
        if bool((local_labels < 0).any()) or bool((local_labels >= num_classes).any()):
            raise RuntimeError("Semantic prototype loader crossed the current-task boundary")
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            features = feature_function(images)
        sums.index_add_(0, local_labels, features.detach().double())
        counts.index_add_(0, local_labels, torch.ones_like(local_labels))
    if bool((counts == 0).any()):
        raise RuntimeError("Semantic prototype collection missed a current class")
    return (sums / counts[:, None]).float().cpu()


@torch.no_grad()
def covariance_effective_rank_rows(
    covariances: torch.Tensor,
    counts: torch.Tensor,
    task_id: int,
    begin_class: int,
    epsilon: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Exact r_0.99 from covariance eigen-energy of centered class features."""
    covariance = covariances.to(device=device, dtype=torch.float32)
    identity = torch.eye(covariance.shape[-1], device=device).unsqueeze(0)
    covariance = covariance - float(epsilon) * identity
    covariance = 0.5 * (covariance + covariance.transpose(1, 2))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min_(0.0).flip(dims=(1,))
    rows: list[dict[str, Any]] = []
    for local_id in range(covariance.shape[0]):
        energy = eigenvalues[local_id]
        total = energy.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            raise FloatingPointError("Invalid covariance spectrum for effective rank")
        cumulative = torch.cumsum(energy, dim=0) / total
        rank_99 = int(torch.searchsorted(cumulative, 0.99).item()) + 1
        numerical_rank = int(
            (energy > torch.finfo(energy.dtype).eps * energy[0] * energy.numel())
            .sum()
            .item()
        )
        rows.append({
            "task_id": task_id,
            "class_id": begin_class + local_id,
            "samples": int(counts[local_id].item()),
            "effective_rank_99": rank_99,
            "numerical_rank": numerical_rank,
            "feature_dim": int(covariance.shape[-1]),
        })
    return rows


def _snapshot_backbone(model: PCDSContinualViT) -> nn.Module:
    reference = copy.deepcopy(model.backbone).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise RuntimeError("SCC reference backbone was not completely frozen")
    return reference


def train_task(
    model: PCDSContinualViT,
    reference_backbone: nn.Module | None,
    old_prototype_means: torch.Tensor,
    loader,
    device: torch.device,
    config: dict[str, Any],
    method: str,
    task_id: int,
    logger,
) -> tuple[list[dict], dict[str, float], float]:
    flags = METHOD_FLAGS[method]
    task_epochs = epochs_for_task(config, task_id)
    scc_enabled = bool(flags["scc"] and task_id > 0)
    if scc_enabled and reference_backbone is None:
        raise RuntimeError("SCC requires the immediately preceding frozen backbone")
    parameters = model.adapter_parameters() + model.classifier.current_parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(config["learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=task_epochs,
        eta_min=float(config["min_learning_rate"]),
    )
    amp = bool(config["amp"])
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
        init_scale=float(config.get("grad_scaler_init_scale", 256.0)),
        growth_interval=int(config.get("grad_scaler_growth_interval", 100000)),
    )
    rows: list[dict] = []
    started = time.time()
    task_step = 0
    model.train()
    if reference_backbone is not None:
        reference_backbone.eval()
    for epoch in range(task_epochs):
        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                if reference_backbone is None:
                    reference_features = torch.empty(
                        images.shape[0], model.feature_dim, device=device
                    )
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=amp
                    ):
                        reference_features = reference_backbone(images)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = model(
                    images,
                    labels=(labels if model.activation_guidance is not None else None),
                    epoch=epoch,
                )
                training_logits = output["logits"]
                classifier_margin = float(config.get("classifier_margin", 0.0))
                if classifier_margin:
                    training_logits = training_logits.clone()
                    row_ids = torch.arange(labels.shape[0], device=labels.device)
                    training_logits[row_ids, labels] -= (
                        float(config["classifier_scale"]) * classifier_margin
                    )
                ce_loss = F.cross_entropy(training_logits, labels)
                if scc_enabled:
                    scc_loss = structural_consistency_loss(
                        output["features"], reference_features, old_prototype_means
                    )
                else:
                    scc_loss = output["features"].sum() * 0.0
                total_loss = ce_loss + float(config["scc_weight"]) * scc_loss
            if not bool(torch.isfinite(total_loss)):
                raise FloatingPointError(
                    f"Non-finite task loss at task={task_id} epoch={epoch} batch={batch_index}"
                )
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            adapter_gradient_norm = tensor_l2_norm([
                parameter.grad
                for parameter in model.adapter_parameters()
                if parameter.grad is not None
            ])
            classifier_gradient_norm = tensor_l2_norm([
                parameter.grad
                for parameter in model.classifier.current_parameters()
                if parameter.grad is not None
            ])
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = int(amp and scale_after < scale_before)
            if skipped and bool(config.get("fail_on_optimizer_skip", True)):
                raise FloatingPointError(
                    f"GradScaler skipped task={task_id} step={task_step}: "
                    f"{scale_before} -> {scale_after}"
                )
            row = {
                "method": method,
                "seed": int(config["seed"]),
                "task_id": task_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "task_step": task_step,
                "ce_loss": float(ce_loss.detach().item()),
                "scc_loss": float(scc_loss.detach().item()),
                "total_loss": float(total_loss.detach().item()),
                "scc_enabled": int(scc_enabled),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "batch_accuracy": float(
                    output["logits"].argmax(dim=1).eq(labels).float().mean().item()
                ),
                "adapter_gradient_norm": adapter_gradient_norm,
                "classifier_gradient_norm": classifier_gradient_norm,
                "optimizer_step_skipped": skipped,
                "grad_scaler_scale_before": scale_before,
                "grad_scaler_scale_after": scale_after,
            }
            rows.append(row)
            if task_step % int(config["log_every_steps"]) == 0:
                logger.info(
                    "task=%d step=%d ce=%.4f scc=%.5f total=%.4f acc=%.3f",
                    task_id,
                    task_step,
                    row["ce_loss"],
                    row["scc_loss"],
                    row["total_loss"],
                    row["batch_accuracy"],
                )
            task_step += 1
        scheduler.step()
    summary = {
        "mean_ce_loss": sum(row["ce_loss"] for row in rows) / len(rows),
        "final_ce_loss": rows[-1]["ce_loss"],
        "mean_scc_loss": sum(row["scc_loss"] for row in rows) / len(rows),
        "final_scc_loss": rows[-1]["scc_loss"],
        "mean_total_loss": sum(row["total_loss"] for row in rows) / len(rows),
        "final_total_loss": rows[-1]["total_loss"],
        "optimizer_skipped_steps": float(
            sum(row["optimizer_step_skipped"] for row in rows)
        ),
        "nonfinite_steps": 0.0,
    }
    return rows, summary, time.time() - started


def adapt_classifier_from_prototypes(
    model: PCDSContinualViT,
    memory: PrototypeMemory,
    config: dict[str, Any],
    task_id: int,
    device: torch.device,
    logger,
) -> tuple[list[dict], dict[str, float]]:
    model.eval()
    model.train_all_classifier_rows()
    parameters = model.classifier_all_parameters()
    optimizer = torch.optim.SGD(
        parameters,
        lr=float(config["ca_learning_rate"]),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["ca_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    means = memory.means.to(device=device, dtype=torch.float32)
    covariances = memory.covariances.to(device=device, dtype=torch.float32)
    factors: list[torch.Tensor] = []
    jitters: list[float] = []
    fallback_count = 0
    for class_id in range(memory.num_classes):
        factor, jitter, method_name = stable_covariance_factor(
            covariances[class_id], float(config["ca_covariance_jitter"])
        )
        factors.append(factor)
        jitters.append(jitter)
        fallback_count += int(method_name != "cholesky")
    factor_matrix = torch.stack(factors, dim=0)
    samples_per_class = int(config["ca_samples_per_class"])
    batch_size = int(config["ca_batch_size"])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + (task_id + 1) * 100003)
    rows: list[dict] = []
    for epoch in range(epochs):
        samples: list[torch.Tensor] = []
        for class_id in range(memory.num_classes):
            noise = torch.randn(
                samples_per_class,
                model.feature_dim,
                device=device,
                generator=generator,
            )
            samples.append(
                means[class_id]
                + noise @ factor_matrix[class_id].transpose(0, 1)
            )
        features = torch.cat(samples, dim=0)
        labels = torch.arange(memory.num_classes, device=device).repeat_interleave(
            samples_per_class
        )
        permutation = torch.randperm(features.shape[0], device=device, generator=generator)
        features = features[permutation]
        labels = labels[permutation]
        losses: list[float] = []
        correct = 0
        for begin in range(0, features.shape[0], batch_size):
            batch_features = features[begin : begin + batch_size]
            batch_labels = labels[begin : begin + batch_size]
            logits = model.classifier(batch_features)
            loss = F.cross_entropy(logits, batch_labels)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite classifier-adaptation loss task={task_id} epoch={epoch}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
            correct += int(logits.argmax(dim=1).eq(batch_labels).sum().item())
        scheduler.step()
        row = {
            "task_id": task_id,
            "epoch": epoch,
            "classes": memory.num_classes,
            "samples_per_class": samples_per_class,
            "synthetic_samples": int(features.shape[0]),
            "mean_loss": sum(losses) / len(losses),
            "synthetic_accuracy": correct / int(features.shape[0]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "feature_extractor_optimized": 0,
        }
        rows.append(row)
        logger.info(
            "CA task=%d epoch=%d/%d loss=%.4f synthetic_acc=%.3f",
            task_id,
            epoch + 1,
            epochs,
            row["mean_loss"],
            row["synthetic_accuracy"],
        )
    return rows, {
        "ca_mean_loss": sum(row["mean_loss"] for row in rows) / len(rows),
        "ca_final_loss": rows[-1]["mean_loss"],
        "ca_final_synthetic_accuracy": rows[-1]["synthetic_accuracy"],
        "ca_max_added_jitter": max(jitters, default=0.0),
        "ca_eigh_fallback_classes": float(fallback_count),
    }


def _accuracy_parts(
    evaluation, task_sizes: list[int], task_id: int
) -> dict[str, float]:
    new_begin = sum(task_sizes[:task_id])
    current_size = task_sizes[task_id]
    new_end = new_begin + current_size
    return {
        "seen_accuracy": evaluation.seen_accuracy,
        "old_accuracy": (
            sum(evaluation.class_correct[:new_begin])
            / sum(evaluation.class_count[:new_begin])
            if new_begin
            else float("nan")
        ),
        "new_accuracy": (
            sum(evaluation.class_correct[new_begin:new_end])
            / sum(evaluation.class_count[new_begin:new_end])
        ),
    }


def _load_checkpoint(
    path: Path,
    model: PCDSContinualViT,
    memory: PrototypeMemory,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    model.classifier.load_weights(state["classifier_blocks"], device)
    model.classifier.freeze_all()
    memory.load(state["prototype_memory"])
    return state


def run_experiment(
    config: dict[str, Any],
    method: str,
    repo_root: Path,
    resume: bool = False,
    activation_guidance_factory: Callable[[dict[str, Any], int], nn.Module] | None = None,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose one of {sorted(METHODS)}")
    protocol = validate_protocol(config, repo_root, require_gpu=True)
    flags = METHOD_FLAGS[method]
    seed = int(config["seed"])
    output_dir = (
        Path(config["output_root"])
        / str(config["experiment_name"])
        / f"seed_{seed}"
        / method
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    completed_path = output_dir / "COMPLETED.json"
    if completed_path.exists():
        raise RuntimeError(f"Run is already complete: {output_dir}")
    if any(output_dir.iterdir()) and not resume:
        raise RuntimeError(
            f"Output directory is non-empty; pass --resume or choose a new experiment: {output_dir}"
        )
    if resume and any(output_dir.iterdir()) and not checkpoint_path.exists():
        raise RuntimeError(
            f"Cannot resume because the task-boundary checkpoint is missing: {checkpoint_path}"
        )
    logger = setup_logger(output_dir)
    logger.addHandler(logging.FileHandler(output_dir / "training.log", encoding="utf-8"))
    set_seed(seed)
    device = torch.device("cuda:0")
    task_sizes = resolve_task_sizes(config)
    num_classes = sum(task_sizes)
    class_order = resolve_class_order(config["class_order"], num_classes=num_classes)
    dump_json(output_dir / "config.json", config)
    dump_yaml(output_dir / "resolved_config.yaml", config)
    dump_json(output_dir / "class_order.json", class_order)
    environment = collect_environment(repo_root, protocol["gpu"])
    dump_json(output_dir / "environment.json", environment)
    (output_dir / "environment.txt").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    dump_json(output_dir / "protocol_check.json", protocol)
    (output_dir / "command.txt").write_text(
        f"CUDA_VISIBLE_DEVICES={int(config.get('physical_gpu', 1))} "
        + " ".join(sys.argv)
        + "\n",
        encoding="utf-8",
    )
    logger.info("method=%s flags=%s output=%s", method, flags, output_dir)
    logger.info("protocol=%s", protocol)

    data_class = (
        Cars196IncrementalData
        if str(config["dataset"]) == "cars196"
        else CIFAR100IncrementalData
    )
    data = data_class(
        root=config["data_root"],
        class_order=class_order,
        image_size=int(config["image_size"]),
    )
    activation_guidance = (
        activation_guidance_factory(config, 768)
        if activation_guidance_factory is not None
        else None
    )
    if bool(config.get("use_sgds", False)) != (activation_guidance is not None):
        raise RuntimeError("use_sgds and activation-guidance factory disagree")
    model = PCDSContinualViT(
        model_name=config["model_name"],
        pretrained=bool(config["pretrained"]),
        classifier_scale=float(config["classifier_scale"]),
        adapter_bottleneck=int(config["adapter_bottleneck"]),
        adapter_dropout=float(config["adapter_dropout"]),
        adapter_placement=str(config.get("adapter_placement", "mlp")),
        activation_guidance=activation_guidance,
        activation_guidance_layer=int(config.get("sgds_target_layers", [-1])[0]),
    ).to(device)
    if model.feature_dim != 768:
        raise RuntimeError(f"Expected ViT-B feature dim 768, found {model.feature_dim}")
    guided_layers = [
        layer_id
        for layer_id, wrapper in enumerate(model.adapter_wrappers)
        if wrapper.activation_guidance is not None
    ]
    if activation_guidance is not None and guided_layers != [11]:
        raise RuntimeError(f"SGDS must target only final ViT block, found {guided_layers}")
    if activation_guidance is None and guided_layers:
        raise RuntimeError(f"SGDS-off path unexpectedly has guided layers: {guided_layers}")
    memory = PrototypeMemory(model.feature_dim)
    metadata = {
        "method": method,
        "adaptation_type": (
            "parallel attention K/V nonlinear bottleneck adapters"
            if str(config.get("adapter_placement", "mlp")) == "attention"
            else "nonlinear MLP-output bottleneck adapter"
        ),
        "adapter_trainable_parameters": sum(
            parameter.numel() for parameter in model.adapter_parameters()
        ),
        "task0_classifier_trainable_parameters": (
            task_sizes[0] * model.feature_dim
        ),
        "task0_total_trainable_parameters": (
            sum(parameter.numel() for parameter in model.adapter_parameters())
            + task_sizes[0] * model.feature_dim
        ),
        "total_model_parameters": count_parameters(model),
        "classifier_scale": float(config["classifier_scale"]),
        "source_paper": "Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure",
        "released_code_reference": ("${RESEARCH_WORKSPACE}/cds-main").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
        "scc_teacher": "transient frozen immediately-preceding backbone deepcopy",
        "classifier_adaptation_sampling": "direct Eq. (11) Gaussian; no age-dependent mean scaling",
        "result_label": f"PCDS {protocol['dataset']} adaptation",
        "task_sizes": task_sizes,
        "training_seed": seed,
        "class_order_seed": config.get("class_order_seed"),
        "use_sgds": bool(config.get("use_sgds", False)),
        "sgds_trainable_parameters": (
            count_parameters(model.activation_guidance, trainable_only=True)
            if model.activation_guidance is not None
            else 0
        ),
    }
    if model.activation_guidance is not None:
        metadata.update({
            "combined_method": "PCDS + SGDS activation-space guidance",
            "sgds_target_layer": model.activation_guidance.target_layer,
            "sgds_paper_k": model.activation_guidance.paper_k,
            "sgds_k_interpretation": "TopK floor(kN): 60% retain, 40% zero",
            "sgds_training_only_mask": True,
            "sgds_eq6_includes_self": True,
            "sgds_excluded": [
                "task-specific adapter bank",
                "inference retrieval/fusion/entropy selection",
                "SGDS prototype classifier and MOS alignment",
            ],
        })
    dump_json(output_dir / "run_metadata.json", metadata)
    logger.info("metadata=%s", metadata)

    start_task = 0
    step_rows: list[dict] = []
    task_rows: list[dict] = []
    class_rows: list[dict] = []
    pcds_rows: list[dict] = []
    ca_epoch_rows: list[dict] = []
    ca_analysis_rows: list[dict] = []
    task_accuracy_history: list[list[float]] = []
    class_accuracy_history: list[list[float]] = []
    reference_prototypes: dict[int, torch.Tensor] = {}
    sgds_semantic_rows: list[dict] = []
    sgds_semantic_task_rows: list[dict] = []
    sgds_activation_rows: list[dict] = []
    sgds_class_activation_rows: list[dict] = []
    sgds_rank_rows: list[dict] = []
    if resume and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path, model, memory, device)
        if state.get("method") != method:
            raise RuntimeError(
                f"Checkpoint method mismatch: {state.get('method')!r} != {method!r}"
            )
        if state.get("config") != config:
            raise RuntimeError("Checkpoint config differs from the requested config")
        start_task = int(state["next_task"])
        step_rows = state["step_rows"]
        task_rows = state["task_rows"]
        class_rows = state["class_rows"]
        pcds_rows = state["pcds_rows"]
        ca_epoch_rows = state["ca_epoch_rows"]
        ca_analysis_rows = state["ca_analysis_rows"]
        task_accuracy_history = state["task_accuracy_history"]
        class_accuracy_history = state["class_accuracy_history"]
        reference_prototypes = state["reference_prototypes"]
        sgds_semantic_rows = state.get("sgds_semantic_rows", [])
        sgds_semantic_task_rows = state.get("sgds_semantic_task_rows", [])
        sgds_activation_rows = state.get("sgds_activation_rows", [])
        sgds_class_activation_rows = state.get("sgds_class_activation_rows", [])
        sgds_rank_rows = state.get("sgds_rank_rows", [])
        logger.info("resumed from task boundary next_task=%d", start_task)

    amp = bool(config["amp"])
    for task_id in range(start_task, int(config["num_tasks"])):
        torch.cuda.reset_peak_memory_stats(device)
        begin_class = sum(task_sizes[:task_id])
        current_task_size = task_sizes[task_id]
        if memory.num_classes != begin_class:
            raise RuntimeError(
                f"Prototype memory has {memory.num_classes} classes before task {task_id}; "
                f"expected {begin_class}"
            )
        logger.info(
            "starting task=%d classes=%d..%d size=%d epochs=%d",
            task_id,
            begin_class,
            begin_class + current_task_size - 1,
            current_task_size,
            epochs_for_task(config, task_id),
        )
        reference_backbone = _snapshot_backbone(model) if task_id > 0 else None
        model.eval()
        statistics_dataset = data.train_dataset(
            task_id, task_sizes, deterministic=True
        )
        statistics_loader = make_loader(
            statistics_dataset,
            batch_size=int(config["statistics_batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 1,
        )
        if model.activation_guidance is not None:
            semantic_means = collect_current_task_means(
                model.base_features,
                statistics_loader,
                device,
                amp,
                begin_class,
                current_task_size,
                model.feature_dim,
            )
            new_semantic_rows = model.activation_guidance.prepare_task(
                task_id,
                begin_class,
                semantic_means,
                num_epochs=epochs_for_task(config, task_id),
            )
            sgds_semantic_rows.extend(new_semantic_rows)
            semantic_task_row = model.activation_guidance.semantic_task_summary()
            sgds_semantic_task_rows.append(semantic_task_row)
            logger.info(
                "SGDS strategy task=%d reuse=%d allocation=%d reuse_ratio=%.3f "
                "Eq6_self=true",
                task_id,
                semantic_task_row["knowledge_reuse_classes"],
                semantic_task_row["new_subspace_allocation_classes"],
                semantic_task_row["knowledge_reuse_ratio"],
            )
        before_feature_function = (
            reference_backbone if reference_backbone is not None else model.features
        )
        new_means_before, new_covariances_before, before_counts = (
            collect_current_task_statistics(
                before_feature_function,
                statistics_loader,
                device,
                amp,
                begin_class,
                current_task_size,
                model.feature_dim,
                float(config["prototype_covariance_epsilon"]),
            )
        )
        model.classifier.add_classes(new_means_before)

        train_dataset = data.train_dataset(
            task_id, task_sizes, deterministic=False
        )
        train_loader = make_loader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 2,
            drop_last=True,
        )
        new_steps, loss_summary, train_seconds = train_task(
            model,
            reference_backbone,
            memory.means,
            train_loader,
            device,
            config,
            method,
            task_id,
            logger,
        )
        step_rows.extend(new_steps)
        write_csv(output_dir / "training_steps.csv", step_rows)

        model.eval()
        new_means_after, new_covariances_after, after_counts = (
            # Deterministic transforms plus eval mode make before/after movement
            # attributable to the task update rather than adapter dropout.
            collect_current_task_statistics(
                model.features,
                statistics_loader,
                device,
                amp,
                begin_class,
                current_task_size,
                model.feature_dim,
                float(config["prototype_covariance_epsilon"]),
            )
        )
        if not torch.equal(before_counts, after_counts):
            raise RuntimeError("Before/after prototype passes observed different sample counts")
        if model.activation_guidance is not None:
            new_rank_rows = covariance_effective_rank_rows(
                new_covariances_after,
                after_counts,
                task_id,
                begin_class,
                float(config["prototype_covariance_epsilon"]),
                device,
            )
            sgds_rank_rows.extend(new_rank_rows)
            activation_row, new_class_activation_rows = (
                model.activation_guidance.activation_task_summary()
            )
            sgds_activation_rows.append(activation_row)
            sgds_class_activation_rows.extend(new_class_activation_rows)
            logger.info(
                "SGDS activation task=%d target_layer=%d actual_nonzero=%.4f "
                "actual_zero=%.4f mean_active=%.1f r99_mean=%.1f",
                task_id,
                activation_row["target_layer"],
                activation_row["actual_nonzero_ratio"],
                activation_row["actual_zero_ratio"],
                activation_row["mean_active_dimensions"],
                sum(row["effective_rank_99"] for row in new_rank_rows)
                / len(new_rank_rows),
            )
        else:
            new_rank_rows = covariance_effective_rank_rows(
                new_covariances_after,
                after_counts,
                task_id,
                begin_class,
                float(config["prototype_covariance_epsilon"]),
                device,
            )
            sgds_rank_rows.extend(new_rank_rows)

        apr_statistics = {
            "mean_new_displacement_norm": float(
                torch.linalg.vector_norm(
                    new_means_after - new_means_before, dim=1
                ).mean().item()
            ),
            "mean_apr_correction_norm": 0.0,
            "max_apr_correction_norm": 0.0,
            "mean_covariance_trace_before": (
                float(memory.covariances.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item())
                if memory.num_classes
                else float("nan")
            ),
            "mean_covariance_trace_after": (
                float(memory.covariances.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item())
                if memory.num_classes
                else float("nan")
            ),
            "mean_covariance_frobenius_change": 0.0,
        }
        if task_id > 0 and flags["apr"]:
            apr = adaptive_prototype_rectification(
                memory.means,
                memory.covariances,
                new_means_before,
                new_means_after,
                gamma=float(config["apr_gamma"]),
                rho=float(config["apr_rho"]),
                epsilon=float(config["apr_epsilon"]),
            )
            memory.replace_old(apr.means, apr.covariances)
            apr_statistics = apr.statistics
        memory.append(new_means_after, new_covariances_after, after_counts)

        seen_classes = begin_class + current_task_size
        test_loader = make_loader(
            data.seen_test_dataset(seen_classes),
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 3,
        )
        before_ca_evaluation = evaluate_seen(
            model, test_loader, device, amp, seen_classes
        )
        before_ca = _accuracy_parts(before_ca_evaluation, task_sizes, task_id)
        ca_summary = {
            "ca_mean_loss": float("nan"),
            "ca_final_loss": float("nan"),
            "ca_final_synthetic_accuracy": float("nan"),
            "ca_max_added_jitter": 0.0,
            "ca_eigh_fallback_classes": 0.0,
        }
        if task_id > 0 and flags["ca"]:
            new_ca_rows, ca_summary = adapt_classifier_from_prototypes(
                model, memory, config, task_id, device, logger
            )
            for row in new_ca_rows:
                row.update({"method": method, "seed": seed})
            ca_epoch_rows.extend(new_ca_rows)
        after_ca_evaluation = evaluate_seen(
            model, test_loader, device, amp, seen_classes
        )
        after_ca = _accuracy_parts(after_ca_evaluation, task_sizes, task_id)
        ca_analysis_row = {
            "method": method,
            "seed": seed,
            "task_id": task_id,
            "ca_applied": int(task_id > 0 and flags["ca"]),
            "seen_accuracy_before_ca": before_ca["seen_accuracy"],
            "seen_accuracy_after_ca": after_ca["seen_accuracy"],
            "seen_accuracy_ca_delta": after_ca["seen_accuracy"] - before_ca["seen_accuracy"],
            "old_accuracy_before_ca": before_ca["old_accuracy"],
            "old_accuracy_after_ca": after_ca["old_accuracy"],
            "old_accuracy_ca_delta": after_ca["old_accuracy"] - before_ca["old_accuracy"],
            "new_accuracy_before_ca": before_ca["new_accuracy"],
            "new_accuracy_after_ca": after_ca["new_accuracy"],
            "new_accuracy_ca_delta": after_ca["new_accuracy"] - before_ca["new_accuracy"],
            **ca_summary,
        }
        ca_analysis_rows.append(ca_analysis_row)

        task_row, task_accuracies = summarize_task_metrics(
            task_id,
            task_sizes,
            after_ca_evaluation,
            task_accuracy_history,
        )
        task_accuracy_history.append(task_accuracies)
        new_class_rows = build_class_rows(
            task_id,
            task_sizes,
            after_ca_evaluation,
            class_accuracy_history,
            reference_prototypes,
        )
        for row in new_class_rows:
            row.update({"method": method, "seed": seed})
        class_accuracy_history.append(after_ca_evaluation.class_accuracy)
        class_rows.extend(new_class_rows)

        new_covariance_change = torch.linalg.matrix_norm(
            new_covariances_after - new_covariances_before,
            ord="fro",
            dim=(1, 2),
        )
        pcds_row = {
            "method": method,
            "seed": seed,
            "task_id": task_id,
            "apr_enabled": int(flags["apr"]),
            "apr_applied": int(task_id > 0 and flags["apr"]),
            "scc_enabled": int(task_id > 0 and flags["scc"]),
            "ca_enabled": int(flags["ca"]),
            **loss_summary,
            **apr_statistics,
            "mean_new_covariance_trace_before": float(
                new_covariances_before.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item()
            ),
            "mean_new_covariance_trace_after": float(
                new_covariances_after.diagonal(dim1=1, dim2=2).sum(dim=1).mean().item()
            ),
            "mean_new_covariance_frobenius_change": float(
                new_covariance_change.mean().item()
            ),
            "prototype_classes": memory.num_classes,
            "prototype_memory_bytes": (
                memory.means.numel() * memory.means.element_size()
                + memory.covariances.numel() * memory.covariances.element_size()
                + memory.counts.numel() * memory.counts.element_size()
            ),
            **ca_summary,
        }
        pcds_rows.append(pcds_row)
        task_row.update({
            "method": method,
            "seed": seed,
            "epochs": epochs_for_task(config, task_id),
            "average_incremental_accuracy": sum(
                row["seen_accuracy"] for row in task_rows + [task_row]
            ) / (task_id + 1),
            "seen_accuracy_before_ca": before_ca["seen_accuracy"],
            "old_accuracy_before_ca": before_ca["old_accuracy"],
            "new_accuracy_before_ca": before_ca["new_accuracy"],
            "train_seconds": train_seconds,
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        })
        task_rows.append(task_row)
        write_csv(output_dir / "task_metrics.csv", task_rows)
        write_csv(output_dir / "class_metrics.csv", class_rows)
        write_csv(output_dir / "pcds_statistics.csv", pcds_rows)
        write_csv(output_dir / "classifier_adaptation.csv", ca_analysis_rows)
        write_csv(output_dir / "classifier_adaptation_epochs.csv", ca_epoch_rows)
        write_csv(output_dir / "sgds_semantic_strategy.csv", sgds_semantic_rows)
        write_csv(
            output_dir / "sgds_semantic_strategy_tasks.csv", sgds_semantic_task_rows
        )
        write_csv(output_dir / "sgds_activation_statistics.csv", sgds_activation_rows)
        write_csv(
            output_dir / "sgds_class_activation_statistics.csv",
            sgds_class_activation_rows,
        )
        write_csv(output_dir / "representation_effective_rank.csv", sgds_rank_rows)
        write_csv(output_dir / "sgds_effective_rank.csv", sgds_rank_rows)
        matrix_rows = [
            {
                "method": method,
                "seed": seed,
                "eval_task": stage,
                **{
                    f"task_{column}": value
                    for column, value in enumerate(accuracies)
                },
            }
            for stage, accuracies in enumerate(task_accuracy_history)
        ]
        write_csv(output_dir / "task_accuracy_matrix.csv", matrix_rows)
        checkpoint = {
            "next_task": task_id + 1,
            "backbone": {
                key: value.detach().cpu()
                for key, value in model.backbone.state_dict().items()
            },
            "classifier_blocks": model.classifier.export_weights(),
            "prototype_memory": memory.export(),
            "reference_prototypes": reference_prototypes,
            "sgds_semantic_rows": sgds_semantic_rows,
            "sgds_semantic_task_rows": sgds_semantic_task_rows,
            "sgds_activation_rows": sgds_activation_rows,
            "sgds_class_activation_rows": sgds_class_activation_rows,
            "sgds_rank_rows": sgds_rank_rows,
            "step_rows": step_rows,
            "task_rows": task_rows,
            "class_rows": class_rows,
            "pcds_rows": pcds_rows,
            "ca_epoch_rows": ca_epoch_rows,
            "ca_analysis_rows": ca_analysis_rows,
            "task_accuracy_history": task_accuracy_history,
            "class_accuracy_history": class_accuracy_history,
            "config": config,
            "method": method,
        }
        torch.save(checkpoint, checkpoint_path)
        model.classifier.freeze_all()
        logger.info(
            "finished task=%d seen=%.4f old=%s new=%.4f forget=%.4f "
            "CA_delta=%+.4f peak=%.2fGB",
            task_id,
            task_row["seen_accuracy"],
            "nan" if math.isnan(task_row["old_accuracy"]) else f"{task_row['old_accuracy']:.4f}",
            task_row["new_accuracy"],
            task_row["average_forgetting"],
            ca_analysis_row["seen_accuracy_ca_delta"],
            task_row["peak_cuda_allocated_gb"],
        )
        del reference_backbone
        torch.cuda.empty_cache()

    dump_json(completed_path, {
        "method": method,
        "tasks_completed": int(config["num_tasks"]),
        "final_accuracy": task_rows[-1]["seen_accuracy"],
        "average_incremental_accuracy": task_rows[-1]["average_incremental_accuracy"],
        "completed_unix_time": time.time(),
        "pid": os.getpid(),
    })
    logger.info("run complete final_accuracy=%.4f", task_rows[-1]["seen_accuracy"])
    return output_dir
