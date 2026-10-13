from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import CIFAR100IncrementalData, make_loader
from .experiment import initialize_classifier_centroids, write_csv
from .keeplora_model import KeepLoRAContinualViT
from .keeplora_subspace import (
    KeepLoRASubspaceBank,
    collect_official_style_features,
)
from .metrics import build_class_rows, evaluate_seen, summarize_task_metrics
from .utils import (
    assert_physical_gpu_one,
    collect_environment,
    count_parameters,
    dump_json,
    dump_yaml,
    resolve_class_order,
    set_seed,
    setup_logger,
    tensor_l2_norm,
)


METHOD_FLAGS = {
    "vanilla_lora": (False, False),
    "keep_lora_wp_only": (True, False),
    "keep_lora_m_only": (False, True),
    "keep_lora_full": (True, True),
}
METHODS = set(METHOD_FLAGS)


def probe_full_gradients(
    model: KeepLoRAContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> tuple[dict[str, torch.Tensor], int]:
    if max_batches < 1:
        raise ValueError("gradient_probe_batches must be positive")
    classifier_states = model.classifier.freeze_all()
    model.zero_grad(set_to_none=True)
    for module in model.lora_modules:
        module.enable_full_weight_gradient()
    model.train()
    batches = 0
    try:
        for images, labels in loader:
            if batches >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                loss = F.cross_entropy(model(images)["logits"], labels)
            loss.backward()
            batches += 1
        if batches == 0:
            raise RuntimeError("Gradient probe loader was empty")
        gradients = {
            module.site_key(target): module.full_weight_gradient(target) / batches
            for module, target in model.projection_sites()
        }
    finally:
        for module in model.lora_modules:
            module.disable_full_weight_gradient()
        model.classifier.restore_trainable_states(classifier_states)
        model.zero_grad(set_to_none=True)
    return gradients, batches


def initialize_from_gradients(
    model: KeepLoRAContinualViT,
    gradients: dict[str, torch.Tensor],
    bank: KeepLoRASubspaceBank,
    use_wp: bool,
    use_m: bool,
    tolerance: float,
    task_id: int,
) -> list[dict]:
    rows: list[dict] = []
    for module, target in model.projection_sites():
        site_key = module.site_key(target)
        basis = bank.protected_basis(site_key, use_wp, use_m)
        stats = module.initialize_branch(
            target=target,
            gradient_weight=gradients[site_key],
            protected_basis=basis,
            tolerance=tolerance,
        )
        rows.append({"task_id": task_id, **stats.__dict__})
    return rows


def effective_update_geometry(
    model: KeepLoRAContinualViT,
    before: dict[str, torch.Tensor],
    bank: KeepLoRASubspaceBank,
    use_wp: bool,
    use_m: bool,
) -> tuple[float, float, float]:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float64)
    parallel = torch.zeros_like(total)
    residual = torch.zeros_like(total)
    for module, target in model.projection_sites():
        site_key = module.site_key(target)
        B = getattr(module, f"B_{target}")
        delta_B = B.detach().float() - before[site_key]
        delta_weight = module.effective_delta_from_B_delta(target, delta_B).float()
        total += delta_weight.double().square().sum()
        Q = bank.protected_basis(site_key, use_wp, use_m, delta_weight.device)
        if Q.numel():
            component = (delta_weight @ Q) @ Q.transpose(0, 1)
            remainder = delta_weight - component
            parallel += component.double().square().sum()
            residual += remainder.double().square().sum()
        else:
            residual += delta_weight.double().square().sum()
    return tuple(float(value.sqrt().item()) for value in (total, parallel, residual))


def train_task(
    model: KeepLoRAContinualViT,
    loader,
    device: torch.device,
    config: dict[str, Any],
    method: str,
    task_id: int,
    bank: KeepLoRASubspaceBank,
    use_wp: bool,
    use_m: bool,
    logger,
) -> tuple[list[dict], float]:
    parameters = model.lora_B_parameters() + model.classifier.current_parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    batches_per_epoch = len(loader)
    if config.get("max_train_batches") is not None:
        batches_per_epoch = min(batches_per_epoch, int(config["max_train_batches"]))
    total_steps = int(config["epochs_per_task"]) * batches_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=0.0
    )
    amp = bool(config["amp"])
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp,
        init_scale=float(config.get("grad_scaler_init_scale", 256.0)),
        growth_interval=int(config.get("grad_scaler_growth_interval", 100000)),
    )
    rows: list[dict] = []
    task_step = 0
    started = time.time()
    model.train()
    for epoch in range(int(config["epochs_per_task"])):
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= batches_per_epoch:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            before = {
                module.site_key(target): getattr(module, f"B_{target}")
                .detach()
                .float()
                .clone()
                for module, target in model.projection_sites()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = model(images)
                loss = F.cross_entropy(output["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            lora_gradient_norm = tensor_l2_norm([
                parameter.grad
                for parameter in model.lora_B_parameters()
                if parameter.grad is not None
            ])
            classifier_gradient_norm = tensor_l2_norm([
                parameter.grad
                for parameter in model.classifier.current_parameters()
                if parameter.grad is not None
            ])
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            skipped = int(amp and scale_after < scale_before)
            if skipped and bool(config.get("fail_on_optimizer_skip", True)):
                raise FloatingPointError(
                    f"GradScaler skipped task={task_id} step={task_step}; "
                    f"scale {scale_before} -> {scale_after}"
                )
            update_norm, parallel_norm, residual_norm = effective_update_geometry(
                model, before, bank, use_wp, use_m
            )
            scheduler.step()
            correct = int(output["logits"].argmax(dim=1).eq(labels).sum().item())
            row = {
                "method": method,
                "seed": int(config["seed"]),
                "task_id": task_id,
                "epoch": epoch,
                "batch_index": batch_index,
                "task_step": task_step,
                "loss": float(loss.detach().item()),
                "batch_accuracy": correct / int(labels.numel()),
                "learning_rate": learning_rate,
                "lora_gradient_norm": lora_gradient_norm,
                "classifier_gradient_norm": classifier_gradient_norm,
                "grad_scaler_scale_before": scale_before,
                "grad_scaler_scale_after": scale_after,
                "optimizer_step_skipped": skipped,
                "parameter_update_norm": update_norm,
                "protected_parallel_update_norm": parallel_norm,
                "protected_residual_update_norm": residual_norm,
                "protected_parallel_ratio": parallel_norm / (update_norm + 1.0e-12),
            }
            rows.append(row)
            if task_step % int(config["log_every_steps"]) == 0:
                logger.info(
                    "task=%d step=%d/%d loss=%.4f acc=%.3f update=%.3e overlap=%.3e",
                    task_id,
                    task_step + 1,
                    total_steps,
                    row["loss"],
                    row["batch_accuracy"],
                    update_norm,
                    row["protected_parallel_ratio"],
                )
            task_step += 1
    return rows, time.time() - started


def _load_checkpoint(
    path: Path,
    model: KeepLoRAContinualViT,
    bank: KeepLoRASubspaceBank,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    model.classifier.load_weights(state["classifier_blocks"], device)
    bank.load(state["subspace_bank"])
    return state


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def run_experiment(
    config: dict[str, Any],
    method: str,
    repo_root: Path,
    resume: bool = False,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose one of {sorted(METHODS)}")
    use_wp, use_m = METHOD_FLAGS[method]
    gpu_info = assert_physical_gpu_one()
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
    logger = setup_logger(output_dir)
    set_seed(seed)
    device = torch.device("cuda:0")
    class_order = resolve_class_order(config["class_order"])
    dump_yaml(output_dir / "resolved_config.yaml", config)
    dump_json(output_dir / "class_order.json", class_order)
    dump_json(output_dir / "environment.json", collect_environment(repo_root, gpu_info))
    (output_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    logger.info("method=%s Wp=%s M=%s output=%s", method, use_wp, use_m, output_dir)
    logger.info("GPU: %s", gpu_info)

    data = CIFAR100IncrementalData(
        root=config["data_root"],
        class_order=class_order,
        image_size=int(config["image_size"]),
    )
    model = KeepLoRAContinualViT(
        model_name=config["model_name"],
        pretrained=bool(config["pretrained"]),
        classifier_scale=float(config["classifier_scale"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_targets=config["lora_targets"],
    ).to(device)
    bank = KeepLoRASubspaceBank(model.site_keys(), model.feature_dim)
    wp_rows = bank.initialize_wp(
        model,
        epsilon_w=float(config["epsilon_w"]),
        tolerance=float(config["subspace_tolerance"]),
    )
    for row in wp_rows:
        row.update({"method": method, "seed": seed, "wp_enabled": int(use_wp)})
    write_csv(output_dir / "wp_spectrum.csv", wp_rows)
    logger.info(
        "parameters total=%d initially_trainable=%d sites=%d Wp_rank_mean=%.1f",
        count_parameters(model),
        count_parameters(model, trainable_only=True),
        len(model.site_keys()),
        _mean([float(row["selected_rank"]) for row in wp_rows]),
    )

    start_task = 0
    step_rows: list[dict] = []
    init_rows: list[dict] = []
    task_rows: list[dict] = []
    class_rows: list[dict] = []
    protected_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    feature_collection_rows: list[dict] = []
    task_accuracy_history: list[list[float]] = []
    class_accuracy_history: list[list[float]] = []
    reference_prototypes: dict[int, torch.Tensor] = {}
    if resume and checkpoint_path.exists():
        state = _load_checkpoint(checkpoint_path, model, bank, device)
        start_task = int(state["next_task"])
        step_rows = state["step_rows"]
        init_rows = state["init_rows"]
        task_rows = state["task_rows"]
        class_rows = state["class_rows"]
        protected_rows = state["protected_rows"]
        spectrum_rows = state["spectrum_rows"]
        feature_collection_rows = state["feature_collection_rows"]
        task_accuracy_history = state["task_accuracy_history"]
        class_accuracy_history = state["class_accuracy_history"]
        reference_prototypes = state["reference_prototypes"]
        logger.info("resumed from task boundary next_task=%d", start_task)

    classes_per_task = int(config["classes_per_task"])
    amp = bool(config["amp"])
    for task_id in range(start_task, int(config["num_tasks"])):
        torch.cuda.reset_peak_memory_stats(device)
        logger.info(
            "starting task=%d classes=%d..%d",
            task_id,
            task_id * classes_per_task,
            (task_id + 1) * classes_per_task - 1,
        )
        centroid_limit = config.get("centroid_images_per_class")
        deterministic_train = data.train_dataset(
            task_id,
            classes_per_task,
            deterministic=True,
            per_class_limit=(None if centroid_limit is None else int(centroid_limit)),
            selection_seed=seed + task_id,
        )
        centroid_loader = make_loader(
            deterministic_train,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 1,
        )
        initial_weights = initialize_classifier_centroids(
            model,
            centroid_loader,
            device,
            amp,
            begin_class=task_id * classes_per_task,
            num_classes=classes_per_task,
        )
        model.classifier.add_classes(initial_weights)

        train_dataset = data.train_dataset(task_id, classes_per_task, deterministic=False)
        probe_loader = make_loader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 2,
            drop_last=True,
        )
        gradients, probe_batches = probe_full_gradients(
            model,
            probe_loader,
            device,
            bool(config.get("gradient_probe_amp", False)),
            int(config["gradient_probe_batches"]),
        )
        new_init_rows = initialize_from_gradients(
            model,
            gradients,
            bank,
            use_wp,
            use_m,
            tolerance=float(config["subspace_tolerance"]),
            task_id=task_id,
        )
        for row in new_init_rows:
            row.update({
                "method": method,
                "seed": seed,
                "probe_batches": probe_batches,
                "wp_enabled": int(use_wp),
                "m_enabled": int(use_m),
            })
        init_rows.extend(new_init_rows)
        write_csv(output_dir / "lora_initialization.csv", init_rows)

        train_loader = make_loader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 3,
            drop_last=True,
        )
        new_step_rows, train_seconds = train_task(
            model,
            train_loader,
            device,
            config,
            method,
            task_id,
            bank,
            use_wp,
            use_m,
            logger,
        )
        step_rows.extend(new_step_rows)
        write_csv(output_dir / "step_updates.csv", step_rows)
        model.merge_all_lora()

        seen_classes = (task_id + 1) * classes_per_task
        test_loader = make_loader(
            data.seen_test_dataset(seen_classes),
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 4,
        )
        evaluation = evaluate_seen(model, test_loader, device, amp, seen_classes)
        task_row, task_accuracies = summarize_task_metrics(
            task_id, classes_per_task, evaluation, task_accuracy_history
        )
        task_accuracy_history.append(task_accuracies)
        new_class_rows = build_class_rows(
            task_id,
            classes_per_task,
            evaluation,
            class_accuracy_history,
            reference_prototypes,
        )
        for row in new_class_rows:
            row.update({
                "method": method,
                "seed": seed,
                "class_accuracy": row["accuracy"],
                "class_forgetting": row["forgetting"],
            })
        class_accuracy_history.append(evaluation.class_accuracy)
        class_rows.extend(new_class_rows)

        feature_matrices: dict[str, torch.Tensor] = {}
        if use_m:
            per_class_limit = config.get("feature_images_per_class")
            feature_dataset = data.train_dataset(
                task_id,
                classes_per_task,
                deterministic=True,
                per_class_limit=(
                    None if per_class_limit is None else int(per_class_limit)
                ),
                selection_seed=seed + task_id,
            )
            feature_loader = make_loader(
                feature_dataset,
                batch_size=int(config["feature_batch_size"]),
                shuffle=False,
                num_workers=int(config["num_workers"]),
                seed=seed + task_id * 1000 + 5,
                drop_last=True,
            )
            max_feature_batches = config.get("max_feature_batches")
            feature_matrices, new_collection_rows = collect_official_style_features(
                model,
                feature_loader,
                device,
                amp,
                None if max_feature_batches is None else int(max_feature_batches),
            )
            for row in new_collection_rows:
                row.update({"method": method, "seed": seed, "task_id": task_id})
            feature_collection_rows.extend(new_collection_rows)

        bank_stats, new_spectrum_rows = bank.update_task_features(
            model,
            feature_matrices,
            use_wp=use_wp,
            use_m=use_m,
            epsilon_f=float(config["epsilon_f"]),
            tolerance=float(config["subspace_tolerance"]),
        )
        del feature_matrices
        for row in new_spectrum_rows:
            row.update({"method": method, "seed": seed, "task_id": task_id})
        spectrum_rows.extend(new_spectrum_rows)
        protected_rows.extend(
            {"method": method, "seed": seed, "task_id": task_id, **row.__dict__}
            for row in bank_stats
        )

        old_class_rows = [row for row in new_class_rows if not row["is_new"]]
        task_row.update({
            "method": method,
            "seed": seed,
            "epochs": int(config["epochs_per_task"]),
            "final_accuracy": (
                evaluation.seen_accuracy
                if task_id + 1 == int(config["num_tasks"])
                else float("nan")
            ),
            "average_incremental_accuracy": _mean(
                [row["seen_accuracy"] for row in task_rows + [task_row]]
            ),
            "mean_lora_gradient_norm": _mean(
                [row["lora_gradient_norm"] for row in new_step_rows]
            ),
            "mean_parameter_update_norm": _mean(
                [row["parameter_update_norm"] for row in new_step_rows]
            ),
            "mean_protected_parallel_ratio": _mean(
                [row["protected_parallel_ratio"] for row in new_step_rows]
            ),
            "optimizer_skipped_steps": sum(
                row["optimizer_step_skipped"] for row in new_step_rows
            ),
            "wp_rank_mean": _mean([float(row.wp_rank) for row in bank_stats]),
            "m_rank_mean": _mean([float(row.m_rank) for row in bank_stats]),
            "unified_rank_mean": _mean([float(row.unified_rank) for row in bank_stats]),
            "residual_rank_mean": _mean([float(row.residual_rank) for row in bank_stats]),
            "residual_rank_min": min(row.residual_rank for row in bank_stats),
            "protected_ratio_mean": _mean(
                [row.protected_ratio for row in bank_stats]
            ),
            "old_prototype_cosine_drift": _mean(
                [row["prototype_cosine_drift"] for row in old_class_rows]
            ),
            "old_prototype_l2_drift": _mean(
                [row["prototype_l2_drift"] for row in old_class_rows]
            ),
            "train_seconds": train_seconds,
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
        })
        task_rows.append(task_row)
        write_csv(output_dir / "task_metrics.csv", task_rows)
        write_csv(output_dir / "class_metrics.csv", class_rows)
        write_csv(output_dir / "protected_space.csv", protected_rows)
        write_csv(output_dir / "feature_spectrum.csv", spectrum_rows)
        write_csv(output_dir / "feature_collection.csv", feature_collection_rows)
        matrix_rows = [
            {
                "method": method,
                "seed": seed,
                "eval_task": index,
                **{f"task_{column}": value for column, value in enumerate(row)},
            }
            for index, row in enumerate(task_accuracy_history)
        ]
        write_csv(output_dir / "accuracy_matrix.csv", matrix_rows)
        checkpoint = {
            "next_task": task_id + 1,
            "backbone": {
                key: value.detach().cpu()
                for key, value in model.backbone.state_dict().items()
            },
            "classifier_blocks": model.classifier.export_weights(),
            "subspace_bank": bank.export(),
            "reference_prototypes": reference_prototypes,
            "step_rows": step_rows,
            "init_rows": init_rows,
            "task_rows": task_rows,
            "class_rows": class_rows,
            "protected_rows": protected_rows,
            "spectrum_rows": spectrum_rows,
            "feature_collection_rows": feature_collection_rows,
            "task_accuracy_history": task_accuracy_history,
            "class_accuracy_history": class_accuracy_history,
            "config": config,
            "method": method,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            "finished task=%d seen=%.4f old=%s new=%.4f forget=%.4f "
            "protected=%.1f/768 residual_min=%d peak=%.2fGB",
            task_id,
            task_row["seen_accuracy"],
            "nan"
            if math.isnan(task_row["old_accuracy"])
            else f"{task_row['old_accuracy']:.4f}",
            task_row["new_accuracy"],
            task_row["average_forgetting"],
            task_row["unified_rank_mean"],
            task_row["residual_rank_min"],
            task_row["peak_cuda_allocated_gb"],
        )

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
