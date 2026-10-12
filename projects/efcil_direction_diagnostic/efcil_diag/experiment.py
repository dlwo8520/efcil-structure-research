from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .data import CIFAR100IncrementalData, make_loader
from .metrics import build_class_rows, evaluate_seen, summarize_task_metrics
from .model import ContinualViT
from .subspace import ProtectedSubspaceBank, collect_class_directions
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


METHODS = {
    "vanilla_lora",
    "small_update_lora",
    "orthogonal_lora",
    "orthogonal_lora_norm_matched",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_norm_schedule(path: Path) -> dict[tuple[int, int], float]:
    if not path.exists():
        return {}
    schedule: dict[tuple[int, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            schedule[(int(row["task_id"]), int(row["task_step"]))] = float(
                row["parameter_update_norm"]
            )
    return schedule


@torch.no_grad()
def initialize_classifier_centroids(
    model: ContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    begin_class: int,
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    sums = torch.zeros(num_classes, model.feature_dim, device=device, dtype=torch.float64)
    counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        local_labels = labels.to(device, non_blocking=True) - begin_class
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            features = model.features(images)
        sums.index_add_(0, local_labels, features.double())
        counts.index_add_(0, local_labels, torch.ones_like(local_labels))
    if bool((counts == 0).any()):
        raise RuntimeError("Classifier initialization missed a current class")
    return F.normalize((sums / counts[:, None]).float(), dim=1)


def probe_full_gradients(
    model: ContinualViT,
    loader,
    device: torch.device,
    amp: bool,
    max_batches: int,
) -> tuple[dict[tuple[int, str], torch.Tensor], int]:
    classifier_states = model.classifier.freeze_all()
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
            (module.layer_id, target): module.full_weight_gradient(target) / batches
            for module in model.lora_modules
            for target in module.targets
        }
    finally:
        for module in model.lora_modules:
            module.disable_full_weight_gradient()
        model.classifier.restore_trainable_states(classifier_states)
    return gradients, batches


def initialize_lora_from_gradients(
    model: ContinualViT,
    gradients: dict[tuple[int, str], torch.Tensor],
    bank: ProtectedSubspaceBank,
    orthogonal: bool,
    tolerance: float,
    task_id: int,
) -> list[dict]:
    rows: list[dict] = []
    for module in model.lora_modules:
        basis = bank.basis(module.layer_id)
        for target in module.targets:
            stats = module.initialize_branch(
                target=target,
                gradient=gradients[(module.layer_id, target)],
                protected_basis=basis,
                orthogonal=orthogonal,
                tolerance=tolerance,
            )
            row = {"task_id": task_id, **stats.__dict__}
            rows.append(row)
    return rows


def effective_update_geometry(
    model: ContinualViT,
    before: dict[tuple[int, str], torch.Tensor],
    bank: ProtectedSubspaceBank,
) -> tuple[float, float, float]:
    total = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    parallel = torch.zeros_like(total)
    residual = torch.zeros_like(total)
    for module in model.lora_modules:
        basis = bank.basis(module.layer_id)
        Q = basis.to(device=module.base.weight.device, dtype=torch.float32)
        for target, _, parameter in module.branch_items():
            delta_B = parameter.detach().float() - before[(module.layer_id, target)]
            delta_W = module.effective_delta_from_B_delta(target, delta_B).float()
            total += delta_W.double().pow(2).sum()
            if Q.numel():
                component = (delta_W @ Q) @ Q.transpose(0, 1)
                remainder = delta_W - component
                parallel += component.double().pow(2).sum()
                residual += remainder.double().pow(2).sum()
            else:
                residual += delta_W.double().pow(2).sum()
    return tuple(float(value.sqrt().item()) for value in (total, parallel, residual))


def scale_B_step(
    model: ContinualViT,
    before: dict[tuple[int, str], torch.Tensor],
    factor: float,
) -> None:
    with torch.no_grad():
        for module in model.lora_modules:
            for target, _, parameter in module.branch_items():
                old = before[(module.layer_id, target)]
                parameter.copy_(old + factor * (parameter - old))


def train_task(
    model: ContinualViT,
    loader,
    device: torch.device,
    config: dict[str, Any],
    method: str,
    task_id: int,
    bank: ProtectedSubspaceBank,
    norm_schedule: dict[tuple[int, int], float],
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
                (module.layer_id, target): parameter.detach().float().clone()
                for module in model.lora_modules
                for target, _, parameter in module.branch_items()
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                output = model(images)
                loss = F.cross_entropy(output["logits"], labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            lora_gradient_norm = tensor_l2_norm([
                parameter.grad for parameter in model.lora_B_parameters()
                if parameter.grad is not None
            ])
            classifier_gradient_norm = tensor_l2_norm([
                parameter.grad for parameter in model.classifier.current_parameters()
                if parameter.grad is not None
            ])
            learning_rate = float(optimizer.param_groups[0]["lr"])
            scaler_scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scaler_scale_after = float(scaler.get_scale())
            optimizer_step_skipped = int(amp and scaler_scale_after < scaler_scale_before)
            if optimizer_step_skipped and bool(config.get("fail_on_optimizer_skip", True)):
                raise FloatingPointError(
                    f"GradScaler skipped task={task_id} step={task_step}; "
                    f"scale {scaler_scale_before} -> {scaler_scale_after}. "
                    "The run is stopped to avoid a condition-dependent norm confound."
                )
            raw_norm, raw_parallel, raw_residual = effective_update_geometry(
                model, before, bank
            )
            target_norm = float("nan")
            factor = 1.0
            saturated = False
            if method in {"small_update_lora", "orthogonal_lora_norm_matched"}:
                target_norm = norm_schedule.get((task_id, task_step), float("nan"))
                if method == "small_update_lora" and not math.isnan(target_norm):
                    target_norm *= float(config["small_update_ratio"])
                if math.isnan(target_norm):
                    if method == "small_update_lora":
                        factor = float(config["small_update_fallback_scale"])
                    else:
                        factor = 1.0
                    saturated = True
                elif raw_norm > 0.0:
                    requested = target_norm / raw_norm
                    if method == "small_update_lora":
                        factor = min(1.0, requested)
                    else:
                        factor = min(float(config["norm_match_max_scale"]), requested)
                    saturated = not math.isclose(factor, requested, rel_tol=1.0e-6)
                elif target_norm > 0.0:
                    factor = 1.0
                    saturated = True
            if factor != 1.0:
                scale_B_step(model, before, factor)
            applied_norm, parallel_norm, residual_norm = effective_update_geometry(
                model, before, bank
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
                "grad_scaler_scale_before": scaler_scale_before,
                "grad_scaler_scale_after": scaler_scale_after,
                "optimizer_step_skipped": optimizer_step_skipped,
                "raw_parameter_update_norm": raw_norm,
                "target_update_norm": target_norm,
                "applied_scale": factor,
                "norm_match_saturated": int(saturated),
                "parameter_update_norm": applied_norm,
                "protected_parallel_update_norm": parallel_norm,
                "protected_residual_update_norm": residual_norm,
                "protected_parallel_ratio": parallel_norm / (applied_norm + 1.0e-12),
                "raw_protected_parallel_ratio": raw_parallel / (raw_norm + 1.0e-12),
                "raw_protected_residual_norm": raw_residual,
            }
            rows.append(row)
            if task_step % int(config["log_every_steps"]) == 0:
                logger.info(
                    "task=%d step=%d/%d loss=%.4f acc=%.3f update=%.3e scale=%.3f",
                    task_id,
                    task_step + 1,
                    total_steps,
                    row["loss"],
                    row["batch_accuracy"],
                    applied_norm,
                    factor,
                )
            task_step += 1
    return rows, time.time() - started


def _reference_schedule_path(output_dir: Path, method: str) -> Path | None:
    if method == "small_update_lora":
        return output_dir.parent / "vanilla_lora" / "step_updates.csv"
    if method == "orthogonal_lora_norm_matched":
        return output_dir.parent / "small_update_lora" / "step_updates.csv"
    return None


def _load_checkpoint(
    path: Path,
    model: ContinualViT,
    bank: ProtectedSubspaceBank,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(state["backbone"])
    model.classifier.load_weights(state["classifier_blocks"], device)
    bank.load(state["protected_bases"])
    return state


def run_experiment(
    config: dict[str, Any],
    method: str,
    repo_root: Path,
    resume: bool = False,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose one of {sorted(METHODS)}")
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
    logger.info("method=%s output=%s", method, output_dir)
    logger.info("GPU: %s", gpu_info)

    data = CIFAR100IncrementalData(
        root=config["data_root"],
        class_order=class_order,
        image_size=int(config["image_size"]),
    )
    model = ContinualViT(
        model_name=config["model_name"],
        pretrained=bool(config["pretrained"]),
        classifier_scale=float(config["classifier_scale"]),
        lora_rank=int(config["lora_rank"]),
        lora_alpha=float(config["lora_alpha"]),
        lora_targets=config["lora_targets"],
    ).to(device)
    bank = ProtectedSubspaceBank(len(model.lora_modules), model.lora_modules[0].dim)
    logger.info(
        "parameters total=%d initially_trainable=%d feature_dim=%d",
        count_parameters(model),
        count_parameters(model, trainable_only=True),
        model.feature_dim,
    )

    start_task = 0
    step_rows: list[dict] = []
    init_rows: list[dict] = []
    task_rows: list[dict] = []
    class_rows: list[dict] = []
    subspace_rows: list[dict] = []
    subspace_class_rows: list[dict] = []
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
        subspace_rows = state["subspace_rows"]
        subspace_class_rows = state["subspace_class_rows"]
        task_accuracy_history = state["task_accuracy_history"]
        class_accuracy_history = state["class_accuracy_history"]
        reference_prototypes = state["reference_prototypes"]
        logger.info("resumed from task boundary next_task=%d", start_task)

    reference_path = _reference_schedule_path(output_dir, method)
    norm_schedule = read_norm_schedule(reference_path) if reference_path else {}
    if reference_path:
        if not norm_schedule:
            raise RuntimeError(f"Required norm reference is missing or empty: {reference_path}")
        logger.info("loaded %d norm targets from %s", len(norm_schedule), reference_path)

    classes_per_task = int(config["classes_per_task"])
    amp = bool(config["amp"])
    for task_id in range(start_task, int(config["num_tasks"])):
        torch.cuda.reset_peak_memory_stats(device)
        logger.info("starting task=%d classes=%d..%d", task_id, task_id * classes_per_task, (task_id + 1) * classes_per_task - 1)
        deterministic_train = data.train_dataset(
            task_id, classes_per_task, deterministic=True
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
            amp,
            int(config["gradient_probe_batches"]),
        )
        new_init_rows = initialize_lora_from_gradients(
            model,
            gradients,
            bank,
            orthogonal=method.startswith("orthogonal_lora"),
            tolerance=float(config["subspace_tolerance"]),
            task_id=task_id,
        )
        for row in new_init_rows:
            row["method"] = method
            row["seed"] = seed
            row["probe_batches"] = probe_batches
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
            norm_schedule,
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
            task_id,
            classes_per_task,
            evaluation,
            task_accuracy_history,
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
            row["method"] = method
            row["seed"] = seed
            row["class_accuracy"] = row["accuracy"]
            row["class_forgetting"] = row["forgetting"]
        class_accuracy_history.append(evaluation.class_accuracy)
        class_rows.extend(new_class_rows)

        subspace_dataset = data.train_dataset(
            task_id,
            classes_per_task,
            deterministic=True,
            per_class_limit=int(config["subspace_images_per_class"]),
            selection_seed=seed + task_id,
        )
        subspace_loader = make_loader(
            subspace_dataset,
            batch_size=int(config["eval_batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            seed=seed + task_id * 1000 + 5,
        )
        candidates, new_subspace_class_rows = collect_class_directions(
            model,
            subspace_loader,
            device,
            amp,
            tokens_per_image=int(config["subspace_tokens_per_image"]),
            topk=int(config["subspace_topk"]),
            energy_threshold=config.get("subspace_energy_threshold"),
            tolerance=float(config["subspace_tolerance"]),
        )
        bank_stats = bank.update(
            candidates,
            rank_cap=int(config["protected_rank_cap"]),
            tolerance=float(config["subspace_tolerance"]),
        )
        for row in new_subspace_class_rows:
            row["method"] = method
            row["seed"] = seed
            row["task_id"] = task_id
        subspace_class_rows.extend(new_subspace_class_rows)
        subspace_rows.extend(
            {"method": method, "seed": seed, "task_id": task_id, **row.__dict__}
            for row in bank_stats
        )

        old_class_rows = [row for row in new_class_rows if not row["is_new"]]
        mean_value = lambda values: sum(values) / len(values) if values else float("nan")
        task_row.update({
            "method": method,
            "seed": seed,
            "epochs": int(config["epochs_per_task"]),
            "final_accuracy": evaluation.seen_accuracy if task_id + 1 == int(config["num_tasks"]) else float("nan"),
            "average_incremental_accuracy": sum(row["seen_accuracy"] for row in task_rows + [task_row]) / (task_id + 1),
            "mean_raw_gradient_norm": mean_value([row["lora_gradient_norm"] for row in new_step_rows]),
            "mean_parameter_update_norm": mean_value([row["parameter_update_norm"] for row in new_step_rows]),
            "mean_projection_component_norm": mean_value([row["protected_parallel_update_norm"] for row in new_step_rows]),
            "mean_residual_component_norm": mean_value([row["protected_residual_update_norm"] for row in new_step_rows]),
            "mean_projection_ratio": mean_value([row["protected_parallel_ratio"] for row in new_step_rows]),
            "optimizer_skipped_steps": sum(row["optimizer_step_skipped"] for row in new_step_rows),
            "protected_rank": mean_value([row.protected_rank for row in bank_stats]),
            "protected_ratio": mean_value([row.protected_ratio for row in bank_stats]),
            "old_prototype_cosine_drift": mean_value([row["prototype_cosine_drift"] for row in old_class_rows]),
            "old_prototype_l2_drift": mean_value([row["prototype_l2_drift"] for row in old_class_rows]),
            "train_seconds": train_seconds,
            "peak_cuda_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_cuda_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
            "cuda_allocated_after_task_gb": torch.cuda.memory_allocated(device) / (1024**3),
            "cuda_reserved_after_task_gb": torch.cuda.memory_reserved(device) / (1024**3),
        })
        task_rows.append(task_row)
        write_csv(output_dir / "task_metrics.csv", task_rows)
        write_csv(output_dir / "class_metrics.csv", class_rows)
        write_csv(output_dir / "subspace_rank.csv", subspace_rows)
        write_csv(output_dir / "subspace_class_stats.csv", subspace_class_rows)
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
            "backbone": {key: value.detach().cpu() for key, value in model.backbone.state_dict().items()},
            "classifier_blocks": model.classifier.export_weights(),
            "protected_bases": bank.export(),
            "reference_prototypes": reference_prototypes,
            "step_rows": step_rows,
            "init_rows": init_rows,
            "task_rows": task_rows,
            "class_rows": class_rows,
            "subspace_rows": subspace_rows,
            "subspace_class_rows": subspace_class_rows,
            "task_accuracy_history": task_accuracy_history,
            "class_accuracy_history": class_accuracy_history,
            "config": config,
            "method": method,
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(
            "finished task=%d seen_acc=%.4f old_acc=%s new_acc=%.4f forgetting=%.4f peak=%.2fGB",
            task_id,
            task_row["seen_accuracy"],
            "nan" if math.isnan(task_row["old_accuracy"]) else f'{task_row["old_accuracy"]:.4f}',
            task_row["new_accuracy"],
            task_row["average_forgetting"],
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
