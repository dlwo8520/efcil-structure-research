from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from oracle_audit.config import bootstrap_paths, protocol_config
from oracle_audit.metrics import ClassStats


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def discover_checkpoints(run_dir: Path, explicit: Iterable[str] = ()) -> list[Path]:
    explicit_paths = [Path(item).expanduser().resolve() for item in explicit]
    if explicit_paths:
        paths = explicit_paths
    else:
        snapshot_dir = run_dir / "task_checkpoints"
        paths = sorted(snapshot_dir.glob("task_*.pt")) if snapshot_dir.is_dir() else []
        if not paths and (run_dir / "checkpoint_latest.pt").is_file():
            paths = [run_dir / "checkpoint_latest.pt"]
    if not paths:
        raise FileNotFoundError(f"no task checkpoints found in {run_dir}")
    inspected = []
    for path in paths:
        checkpoint = torch_load(path)
        if checkpoint.get("stage") != "task_end":
            raise ValueError(f"audit requires task_end checkpoint, got {checkpoint.get('stage')}: {path}")
        inspected.append((int(checkpoint["cur_task"]) + 1, path))
        del checkpoint
    tasks = [task for task, _ in inspected]
    if len(tasks) != len(set(tasks)):
        raise ValueError(f"duplicate checkpoint tasks: {tasks}")
    return [path for _, path in sorted(inspected)]


def build_data_manager(protocol: str, data_root: str | Path, seed: int):
    bootstrap_paths()
    from selective.protocol import SweepDataManager

    config = protocol_config(protocol, data_root)
    config.update({"seed": int(seed), "run_dir": None, "verify_split_content": False})
    return SweepDataManager(
        config["dataset"], config["shuffle"], seed, config["init_cls"], config["increment"], args=config
    ), config


def load_frozen_learner(
    checkpoint: dict,
    device: torch.device,
    scratch_dir: Path,
):
    bootstrap_paths()
    from models.CDS_adapter import Learner as StructureLearner

    config = dict(checkpoint["config"])
    config.update(
        {
            "device": [device],
            "run_dir": str(scratch_dir),
            "num_workers": 0,
        }
    )
    learner = StructureLearner(config)
    network = learner._network
    for size in checkpoint["task_sizes"]:
        network.update_fc(int(size))
    network.load_state_dict(checkpoint["network_state_dict"], strict=True)
    network.convnet.flag = int(checkpoint.get("network_flag_after_task", checkpoint["cur_task"] + 1))
    network.to(device)
    network.eval()
    network.requires_grad_(False)
    if network.training or any(parameter.requires_grad for parameter in network.parameters()):
        raise AssertionError("oracle audit model must be frozen and in eval mode")
    learner._cur_task = int(checkpoint["cur_task"])
    learner._known_classes = int(checkpoint["known_classes"])
    learner._total_classes = int(checkpoint["total_classes"])
    learner.task_sizes = list(map(int, checkpoint["task_sizes"]))
    return learner


def extract_split(
    learner,
    data_manager,
    class_ids: np.ndarray,
    source: str,
    batch_size: int,
    num_workers: int,
    max_samples_per_class: int | None = None,
    seed: int = 1993,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = data_manager.get_dataset(class_ids, source=source, mode="test")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=learner._device.type == "cuda",
    )
    features, labels, predictions = [], [], []
    if torch.is_grad_enabled() is False:
        raise AssertionError("global grad state was unexpectedly disabled before audit context")
    with torch.no_grad():
        if torch.is_grad_enabled():
            raise AssertionError("torch.no_grad leakage guard failed")
        for _, inputs, targets in loader:
            inputs = inputs.to(learner._device, non_blocking=True)
            output = learner._network(inputs)
            features.append(output["sfeatures"].detach().cpu().float().numpy())
            labels.append(targets.numpy().astype(np.int64, copy=False))
            predictions.append(output["logits"].argmax(dim=1).detach().cpu().numpy())
    feature_array = np.concatenate(features) if features else np.empty((0, learner.feature_dim), np.float32)
    label_array = np.concatenate(labels) if labels else np.empty(0, np.int64)
    prediction_array = np.concatenate(predictions) if predictions else np.empty(0, np.int64)
    if max_samples_per_class is not None:
        rng = np.random.RandomState(seed)
        keep = []
        for class_id in class_ids:
            indices = np.flatnonzero(label_array == int(class_id))
            if indices.size > max_samples_per_class:
                indices = np.sort(rng.choice(indices, max_samples_per_class, replace=False))
            keep.extend(indices.tolist())
        keep = np.asarray(sorted(keep), dtype=np.int64)
        feature_array, label_array, prediction_array = (
            feature_array[keep], label_array[keep], prediction_array[keep]
        )
    return feature_array, label_array, prediction_array


def class_statistics(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: Iterable[int],
    covariance_jitter: float,
) -> dict[int, ClassStats]:
    result: dict[int, ClassStats] = {}
    dimension = int(features.shape[1])
    identity = np.eye(dimension, dtype=np.float64)
    for class_id in class_ids:
        values = np.asarray(features[labels == int(class_id)], dtype=np.float64)
        if not values.size:
            raise RuntimeError(f"no extracted samples for class {class_id}")
        if values.shape[0] < 2:
            covariance = identity * covariance_jitter
        else:
            covariance = np.cov(values, rowvar=False) + identity * covariance_jitter
        result[int(class_id)] = ClassStats(
            mean=values.mean(axis=0), covariance=covariance, features=values
        )
    return result


def birth_task(class_id: int, partition: list[int]) -> int:
    boundary = 0
    for task, size in enumerate(partition, start=1):
        boundary += int(size)
        if class_id < boundary:
            return task
    raise ValueError(f"class {class_id} outside partition {partition}")


def release_learner(learner) -> None:
    del learner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
