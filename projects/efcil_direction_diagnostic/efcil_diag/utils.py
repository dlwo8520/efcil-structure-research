from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


GOD_SEED_1993_ORDER = [
    68, 56, 78, 8, 23, 84, 90, 65, 74, 76,
    40, 89, 3, 92, 55, 9, 26, 80, 43, 38,
    58, 70, 77, 1, 85, 19, 17, 50, 28, 53,
    13, 81, 45, 82, 6, 59, 83, 16, 15, 44,
    91, 41, 72, 60, 79, 52, 20, 10, 31, 54,
    37, 95, 14, 71, 96, 98, 97, 2, 64, 66,
    42, 22, 35, 86, 24, 34, 87, 21, 99, 0,
    88, 27, 18, 94, 11, 12, 47, 25, 30, 46,
    62, 69, 36, 61, 7, 63, 75, 5, 32, 4,
    51, 48, 73, 93, 39, 67, 29, 49, 57, 33,
]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError("Configuration root must be a mapping")
    return config


def resolve_class_order(value: str | list[int], num_classes: int = 100) -> list[int]:
    if isinstance(value, list):
        order = [int(item) for item in value]
    elif value == "god_seed1993":
        order = GOD_SEED_1993_ORDER.copy()
    elif value == "identity":
        order = list(range(100))
    elif value == "seed0":
        generator = np.random.RandomState(0)
        order = generator.permutation(num_classes).tolist()
    elif value == "cds_cars196_seed1993":
        if num_classes != 196:
            raise ValueError("cds_cars196_seed1993 requires num_classes=196")
        # Released CDS Cars config uses shuffle=true and primary seed 1993.
        # Keep this class-order seed separate from the stochastic training seed.
        generator = np.random.RandomState(1993)
        order = generator.permutation(196).tolist()
    else:
        raise ValueError(f"Unknown class order: {value}")
    if sorted(order) != list(range(num_classes)):
        raise ValueError(
            f"Class order must be a permutation of [0, {num_classes - 1}]"
        )
    return order


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"efcil_diag:{output_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "stdout_stderr.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def run_readonly(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def assert_physical_gpu_one() -> dict[str, Any]:
    return assert_physical_gpu(
        expected_index=1,
        expected_uuid="afb08c1c-8a66-104a-01d1-046df70f0552",
    )


def assert_physical_gpu(expected_index: int, expected_uuid: str) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_visible = str(int(expected_index))
    if visible != expected_visible:
        raise RuntimeError(
            f"This experiment is authorized only with CUDA_VISIBLE_DEVICES={expected_visible}; "
            f"received {visible!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU or another GPU fallback is forbidden")
    count = torch.cuda.device_count()
    if count != 1:
        raise RuntimeError(f"Expected one visible GPU after masking, found {count}")
    current = torch.cuda.current_device()
    if current != 0:
        raise RuntimeError(f"Expected logical cuda:0, found cuda:{current}")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = str(properties.uuid).lower()
    expected_uuid = str(expected_uuid).lower()
    if actual_uuid != expected_uuid:
        raise RuntimeError(
            f"Visible GPU UUID does not match physical GPU {expected_index}: "
            f"expected {expected_uuid}, found {actual_uuid}"
        )
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "torch_cuda_device_count": count,
        "torch_cuda_current_device": current,
        "torch_cuda_device_name": torch.cuda.get_device_name(0),
        "torch_cuda_device_uuid": actual_uuid,
    }


def collect_environment(repo_root: Path, gpu_info: dict[str, Any]) -> dict[str, Any]:
    physical_gpu = str(gpu_info.get("CUDA_VISIBLE_DEVICES", ""))
    environment = {
        **gpu_info,
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "timm": _package_version("timm"),
        "numpy": np.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "hostname": platform.node(),
        "git_commit": run_readonly(["git", "-C", str(repo_root), "rev-parse", "HEAD"]),
        "git_status": run_readonly(["git", "-C", str(repo_root), "status", "--short"]),
        "nvidia_smi_physical_gpu": run_readonly([
            "nvidia-smi", "-i", physical_gpu,
            "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ]),
        "nvidia_smi_physical_gpu_processes": run_readonly([
            "nvidia-smi", "-i", physical_gpu,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ]),
    }
    # Keep the completed CIFAR-100 environment schema backward compatible.
    if physical_gpu == "1":
        environment["nvidia_smi_gpu1"] = environment["nvidia_smi_physical_gpu"]
        environment["nvidia_smi_gpu1_processes"] = environment[
            "nvidia_smi_physical_gpu_processes"
        ]
    return environment


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception as error:  # pragma: no cover - diagnostic only
        return f"unavailable: {error}"


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def dump_yaml(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def tensor_l2_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    total = torch.zeros((), dtype=torch.float64)
    for tensor in tensors:
        total += tensor.detach().double().pow(2).sum().cpu()
    return float(total.sqrt().item())
