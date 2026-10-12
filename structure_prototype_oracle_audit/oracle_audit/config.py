from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT.parent
BASE_ROOT = HOME / "structure_multicenter_separation"
E3_ROOT = HOME / "structure_e3_followup"
SIMILARITY_ROOT = HOME / "structure_similarity_aware_followup"
SWEEP_ROOT = HOME / "structure_selective_similarity_sweep"
PROTOCOLS = ("IA5", "IA10", "CAR5", "CAR10")
DEFAULT_DATA_ROOTS = {
    "IA5": HOME / "datasets/ptm_guided/imagenet-a",
    "IA10": HOME / "datasets/ptm_guided/imagenet-a",
    "CAR5": HOME / "cds-main/data/cars196",
    "CAR10": HOME / "cds-main/data/cars196",
}
LEGACY_FINAL_CHECKPOINTS = {
    "IA5": HOME / "structure_e3_followup/outputs/imageneta/B0_STRUCTURE/seed_1993/checkpoint_latest.pt",
    "CAR5": HOME / "structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/checkpoint_latest.pt",
}


def bootstrap_paths() -> None:
    """Install dependency roots in the same priority used by the sweep."""
    for path, marker in (
        (BASE_ROOT, "trainer.py"),
        (E3_ROOT, "followup/learner.py"),
        (SIMILARITY_ROOT, "similarity/learner.py"),
        (SWEEP_ROOT, "selective/protocol.py"),
    ):
        if not (path / marker).is_file():
            raise FileNotFoundError(f"invalid dependency root {path}: missing {marker}")
    for path in (BASE_ROOT, E3_ROOT, SIMILARITY_ROOT, SWEEP_ROOT, ROOT):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    visited = set() if seen is None else set(seen)
    if path in visited:
        raise ValueError(f"circular config inheritance: {path}")
    visited.add(path)
    current = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in current:
        return current
    return merge(load_config(path.parent / current["extends"], visited), current)


def configure_device(gpu: str | None, cpu: bool = False) -> tuple[list[str], list[int]]:
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return [], [-1]
    physical = [item.strip() for item in str(gpu).split(",") if item.strip()]
    if not physical or any(not item.isdigit() for item in physical):
        raise ValueError("--gpu must contain non-negative physical GPU IDs")
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)
    return physical, list(range(len(physical)))


def protocol_config(protocol: str, data_root: str | Path | None = None) -> dict[str, Any]:
    protocol = protocol.upper()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; choose {PROTOCOLS}")
    bootstrap_paths()
    from selective.registry import protocol_config_path, structure_method

    config = merge(load_config(protocol_config_path(protocol)), structure_method())
    config.update(
        {
            "experiment_id": "Structure_AUDIT",
            "method": "Structure",
            "method_name": "vanilla Structure (audit-ready snapshots)",
            "data_root": str(Path(data_root or DEFAULT_DATA_ROOTS[protocol]).expanduser().resolve()),
        }
    )
    return config


def validate_protocol_config(config: dict[str, Any]) -> None:
    bootstrap_paths()
    from selective.protocol import validate_partition

    validate_partition(config["protocol_id"], config["task_partition"], config["expected_num_classes"])
    expected = {
        "method": "Structure",
        "old_new_separation": False,
        "selective_method": "none",
        "adaptive_margin_mode": "none",
        "similarity_relation_mode": "none",
        "similarity_scc_enabled": False,
        "multi_center": False,
        "apr": True,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if float(config.get("sep_lambda", 0.0)) != 0.0:
        mismatches["sep_lambda"] = {"expected": 0.0, "actual": config.get("sep_lambda")}
    if mismatches:
        raise ValueError(f"not a vanilla Structure audit config: {mismatches}")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {"outputs", "training_outputs", "__pycache__"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def launch_command() -> str:
    return " ".join(shlex.quote(item) for item in sys.argv)
