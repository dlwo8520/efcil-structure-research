"""Preflight checks for the Cars-196 PCDS/SGDS comparison."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[1])))

import json
import sys
from pathlib import Path

import torch


CDS_ROOT = Path(__file__).resolve().parent
EFCIL_ROOT = Path(("${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)))
if str(EFCIL_ROOT) not in sys.path:
    sys.path.insert(0, str(EFCIL_ROOT))

from efcil_diag.pcds_experiment import validate_protocol  # noqa: E402
from models.sgds_activation_guidance import (  # noqa: E402
    NEW_SUBSPACE_ALLOCATION,
    SGDSActivationGuidance,
)


METHOD_ONLY_KEYS = {
    "method_label",
    "use_sgds",
    "sgds_target_layers",
    "sgds_sparsity",
    "sgds_k_interpretation",
    "sgds_requested_zero_ratio",
    "sgds_beta",
    "sgds_gamma",
    "sgds_exploration_ratio",
    "sgds_epsilon",
    "sgds_eq6_includes_self",
    "sgds_training_only_mask",
    "sgds_token_reduction",
}


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Config root is not an object: {path}")
    return value


def _shared_protocol(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in METHOD_ONLY_KEYS}


def run_static_checks(config: dict, require_gpu: bool) -> dict:
    protocol = validate_protocol(config, CDS_ROOT, require_gpu=require_gpu)
    if int(config.get("physical_gpu", -1)) != 3:
        raise RuntimeError("Cars comparison is restricted to physical GPU 3")
    if str(config.get("physical_gpu_uuid", "")).lower() != (
        "d213eba7-1551-206d-b6e9-bc708945a73a"
    ):
        raise RuntimeError("Cars config does not record the physical GPU 3 UUID")

    baseline_config = _load_json(config["cars_pcds_config"])
    sgds_config = _load_json(config["cars_sgds_config"])
    if _shared_protocol(baseline_config) != _shared_protocol(sgds_config):
        raise RuntimeError("Cars PCDS and PCDS+SGDS shared protocols differ")
    if bool(config.get("use_sgds", False)):
        expected = {
            "sgds_target_layers": [-1],
            "sgds_sparsity": 0.6,
            "sgds_beta": 0.5,
            "sgds_gamma": 1.0,
            "sgds_exploration_ratio": 0.5,
            "sgds_eq6_includes_self": True,
            "sgds_training_only_mask": True,
        }
        wrong = [
            f"{key}={config.get(key)!r} (expected {value!r})"
            for key, value in expected.items()
            if config.get(key) != value
        ]
        if wrong:
            raise RuntimeError("Invalid Cars SGDS config: " + ", ".join(wrong))
        baseline_dir = Path(config["cars_pcds_output_dir"])
        if not (baseline_dir / "COMPLETED.json").exists():
            raise RuntimeError(
                "Cars PCDS baseline is not complete. Run cars196_pcds.json first."
            )
        _tiny_guidance_check(torch.device("cuda:0" if require_gpu else "cpu"))

    return {**protocol, "status": "PASS"}


def _tiny_guidance_check(device: torch.device) -> None:
    guidance = SGDSActivationGuidance(
        feature_dim=4,
        num_classes=4,
        num_epochs=4,
        target_layer=11,
        paper_k=0.5,
        beta=0.5,
        gamma=1.0,
        exploration_ratio=0.5,
    ).to(device)
    rows = guidance.prepare_task(0, 0, torch.eye(4, device=device)[:2])
    if any(row["strategy"] != NEW_SUBSPACE_ALLOCATION for row in rows):
        raise AssertionError("Task 0 strategy check failed")
    guidance.train()
    values = torch.ones(2, 3, 4, device=device)
    guidance.set_batch_context(torch.tensor([0, 1], device=device), epoch=0)
    output = guidance(values)
    guidance.clear_batch_context()
    if output.ne(0).any(dim=1).sum(dim=1).tolist() != [2, 2]:
        raise AssertionError("SGDS Top-K check failed")
