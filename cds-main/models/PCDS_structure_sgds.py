"""Entry point for PCDS + SGDS activation guidance on the matched protocol."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import json
import sys
from pathlib import Path


CDS_ROOT = Path(__file__).resolve().parents[1]
EFCIL_ROOT = Path(("${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)))
if str(EFCIL_ROOT) not in sys.path:
    sys.path.insert(0, str(EFCIL_ROOT))

from efcil_diag.pcds_experiment import run_experiment  # noqa: E402

from models.sgds_activation_guidance import SGDSActivationGuidance  # noqa: E402
from sgds_static_check import run_static_checks  # noqa: E402


def guidance_factory(config: dict, feature_dim: int) -> SGDSActivationGuidance:
    target_layers = [int(value) for value in config["sgds_target_layers"]]
    if target_layers != [-1]:
        raise ValueError("This controlled experiment targets only the final ViT block")
    num_classes = int(config.get("num_classes", 0))
    if not num_classes:
        if "task_sizes" in config:
            num_classes = sum(int(value) for value in config["task_sizes"])
        else:
            num_classes = int(config["num_tasks"]) * int(config["classes_per_task"])
    return SGDSActivationGuidance(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_epochs=int(config["epochs_per_task"]),
        target_layer=11,
        paper_k=float(config["sgds_sparsity"]),
        beta=float(config["sgds_beta"]),
        gamma=float(config["sgds_gamma"]),
        exploration_ratio=float(config["sgds_exploration_ratio"]),
        epsilon=float(config["sgds_epsilon"]),
    )


def run_from_config(config: dict) -> Path:
    if config.get("experiment_name") != "cifar100_10task":
        raise ValueError("Unexpected experiment name")
    if not bool(config.get("use_sgds", False)):
        raise ValueError("pcds_structure_sgds runner requires use_sgds=true")
    run_static_checks(config, require_gpu=True)

    output_dir = (
        Path(config["output_root"])
        / str(config["experiment_name"])
        / f"seed_{int(config['seed'])}"
        / "pcds_structure_sgds"
    )
    completed = output_dir / "COMPLETED.json"
    if completed.exists():
        print(f"[PCDS+SGDS] completed training found; analysis only: {output_dir}")
    else:
        resume = (output_dir / "checkpoint_latest.pt").exists()
        output_dir = run_experiment(
            config,
            method="pcds_structure_sgds",
            repo_root=CDS_ROOT,
            resume=resume,
            activation_guidance_factory=guidance_factory,
        )

    from analyze_pcds_structure_sgds import analyze_results

    analyze_results(config, output_dir)
    with completed.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    print(
        "[PCDS+SGDS] complete: final_accuracy={:.4f}, avg_incremental={:.4f}".format(
            result["final_accuracy"], result["average_incremental_accuracy"]
        )
    )
    return output_dir
