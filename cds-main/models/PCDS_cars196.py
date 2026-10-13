"""Cars-196 PCDS and PCDS+SGDS controlled comparison runner."""

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

from cars196_static_check import run_static_checks  # noqa: E402
from models.sgds_activation_guidance import SGDSActivationGuidance  # noqa: E402


def guidance_factory(config: dict, feature_dim: int) -> SGDSActivationGuidance:
    if [int(value) for value in config["sgds_target_layers"]] != [-1]:
        raise ValueError("Cars comparison targets only the final ViT block")
    return SGDSActivationGuidance(
        feature_dim=feature_dim,
        num_classes=int(config["num_classes"]),
        # Task 0 is updated to its actual duration at prepare_task().
        num_epochs=int(config["initial_epochs"]),
        target_layer=11,
        paper_k=float(config["sgds_sparsity"]),
        beta=float(config["sgds_beta"]),
        gamma=float(config["sgds_gamma"]),
        exploration_ratio=float(config["sgds_exploration_ratio"]),
        epsilon=float(config["sgds_epsilon"]),
    )


def run_from_config(config: dict) -> Path:
    if config.get("experiment_name") != "cars196_10task":
        raise ValueError("Unexpected Cars-196 experiment name")
    method = str(config.get("method_label"))
    if method not in {"pcds_full", "pcds_structure_sgds"}:
        raise ValueError("Cars runner supports pcds_full or pcds_structure_sgds")
    use_sgds = bool(config.get("use_sgds", False))
    if use_sgds != (method == "pcds_structure_sgds"):
        raise ValueError("method_label and use_sgds disagree")

    run_static_checks(config, require_gpu=True)
    output_dir = (
        Path(config["output_root"])
        / str(config["experiment_name"])
        / f"seed_{int(config['seed'])}"
        / method
    )
    completed = output_dir / "COMPLETED.json"
    if completed.exists():
        print(f"[Cars-196 {method}] completed training found: {output_dir}")
    else:
        resume = (output_dir / "checkpoint_latest.pt").exists()
        output_dir = run_experiment(
            config,
            method=method,
            repo_root=CDS_ROOT,
            resume=resume,
            activation_guidance_factory=(guidance_factory if use_sgds else None),
        )

    if use_sgds:
        from analyze_cars196_pcds_sgds import analyze_results

        analyze_results(config, output_dir)
    else:
        print(
            "[Cars-196 PCDS] baseline complete. Run the SGDS config next to "
            "generate the comparison report."
        )

    with completed.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    print(
        "[Cars-196 {}] complete: final_accuracy={:.4f}, "
        "avg_incremental={:.4f}".format(
            method,
            result["final_accuracy"],
            result["average_incremental_accuracy"],
        )
    )
    return output_dir
