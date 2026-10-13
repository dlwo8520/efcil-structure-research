"""Static and tiny-tensor checks for the controlled PCDS+SGDS run."""

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


def run_static_checks(config: dict, require_gpu: bool) -> dict:
    protocol = validate_protocol(config, CDS_ROOT, require_gpu=require_gpu)
    errors: list[str] = []
    expected = {
        "use_sgds": True,
        "sgds_target_layers": [-1],
        "sgds_sparsity": 0.6,
        "sgds_beta": 0.5,
        "sgds_gamma": 1.0,
        "sgds_exploration_ratio": 0.5,
        "sgds_eq6_includes_self": True,
        "sgds_training_only_mask": True,
    }
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            errors.append(f"{key}: expected {wanted!r}, found {config.get(key)!r}")
    if config.get("sgds_k_interpretation") != "paper_k_is_TopK_retain_ratio":
        errors.append("SGDS k must follow Algorithm 1 TopK floor(kN)")

    baseline_summary = Path(config["pcds_baseline_summary"])
    baseline_dir = Path(config["pcds_baseline_dir"])
    if not baseline_summary.exists():
        errors.append(f"PCDS baseline summary missing: {baseline_summary}")
    if not (baseline_dir / "COMPLETED.json").exists():
        errors.append(f"PCDS baseline completion marker missing: {baseline_dir}")
    if errors:
        raise RuntimeError("PCDS+SGDS static config check failed:\n- " + "\n- ".join(errors))

    device = torch.device("cuda:0" if require_gpu else "cpu")
    guidance = SGDSActivationGuidance(
        feature_dim=4,
        num_classes=4,
        num_epochs=2,
        target_layer=11,
        paper_k=0.5,
        beta=0.5,
        gamma=1.0,
        exploration_ratio=0.5,
    ).to(device)
    semantic0 = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device
    )
    strategies0 = guidance.prepare_task(0, 0, semantic0)
    if any(row["strategy"] != NEW_SUBSPACE_ALLOCATION for row in strategies0):
        raise AssertionError("Task 0 must naturally use new-subspace allocation")
    guidance.train()
    labels = torch.tensor([0, 1], device=device)
    activations = torch.ones(2, 3, 4, device=device)
    guidance.set_batch_context(labels, epoch=0)
    masked = guidance(activations)
    guidance.clear_batch_context()
    active = masked.ne(0).any(dim=1).sum(dim=1)
    if active.tolist() != [2, 2]:
        raise AssertionError(f"TopK retain check failed: {active.tolist()}")
    if int(guidance.global_counter.sum().item()) != 4:
        raise AssertionError("Global counter did not count final TopK dimensions")
    if int(guidance.class_counter[0].sum().item()) != 2 or int(
        guidance.class_counter[1].sum().item()
    ) != 2:
        raise AssertionError("Class counters were updated under the wrong labels")

    counter_before_eval = guidance.global_counter.clone()
    guidance.eval()
    evaluated = guidance(activations)
    if not torch.equal(evaluated, activations):
        raise AssertionError("Evaluation unexpectedly applied SGDS masking")
    if not torch.equal(counter_before_eval, guidance.global_counter):
        raise AssertionError("Evaluation unexpectedly changed SGDS counters")

    counter_before_task = guidance.global_counter.clone()
    semantic1 = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0], [0.0, 0.1, 0.9, 0.0]], device=device
    )
    guidance.prepare_task(1, 2, semantic1)
    if not torch.equal(counter_before_task, guidance.global_counter):
        raise AssertionError("Global counter was reset between tasks")

    result = {
        **protocol,
        "method": "pcds_structure_sgds",
        "sgds_off_path": "activation_guidance_factory=None; baseline model path unchanged",
        "target_layers": [-1],
        "paper_eq6_includes_self": True,
        "paper_k": 0.6,
        "paper_k_interpretation": "TopK floor(kN) retains 60%; requested zero ratio 40%",
        "counter_update": "only final Bernoulli+TopK nonzero dimensions",
        "counter_persists_between_tasks": True,
        "evaluation_random_mask": False,
        "tiny_tensor_sparsity_check": "PASS",
        "class_counter_label_check": "PASS",
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    config_path = CDS_ROOT / "exps_jsons/cifar100_pcds_structure_sgds.json"
    with config_path.open("r", encoding="utf-8") as handle:
        run_static_checks(json.load(handle), require_gpu=True)

