#!/usr/bin/env python3
"""Dataset-free CPU integration test for S1-S4 and required logging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT.parent / "structure_multicenter_separation"
DEFAULT_E3 = ROOT.parent / "structure_e3_followup"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--e3-root", default=str(DEFAULT_E3))
    parser.add_argument("--output", default=str(ROOT / "reports" / "smoke" / "latest.json"))
    return parser.parse_args()


def main():
    cli = parse_args()
    sys.path.insert(0, str(Path(cli.base_root).resolve()))
    sys.path.insert(0, str(Path(cli.e3_root).resolve()))
    sys.path.insert(0, str(ROOT))

    import numpy as np
    import torch
    from utils.distribution_regions import SeparationCache

    from similarity.learner import SimilarityAwareLearner
    from similarity.losses import (
        cosine_similarity_matrix,
        hard_relative_relation,
        legacy_absolute_relation,
        similarity_tau_relation,
        similarity_weighted_scc,
        soft_hybrid_relation,
    )

    torch.manual_seed(1993)
    dimension, old_count, new_count = 8, 3, 3
    regions = []
    for class_id in range(old_count):
        regions.append(
            {
                "class_id": class_id,
                "component_id": 0,
                "birth_task": 0,
                "feature_space": "smoke",
                "mean": torch.randn(dimension) * 0.15 + class_id * 0.4,
                "covariance": torch.diag(torch.linspace(0.4, 1.1, dimension)),
                "tau_by_covariance": {"diagonal": 8.0},
            }
        )
    cache = SeparationCache(
        regions,
        "diagonal",
        "smoke",
        {
            "sep_region_chunk_size": 2,
            "sep_tau_floor": 1e-8,
            "sep_cov_eigen_floor": 1e-6,
            "sep_factor_device": "cpu",
        },
    )
    old = torch.stack([row["mean"] for row in regions])
    q_old = old + torch.tensor([[0.02], [0.12], [0.24]])
    current = torch.randn(new_count, dimension) * 0.2 + 0.6
    similarities = cosine_similarity_matrix(q_old, old)
    # Force all routing branches to be exercised deterministically.
    similarities = torch.tensor(
        [[0.90, 0.20, -0.10], [0.65, 0.80, 0.10], [0.30, 0.40, 0.55]],
        dtype=torch.float32,
    )
    features = torch.nn.Parameter(torch.randn(6, dimension) * 0.3 + 0.6)
    targets = torch.tensor([3, 3, 4, 4, 5, 5])

    legacy = legacy_absolute_relation(features, cache)
    s1 = similarity_tau_relation(features, targets, cache, similarities, old_count, alpha=0.5)
    s2 = hard_relative_relation(
        features, targets, cache, old, current, similarities, old_count, beta=0.75, margin=0.05
    )
    s4 = soft_hybrid_relation(
        features,
        targets,
        cache,
        old,
        current,
        similarities,
        old_count,
        beta=0.75,
        temperature=0.1,
        margin=0.05,
    )
    if not s2.diagnostics["used_relative"].any() or s2.diagnostics["used_relative"].all():
        raise AssertionError("S2 smoke did not exercise both hard routes")
    gates = s4.diagnostics["gate_weight"]
    if not ((gates > 0).all() and (gates < 1).all()):
        raise AssertionError("S4 gate is not a soft probability")
    if not torch.all(s1.diagnostics["all_pair_taus"] <= cache.taus.max() + 1e-6):
        raise AssertionError("S1 similarity scale unexpectedly enlarged tau")

    outputs = {
        "sfeatures": features,
        "tfeatures": features.detach() + 0.05,
    }
    scc = similarity_weighted_scc(
        outputs,
        old.numpy(),
        targets,
        similarities,
        old_count,
        alpha=0.5,
        minimum_weight=0.5,
    )
    if not ((scc.diagnostics["scc_weight"] >= 0.5).all() and (scc.diagnostics["scc_weight"] <= 1).all()):
        raise AssertionError("S3 SCC weights violate configured bounds")
    combined = legacy.loss + s1.loss + s2.loss + s4.loss + scc.weighted_loss
    combined.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise AssertionError("similarity losses produced invalid gradients")

    class DummyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(dimension, dimension)
            self.fc = torch.nn.Linear(dimension, old_count + new_count, bias=False)

        def forward(self, inputs, label=None):
            student = self.encoder(inputs)
            teacher = inputs.detach() + 0.05
            logits = torch.nn.functional.linear(
                torch.nn.functional.normalize(student, dim=1),
                torch.nn.functional.normalize(self.fc.weight, dim=1),
            )
            return {"sfeatures": student, "tfeatures": teacher, "logits": logits}

        def ca_forward(self, values):
            return {
                "logits": torch.nn.functional.linear(
                    torch.nn.functional.normalize(values, dim=1),
                    torch.nn.functional.normalize(self.fc.weight, dim=1),
                )
            }

    with tempfile.TemporaryDirectory(prefix="similarity_smoke_") as temporary:
        run_dir = Path(temporary)
        learner = SimilarityAwareLearner.__new__(SimilarityAwareLearner)
        learner.args = {
            "experiment_id": "S3_SMOKE",
            "seed": 1993,
            "scale": 2.0,
            "margin": 0.2,
            "scc": True,
            "scc_value": 5.0,
            "feature": False,
            "logit": False,
            "old_new_separation": True,
            "separation_covariance": "diagonal",
            "sep_lambda": 0.3,
            "similarity_relation_mode": "hard_relative",
            "relative_margin": 0.05,
            "similarity_scc_enabled": True,
            "similarity_scc_alpha": 0.5,
            "similarity_scc_min_weight": 0.5,
            "soft_gate_temperature": 0.1,
            "prototype_refresh_epochs": 1,
            "sep_refresh_epochs": 1,
            "diagnostic_grad_interval": 1,
            "diagnostic_sample_rows": True,
            "checkpoint_every_epoch": False,
        }
        learner._cur_task = 1
        learner._known_classes = old_count
        learner._total_classes = old_count + new_count
        learner._device = torch.device("cpu")
        learner._network = DummyNetwork()
        learner._class_means = old.numpy()
        learner._separation_cache = cache
        learner._followup_old_prototypes = old
        learner._followup_current_prototypes = current
        learner._similarity_matrix = similarities
        learner._similarity_beta = 0.75
        learner._adaptive_rows = []
        learner.tuned_epochs = 1
        learner.run_dir = run_dir
        learner._refresh_followup_cache = lambda epoch: None
        learner._compute_accuracy = lambda network, loader: 0.0
        batch_inputs = torch.randn(6, dimension)
        batch = (torch.arange(6), batch_inputs, targets)
        optimizer = torch.optim.SGD(learner._network.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        learner._init_train([batch], [], optimizer, scheduler)
        required = (
            "epoch_metrics.csv",
            "feature_norm.csv",
            "geometry_diagnostics.csv",
            "gradient_diagnostics.csv",
            "relative_margin_diagnostics.csv",
            "scc_similarity_diagnostics.csv",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise AssertionError(f"integrated learner did not write: {missing}")

        with (run_dir / "class_accuracy.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("trained_task", "eval_stage", "mapped_class_id", "accuracy"),
            )
            writer.writeheader()
            for class_id in range(old_count):
                writer.writerow(
                    {
                        "trained_task": 1,
                        "eval_stage": "task_final",
                        "mapped_class_id": class_id,
                        "accuracy": 50.0,
                    }
                )
        learner.test_loader = object()
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = np.array([[0], [3], [2], [0], [4], [1]])
        learner._write_evaluation_diagnostics(y_pred, y_true)
        if not (run_dir / "class_forgetting.csv").is_file():
            raise AssertionError("class forgetting log is missing")
        if not (run_dir / "cross_task_confusion.csv").is_file():
            raise AssertionError("confusion log is missing")

    values = [legacy.loss, s1.loss, s2.loss, s4.loss, scc.weighted_loss, combined]
    if not all(math.isfinite(float(value.detach())) for value in values):
        raise AssertionError("a smoke scalar is NaN/Inf")
    payload = {
        "status": "PASS",
        "device": "cpu",
        "torch": torch.__version__,
        "losses": {
            "B1": float(legacy.loss.detach()),
            "S1": float(s1.loss.detach()),
            "S2": float(s2.loss.detach()),
            "S3_SCC": float(scc.weighted_loss.detach()),
            "S4": float(s4.loss.detach()),
        },
        "s2_relative_active_ratio": float(s2.diagnostics["used_relative"].float().mean()),
        "s4_gate_mean": float(gates.mean()),
        "learner_epoch_integration": True,
        "evaluation_logging": True,
    }
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

