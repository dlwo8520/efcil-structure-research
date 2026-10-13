#!/usr/bin/env python3
"""Dataset-free CPU test for control selection, loss parity, and logging."""

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
DEFAULT_SIMILARITY = ROOT.parent / "structure_similarity_aware_followup"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--e3-root", default=str(DEFAULT_E3))
    parser.add_argument("--similarity-root", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--output", default=str(ROOT / "reports" / "smoke" / "latest.json"))
    return parser.parse_args()


def main():
    cli = parse_args()
    for root in (cli.base_root, cli.e3_root, cli.similarity_root, ROOT):
        sys.path.insert(0, str(Path(root).resolve()))

    import numpy as np
    import torch
    from followup.losses import diagonal_region_loss
    from similarity.losses import legacy_absolute_relation
    from utils.distribution_regions import SeparationCache

    from control.learner import ControlLearner
    from control.losses import selective_control_relation

    torch.manual_seed(1993)
    dimension, old_count, new_count = 8, 8, 3
    regions = []
    for class_id in range(old_count):
        regions.append(
            {
                "class_id": class_id,
                "component_id": 0,
                "birth_task": 0,
                "feature_space": "control_smoke",
                "mean": torch.randn(dimension) * 0.1 + class_id * 0.08,
                "covariance": torch.diag(torch.linspace(0.4, 1.1, dimension)),
                "tau_by_covariance": {"diagonal": 8.0},
            }
        )
    cache = SeparationCache(
        regions,
        "diagonal",
        "control_smoke",
        {
            "sep_region_chunk_size": 4,
            "sep_tau_floor": 1e-8,
            "sep_cov_eigen_floor": 1e-6,
            "sep_factor_device": "cpu",
        },
    )
    features = torch.nn.Parameter(torch.randn(6, dimension) * 0.2 + 0.4)
    targets = torch.tensor([8, 8, 9, 9, 10, 10])
    all_one = selective_control_relation(features, cache, torch.ones(6, old_count))
    legacy = legacy_absolute_relation(features, cache)
    inherited = diagonal_region_loss(features, cache)
    if not torch.allclose(all_one.loss, legacy.loss, atol=1e-7, rtol=1e-6):
        raise AssertionError("C0 is not identical to similarity-layer AM-Drift")
    if not torch.allclose(all_one.loss, inherited.loss, atol=1e-7, rtol=1e-6):
        raise AssertionError("C0 is not identical to inherited E3 AM-Drift")

    learner = ControlLearner.__new__(ControlLearner)
    learner.args = {"seed": 1993, "control_fraction": 0.25}
    learner._cur_task = 1
    before = torch.get_rng_state().clone()
    random_a = learner._random_mask(new_count, old_count, 2)
    after = torch.get_rng_state().clone()
    random_b = learner._random_mask(new_count, old_count, 2)
    if not torch.equal(before, after):
        raise AssertionError("C3 selection changed the global torch RNG")
    if not torch.equal(random_a, random_b):
        raise AssertionError("C3 selection is not deterministic")
    similarities = torch.tensor(
        [
            [0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.0],
            [0.1, 0.8, 0.3, 0.6, 0.9, 0.2, 0.7, 0.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        ]
    )
    high = torch.zeros_like(similarities, dtype=torch.bool)
    low = torch.zeros_like(similarities, dtype=torch.bool)
    ranks = torch.empty_like(similarities, dtype=torch.long)
    for index, row in enumerate(similarities):
        desc = learner._stable_order(row, True)
        asc = learner._stable_order(row, False)
        high[index, desc[:2]] = True
        low[index, asc[:2]] = True
        ranks[index, desc] = torch.arange(1, old_count + 1)
    if not (
        torch.equal(high.sum(dim=1), torch.full((new_count,), 2))
        and torch.equal(low.sum(dim=1), torch.full((new_count,), 2))
        and torch.equal(random_a.sum(dim=1), torch.full((new_count,), 2))
    ):
        raise AssertionError("C1/C2/C3 do not select equal pair counts")

    weights = torch.ones_like(similarities)
    weights[random_a] = 0.0
    high_off = torch.ones_like(similarities)
    high_off[high] = 0.0
    low_off = torch.ones_like(similarities)
    low_off[low] = 0.0
    high_weak = torch.ones_like(similarities)
    high_weak[high] = 0.3
    if not (
        torch.equal(high_off.eq(0), high)
        and torch.equal(low_off.eq(0), low)
        and torch.equal(weights.eq(0), random_a)
        and torch.allclose(high_weak[high], torch.full_like(high_weak[high], 0.3))
        and torch.all(high_weak[~high] == 1)
    ):
        raise AssertionError("C1/C2/C3/C4 pair weights do not match their selected masks")
    selective = selective_control_relation(features, cache, weights[targets - old_count])
    selective.loss.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise AssertionError("control loss produced an invalid gradient")

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
            logits = torch.nn.functional.linear(
                torch.nn.functional.normalize(values, dim=1),
                torch.nn.functional.normalize(self.fc.weight, dim=1),
            )
            return {"logits": logits}

    with tempfile.TemporaryDirectory(prefix="control_smoke_") as temporary:
        run_dir = Path(temporary)
        integrated = ControlLearner.__new__(ControlLearner)
        integrated.args = {
            "experiment_id": "C3_SMOKE",
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
            "similarity_relation_mode": "legacy_absolute",
            "similarity_scc_enabled": False,
            "prototype_refresh_epochs": 1,
            "sep_refresh_epochs": 1,
            "diagnostic_grad_interval": 0,
            "diagnostic_sample_rows": False,
            "checkpoint_every_epoch": False,
            "control_selection_mode": "random_off",
            "control_fraction": 0.25,
            "control_weak_weight": 0.3,
        }
        integrated._cur_task = 1
        integrated._known_classes = old_count
        integrated._total_classes = old_count + new_count
        integrated._device = torch.device("cpu")
        integrated._network = DummyNetwork()
        integrated._class_means = torch.stack([row["mean"] for row in regions]).numpy()
        integrated._separation_cache = cache
        integrated._followup_old_prototypes = torch.as_tensor(integrated._class_means)
        integrated._followup_current_prototypes = torch.randn(new_count, dimension)
        integrated._similarity_matrix = similarities
        integrated._similarity_beta = 0.75
        integrated._control_high_mask = high
        integrated._control_low_mask = low
        integrated._control_random_mask = random_a
        integrated._control_pair_weights = weights
        integrated._control_rank = ranks
        integrated._adaptive_rows = [
            {
                "class_id": index,
                "tau_birth": 8.0,
                "adaptive_scale": 1.0,
                "tau": 8.0,
            }
            for index in range(old_count)
        ]
        integrated.tuned_epochs = 1
        integrated.run_dir = run_dir
        integrated._refresh_followup_cache = lambda epoch: None
        integrated._compute_accuracy = lambda network, loader: 0.0
        batch = (torch.arange(6), torch.randn(6, dimension), targets)
        optimizer = torch.optim.SGD(integrated._network.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        integrated._init_train([batch], [], optimizer, scheduler)
        required = (
            "control_pair_separation.csv",
            "control_intrusion.csv",
            "epoch_metrics.csv",
            "feature_norm.csv",
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise AssertionError(f"integrated learner did not write: {missing}")
        pair_rows = list(csv.DictReader((run_dir / "control_pair_separation.csv").open()))
        if len(pair_rows) != new_count * old_count:
            raise AssertionError("pair-separation log has the wrong row count")

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
        y_true = np.arange(old_count + new_count)
        y_pred = np.asarray([[value] for value in [0, 1, 2, 3, 4, 5, 8, 7, 0, 9, 1]])
        integrated._write_evaluation_diagnostics(y_pred, y_true)
        for name in ("control_confusion.csv", "control_group_metrics.csv", "class_forgetting.csv"):
            if not (run_dir / name).is_file():
                raise AssertionError(f"evaluation log missing: {name}")

    values = (all_one.loss, selective.loss)
    if not all(math.isfinite(float(value.detach())) for value in values):
        raise AssertionError("a control loss is NaN/Inf")
    payload = {
        "status": "PASS",
        "device": "cpu",
        "torch": torch.__version__,
        "c0_e3_parity": True,
        "equal_pair_counts": True,
        "random_deterministic": True,
        "random_preserves_global_rng": True,
        "finite_gradient": True,
        "learner_epoch_integration": True,
        "evaluation_logging": True,
        "losses": {
            "C0": float(all_one.loss.detach()),
            "C3": float(selective.loss.detach()),
        },
    }
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
