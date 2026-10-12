#!/usr/bin/env python3
"""Tiny CPU-only functional check; no dataset or pretrained model is loaded."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT.parent / "structure_multicenter_separation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--output", default=str(ROOT / "reports" / "smoke" / "latest.json"))
    return parser.parse_args()


def main():
    cli = parse_args()
    base_root = Path(cli.base_root).expanduser().resolve()
    sys.path.insert(0, str(base_root))
    sys.path.insert(0, str(ROOT))

    import torch
    from utils.distribution_regions import SeparationCache

    from followup.adaptive_margin import recalibrate_taus
    from followup.ceos import sample_ceos
    from followup.diagnostics import component_gradient_metrics
    from followup.learner import FollowupLearner
    from followup.losses import (
        angular_hard_negative_loss,
        diagonal_region_loss,
        pic_loss,
    )
    from followup.snapshot import create_run_snapshot
    from followup.stage2 import run_ceos_classifier_retraining

    torch.manual_seed(1993)
    dimension = 8
    regions = []
    for class_id in range(3):
        mean = torch.randn(dimension) * 0.2 + class_id
        covariance = torch.diag(torch.linspace(0.4, 1.1, dimension))
        regions.append(
            {
                "class_id": class_id,
                "component_id": 0,
                "birth_task": 0,
                "feature_space": "smoke",
                "mean": mean,
                "covariance": covariance,
                "tau_by_covariance": {"diagonal": 8.0},
            }
        )
    cache_config = {
        "sep_region_chunk_size": 2,
        "sep_tau_floor": 1e-8,
        "sep_cov_eigen_floor": 1e-6,
        "sep_factor_device": "cpu",
    }
    cache = SeparationCache(regions, "diagonal", "smoke", cache_config)
    features = torch.nn.Parameter(torch.randn(6, dimension) * 0.4 + 0.8)
    targets = torch.tensor([3, 3, 4, 4, 5, 5])
    old_prototypes = torch.stack([region["mean"] for region in regions])
    current_prototypes = torch.randn(3, dimension) * 0.2 + 0.8

    base_loss, _ = cache.loss(features)
    diagonal = diagonal_region_loss(features, cache)
    if not torch.allclose(base_loss, diagonal.loss, atol=1e-7, rtol=1e-6):
        raise AssertionError("follow-up diagonal loss differs from base E3")
    normalized = diagonal_region_loss(features, cache, normalize=True)
    pic = pic_loss(
        features,
        targets,
        old_prototypes,
        current_prototypes,
        known_classes=3,
        temperature=0.07,
    )
    ahn = angular_hard_negative_loss(
        features,
        targets,
        old_prototypes,
        current_prototypes,
        known_classes=3,
        margin=0.5,
        top_k=2,
    )
    gradient_metrics = component_gradient_metrics(
        {"separation": diagonal.loss, "pic": pic.loss, "ahn": ahn.loss}, [features]
    )
    optimizer = torch.optim.SGD([features], lr=1e-3)
    optimizer.zero_grad()
    combined = diagonal.loss + normalized.loss + pic.loss + ahn.loss
    combined.backward()
    if features.grad is None or not torch.isfinite(features.grad).all():
        raise AssertionError("combined follow-up loss produced an invalid gradient")
    optimizer.step()

    adaptive_results = {}
    for mode in ("variance", "age", "drift"):
        result = recalibrate_taus(
            birth_taus=cache.taus,
            variances=cache.factors,
            class_ids=cache.class_ids,
            birth_tasks=torch.tensor([0, 0, 1]),
            current_task=2,
            drift_norms=torch.tensor([0.1, 0.4, 0.8]),
            mode=mode,
            strength=0.25,
        )
        if not torch.isfinite(result.taus).all() or (result.taus <= 0).any():
            raise AssertionError(f"invalid adaptive taus for {mode}")
        adaptive_results[mode] = result.taus.tolist()

    current_features = torch.randn(12, dimension) + 2.0
    current_labels = torch.tensor([3] * 4 + [4] * 4 + [5] * 4)
    ceos50 = sample_ceos(
        old_prototypes[0], current_features, current_labels, 8, lambda_min=0.7
    )
    ceos100 = sample_ceos(
        old_prototypes[0], current_features, current_labels, 16, lambda_min=0.7
    )
    if ceos50.features.shape[0] + 8 != 16 or ceos100.features.shape[0] != 16:
        raise AssertionError("CEOS50/100 controlled replay counts are wrong")
    if not ceos50.old_dominance.all() or not ceos100.old_dominance.all():
        raise AssertionError("CEOS lambda constraint did not preserve old dominance")
    if not torch.allclose(ceos100.features.norm(dim=1), torch.ones(16), atol=1e-6):
        raise AssertionError("CEOS samples are not L2-normalized")

    # Exercise the actual subclass epoch and CEOS stage-2 entry points with a
    # tiny network. This catches integration errors without loading ViT/data.
    class DummyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(dimension, dimension)
            self.fc = torch.nn.Linear(dimension, 6, bias=False)

        def forward(self, inputs, label=None):
            student = self.encoder(inputs)
            teacher = student.detach() + 0.05
            logits = torch.nn.functional.linear(
                torch.nn.functional.normalize(student, dim=1),
                torch.nn.functional.normalize(self.fc.weight, dim=1),
            )
            return {"sfeatures": student, "tfeatures": teacher, "logits": logits}

        def ca_forward(self, features):
            logits = torch.nn.functional.linear(
                torch.nn.functional.normalize(features, dim=1),
                torch.nn.functional.normalize(self.fc.weight, dim=1),
            )
            return {"logits": logits}

    with tempfile.TemporaryDirectory(prefix="e3_followup_epoch_") as temporary:
        integration_dir = Path(temporary)
        learner = FollowupLearner.__new__(FollowupLearner)
        learner.args = {
            "experiment_id": "SMOKE_INTEGRATION",
            "seed": 1993,
            "scale": 2.0,
            "margin": 0.2,
            "scc": False,
            "scc_value": 5.0,
            "feature": False,
            "logit": False,
            "old_new_separation": True,
            "separation_covariance": "diagonal",
            "sep_lambda": 0.3,
            "normalized_diagonal": False,
            "pic_enabled": False,
            "pic_lambda": 1.0,
            "ahn_enabled": True,
            "ahn_lambda": 1.0,
            "ahn_margin": 0.5,
            "ahn_top_k": 2,
            "prototype_refresh_epochs": 1,
            "sep_refresh_epochs": 1,
            "diagnostic_grad_interval": 1,
            "checkpoint_every_epoch": False,
            "replay_samples_per_class": 4,
            "ceos_replay_fraction": 0.5,
            "ceos_lambda_min": 0.7,
            "ceos_enemy_top_k": 1,
            "ceos_log_samples": True,
        }
        learner._cur_task = 1
        learner._known_classes = 3
        learner._total_classes = 6
        learner._device = torch.device("cpu")
        learner._network = DummyNetwork()
        learner._class_means = old_prototypes.numpy()
        learner._separation_cache = cache
        learner._followup_old_prototypes = old_prototypes
        learner._followup_current_prototypes = current_prototypes
        learner._adaptive_rows = []
        learner.tuned_epochs = 1
        learner.run_dir = integration_dir
        learner._refresh_followup_cache = lambda epoch: None
        learner._compute_accuracy = lambda network, loader: 0.0
        batch_inputs = torch.randn(6, dimension)
        batch = (torch.arange(6), batch_inputs, targets)
        epoch_optimizer = torch.optim.SGD(learner._network.parameters(), lr=1e-3)
        epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(epoch_optimizer, T_max=1)
        learner._init_train([batch], [], epoch_optimizer, epoch_scheduler)
        if not (integration_dir / "epoch_metrics.csv").is_file():
            raise AssertionError("follow-up learner epoch did not write metrics")
        if not (integration_dir / "gradient_metrics.csv").is_file():
            raise AssertionError("follow-up learner epoch did not write gradients")

        learner._class_means = torch.cat(
            (old_prototypes, current_prototypes), dim=0
        ).numpy()
        learner._class_covs = torch.eye(dimension).repeat(6, 1, 1) * 0.2
        learner._collect_current_features = lambda key: (current_features.cpu(), current_labels.cpu())
        learner._compute_accuracy = lambda network, loader: 0.0
        learner.test_loader = []
        learner.task_sizes = [3, 3]
        learner.logit_norm = None
        learner.init_lr = 1e-3
        learner.weight_decay = 0.0
        run_ceos_classifier_retraining(learner, task_size=3, ca_epochs=1)
        if not (integration_dir / "ceos_summary.csv").is_file():
            raise AssertionError("CEOS stage-2 integration did not write its summary")
        if not list((integration_dir / "ceos_samples").glob("*.csv")):
            raise AssertionError("CEOS stage-2 integration did not write sample diagnostics")

    with tempfile.TemporaryDirectory(prefix="e3_followup_smoke_") as temporary:
        temporary_root = Path(temporary)
        fake_base = temporary_root / "base"
        fake_extension = temporary_root / "extension"
        run_dir = temporary_root / "run"
        fake_base.mkdir()
        fake_extension.mkdir()
        (fake_base / "base.py").write_text("VALUE = 1\n", encoding="utf-8")
        (fake_extension / "extension.py").write_text("VALUE = 2\n", encoding="utf-8")
        create_run_snapshot(
            run_dir,
            config={"experiment_id": "SMOKE"},
            base_root=fake_base,
            extension_root=fake_extension,
            launch_command="smoke",
        )
        if not (run_dir / "source_snapshot" / "manifest.json").is_file():
            raise AssertionError("source snapshot was not created")

    all_scalars = [
        float(base_loss),
        float(normalized.loss),
        float(pic.loss),
        float(ahn.loss),
        float(combined),
    ]
    if not all(math.isfinite(value) for value in all_scalars):
        raise AssertionError("a smoke loss is NaN/Inf")
    payload = {
        "status": "PASS",
        "torch": torch.__version__,
        "device": "cpu",
        "base_diagonal_parity": True,
        "optimizer_step": True,
        "learner_epoch_integration": True,
        "ceos_stage2_integration": True,
        "losses": {
            "diagonal": float(base_loss.detach()),
            "normalized_diagonal": float(normalized.loss.detach()),
            "pic": float(pic.loss.detach()),
            "ahn": float(ahn.loss.detach()),
        },
        "gradient_metrics": gradient_metrics,
        "adaptive_taus": adaptive_results,
        "ceos": {
            "ceos50_count": int(ceos50.features.shape[0]),
            "ceos100_count": int(ceos100.features.shape[0]),
            "old_dominance_rate": float(ceos100.old_dominance.float().mean()),
        },
        "snapshot": True,
    }
    output = Path(cli.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
