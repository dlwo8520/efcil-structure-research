"""CPU synthetic checks of matched training, covariance, RNG forks and gates."""
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from type_a_generative_augmentation.common import bootstrap, read_json, write_csv, write_json, sha256
bootstrap()
from data.data_manager import DummyDataset
from type_a_generative_augmentation.matched_views import MatchedViews, build_strong_transform, isolated_cpu_rng
from type_a_generative_augmentation.learner import AugmentationLearner
from type_a_generative_augmentation.moments import mixed_moments
from type_a_generative_augmentation.results import paired, retention_transitions, summary


class ToyNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, classes, bias=False)

    def forward(self, inputs, label=None):
        logits = torch.nn.functional.linear(torch.nn.functional.normalize(inputs.flatten(1), dim=1),
                                            torch.nn.functional.normalize(self.fc.weight, dim=1))
        return {"logits": logits, "sfeatures": inputs.flatten(1)}


class PilotTests(unittest.TestCase):
    def test_strong_solarize_before_train_tensor_conversion_and_normalization(self):
        # Match the real layout: ToTensor is already in _train_trsf.
        train = [transforms.RandomResizedCrop(8), transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        common = [transforms.Normalize([.5] * 3, [.5] * 3)]
        ordinary = transforms.Compose([transforms.Compose(train), transforms.Compose(common)])
        strong = build_strong_transform(ordinary)
        self.assertLess(next(i for i, t in enumerate(strong.transforms) if isinstance(t, transforms.RandAugment)),
                        next(i for i, t in enumerate(strong.transforms) if isinstance(t, transforms.ToTensor)))
        image = Image.fromarray(np.random.RandomState(12).randint(0, 255, (16, 16, 3), dtype=np.uint8))

        def force_solarize(operation, value):
            self.assertIsInstance(value, Image.Image)
            return transforms.functional.solarize(value, 128)

        with patch.object(transforms.RandAugment, "forward", force_solarize):
            tensor = strong(image)
        self.assertEqual(tuple(tensor.shape), (3, 8, 8))
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertEqual(len(train), 3)  # Original real-anchor pipeline unchanged.
        self.assertEqual(len(common), 1)

    def test_strong_with_actual_IA10_and_CAR10_transform_layouts(self):
        from type_a_generative_augmentation.common import data_manager
        image = Image.fromarray(np.random.RandomState(13).randint(0, 255, (256, 256, 3), dtype=np.uint8))
        for protocol in ("IA10", "CAR10"):
            manager = data_manager(protocol)
            base = SimpleNamespace(trsf=transforms.Compose(manager._train_trsf + manager._common_trsf), labels=[])
            view = MatchedViews(base, [], manager, "S", 1, False)
            with patch.object(transforms.RandAugment, "forward",
                              lambda operation, value: transforms.functional.solarize(value, 128)):
                self.assertEqual(tuple(view.strong(image).shape), (3, 224, 224))
            for seed in range(24):
                with isolated_cpu_rng(seed):
                    tensor = view.strong(image)
                self.assertEqual(tuple(tensor.shape), (3, 224, 224))
                self.assertTrue(torch.isfinite(tensor).all())

    def test_completed_branch_is_skipped_without_constructing_a_model(self):
        from type_a_generative_augmentation import training
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            write_json(target / "task_checkpoints/task_01.pt", {"synthetic_completion_marker": True})
            write_csv(target / "predictions_T01.csv", [{"synthetic_completion_marker": True}])
            with patch.object(training, "run_dir", return_value=target), \
                 patch.object(training, "AugmentationLearner", side_effect=AssertionError("completed branch must not retrain")):
                training.train_one("IA10", "T1", "R", [], torch.device("cpu"))

    def test_imports_from_real_launch_directory_in_fresh_process(self):
        # Workspace-level tests used to miss data.py shadowing the namespace package.
        pilot_root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; from pathlib import Path; "
            "sys.path.insert(0, str(Path.cwd().parent)); "
            "from type_a_generative_augmentation.common import bootstrap; bootstrap(); "
            "from data.data import iCARS196_224, iImageNetA; "
            "from data.data_manager import DummyDataset; "
            "from selective.protocol import SweepDataManager; "
            "from type_a_generative_augmentation.learner import AugmentationLearner; "
            "from type_a_generative_augmentation.matched_views import MatchedViews; "
            "import data.data_manager as module; "
            "assert Path(module.__file__).resolve().parent == "
            "Path('${RESEARCH_WORKSPACE}/structure_multicenter_separation/data'); "
            "print('launch-directory imports PASS')"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=pilot_root,
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                   capture_output=True, text=True, timeout=45)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("launch-directory imports PASS", completed.stdout)

    def test_pooled_covariance_includes_source_shift_and_single_generated(self):
        rng = np.random.RandomState(3)
        real = rng.randn(10, 5)
        for m in (1, 7):
            generated = rng.randn(m, 5) + 4
            mean, covariance = mixed_moments(real, generated)
            combined = np.concatenate((real, generated))
            np.testing.assert_allclose(mean, combined.mean(0), atol=1e-12)
            np.testing.assert_allclose(covariance, np.cov(combined, rowvar=False), atol=1e-12)

    def test_matched_training_anchor_schedule_budget_and_real_only_SCC(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, labels, parents = [], [], []
            rng = np.random.RandomState(4)
            for i in range(4):
                path, generated = root / f"real{i}.png", root / f"gen{i}.png"
                Image.fromarray(rng.randint(0, 255, (12, 12, 3), dtype=np.uint8)).save(path)
                Image.fromarray(rng.randint(0, 255, (12, 12, 3), dtype=np.uint8)).save(generated)
                paths.append(str(path))
                labels.append(2 + i % 2)
                parents.append({"y": 2 + i % 2, "parent_path": str(path), "generated_path": str(generated), "record_id": str(i)})
            manager = SimpleNamespace(_train_trsf=[transforms.RandomResizedCrop(8), transforms.RandomHorizontalFlip()],
                                      _common_trsf=[transforms.ToTensor()])
            base = DummyDataset(np.array(paths), np.array(labels), transforms.Compose(manager._train_trsf + manager._common_trsf), True)
            fingerprints = []
            for branch in ("R", "S", "G"):
                target = root / branch
                dataset = MatchedViews(base, parents, manager, branch, 9, True)
                loader = DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(5))
                learner = AugmentationLearner.__new__(AugmentationLearner)
                torch.manual_seed(7)
                learner._network = ToyNet(4)
                learner._device = torch.device("cpu")
                learner._known_classes, learner._total_classes, learner._cur_task = 2, 4, 8
                learner._class_means = np.zeros((2, 3 * 8 * 8))
                learner.tuned_epochs, learner.batch_size = 2, 2
                learner.run_dir, learner.branch = target, branch
                learner.args = {"scale": 3., "margin": .2, "scc_value": 5., "protocol_id": "TOY", "optimizer": "sgd"}
                optimizer = torch.optim.SGD(learner._network.parameters(), lr=.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
                with patch("type_a_generative_augmentation.learner.LobachevskyRelativeGeomLoss",
                           side_effect=lambda out, means, device: out["logits"].sum() * 0) as scc:
                    learner._init_train(loader, None, optimizer, scheduler)
                    self.assertEqual(scc.call_count, 4)  # Real anchor only, not 8 real+aux forwards.
                trace = read_json(target / "matched_training_T09.json")
                self.assertEqual(sum(e["optimizer_steps"] for e in trace["epochs"]), 4)
                self.assertEqual(sum(e["forward_backward_images"] for e in trace["epochs"]), 16)
                fingerprints.append((trace["anchor_sha256"], trace["parent_schedule_sha256"]))
            self.assertTrue(all(item == fingerprints[0] for item in fingerprints))

    def test_G_GM_shared_CA_random_draws_and_G_state_restoration(self):
        from oracle_audit.snapshot_learner import AuditSnapshotLearner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = AugmentationLearner.__new__(AugmentationLearner)
            learner._network = ToyNet(4)
            learner._cur_task, learner._known_classes = 8, 2
            learner._class_means = np.zeros((4, 2))
            learner._class_covs = torch.eye(2).repeat(4, 1, 1)
            learner.branch, learner.run_dir = "G", root / "G"
            learner.args = {"experiment_id": "G", "protocol_id": "TOY"}
            learner.save_experiment_checkpoint = lambda *args, **kwargs: None
            mixed = learner._class_means.copy()
            mixed[2:] += 1
            learner.mixed_statistics = lambda: (mixed, learner._class_covs.clone(), [])
            noises = []

            def fake_ca(instance, task_size, epochs):
                noises.append(torch.randn(4, 16, 2))
                torch.randperm(64)
                with torch.no_grad():
                    instance._network.fc.weight.add_(float(instance._class_means.sum()))
                instance._audit_pre_ca_sidecar = instance.run_dir / "sidecar.pt"

            original_head = copy.deepcopy(learner._network.fc.state_dict())
            with patch.object(AuditSnapshotLearner, "_stage2_compact_classifier", fake_ca), \
                 patch("type_a_generative_augmentation.learner.run_dir", return_value=root / "GM"):
                learner._stage2_compact_classifier(2, 5)
            self.assertTrue(torch.equal(noises[0], noises[1]))
            self.assertEqual(learner.args["experiment_id"], "G")
            np.testing.assert_array_equal(learner._class_means, np.zeros((4, 2)))
            self.assertTrue(torch.equal(learner._network.fc.state_dict()["weight"], original_head["weight"]))
            self.assertFalse(torch.equal(learner._gm_state["head"]["weight"], original_head["weight"]))

    def test_quality_gate_never_accepts_pending_sample(self):
        from type_a_generative_augmentation import generation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, review = [], []
            image = root / "sample.png"
            Image.new("RGB", (8, 8), "red").save(image)
            for protocol in ("IA10", "CAR10"):
                for task in (1, 9):
                    key = f"{protocol}{task}"
                    rows.append({"record_id": key, "protocol": protocol, "task": task, "y": task,
                                 "review_sample": 1, "generation_status": "generated",
                                 "parent_path": str(image), "generated_path": str(image),
                                 "parent_sha256": sha256(image), "generated_sha256": sha256(image)})
                    review.append({"record_id": key, "review_status": "pending",
                                   "label_fidelity_verified": "no", "variation": "pending"})
            write_csv(root / "generation_manifest.csv", rows)
            write_csv(root / "quality_review.csv", review)
            with patch.object(generation, "ROOT", root), patch.object(generation, "ensure_config", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "quality HOLD"):
                    generation.quality_gate()
                for row in review:
                    row.update(review_status="accept", label_fidelity_verified="yes", variation="meaningful")
                write_csv(root / "quality_review.csv", review)
                self.assertEqual(len(generation.quality_gate()), 4)
                Image.new("RGB", (8, 8), "blue").save(image)
                with self.assertRaisesRegex(RuntimeError, "image changed"):
                    generation.quality_gate()

    def test_retention_and_head_summary(self):
        birth = [{"sample_id": "a", "birth_task": 9, "y": 2, "pred": 2},
                 {"sample_id": "b", "birth_task": 9, "y": 3, "pred": 2}]
        final = [{"sample_id": "b", "birth_task": 9, "y": 3, "pred": 3},
                 {"sample_id": "a", "birth_task": 9, "y": 2, "pred": 4}]
        result = retention_transitions(birth, final)
        self.assertEqual(result["birth_correct_to_T10_wrong"], 1)
        self.assertEqual(result["birth_wrong_to_T10_correct"], 1)
        self.assertEqual(result["T10_minus_T9_accuracy_pp"], 0)
        rows = [{"y": 0, "pred": 2, "birth_task": 1, "local_pred": 0, "head_CE": 1, "margin_all": -1, "margin_within": 1},
                {"y": 2, "pred": 0, "birth_task": 9, "local_pred": 2, "head_CE": 3, "margin_all": -2, "margin_within": 1}]
        result = summary(rows, 2, 9)
        self.assertEqual(result["head_CE_mean"], 2)
        self.assertEqual(result["new_head_CE_mean"], 3)
        self.assertEqual(result["error_directions"]["new_to_old"], 1)

    def test_support_leave_one_out_excludes_self(self):
        from type_a_generative_augmentation.audit import top_support
        bank = np.eye(3)
        np.testing.assert_array_equal(top_support(bank, bank, 1), np.ones(3))
        np.testing.assert_array_equal(top_support(bank, bank, 1, exclude_diagonal=True), np.zeros(3))

    def test_report_end_to_end_stop_decision_serializes(self):
        from type_a_generative_augmentation import results
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def directory_for(protocol, location, branch):
                return root / "training" / protocol / location / branch

            def predictions(task, branch):
                return [{"sample_id": str(i), "y": i, "pred": i, "local_pred": i,
                         "birth_task": i + 1, "evaluated_after_task": task,
                         "fixed_type": "final_correct", "branch": branch,
                         "head_CE": 1., "margin_all": 1., "margin_within": 1.} for i in range(task)]

            trace = {"anchor_sha256": "same", "parent_schedule_sha256": "same", "optimizer": "sgd", "batch_size_anchor": 2,
                     "epochs": [{"anchor_samples": 10, "forward_backward_images": 20, "optimizer_steps": 5, "learning_rates_after_step": [.01]}]}
            for protocol in ("IA10", "CAR10"):
                for location, tasks in (("T1", (1,)), ("T9_T10", (9, 10))):
                    for branch in ("R", "S", "G", "GM"):
                        target = directory_for(protocol, location, branch)
                        write_json(target / "start_manifest.json", {"initial_network_sha256": "same"})
                        write_json(target / "resource_usage.json", {"synthetic_test": True})
                        for task in tasks:
                            write_json(target / f"matched_training_T{task:02d}.json", trace)
                            write_csv(target / f"predictions_T{task:02d}.csv", predictions(task, branch))
            write_json(root / "generation_resource_usage.json", {"synthetic_test": True})
            with patch.object(results, "ROOT", root), patch.object(results, "run_dir", directory_for), \
                 patch.object(results, "source_config", return_value={"task_partition": [1] * 10}), \
                 patch.object(results, "verify_stats_fork", return_value={"synthetic_test": True}), \
                 patch.object(results, "baseline_rows", side_effect=lambda protocol, task: predictions(task, "V")):
                results.build_report()
                write_json(root / "exploratory_config.json", {"mode": "exploratory_automatic_sanity", "label_fidelity_verified": False})
                write_json(root / "automatic_sanity_summary.json", {"resource_usage": {"synthetic_test": True}})
                with patch("type_a_generative_augmentation.automatic_sanity.load_accepted", return_value=[]):
                    results.build_report()
            metrics = read_json(root / "metrics.json")
            self.assertEqual(metrics["IA10"]["decision"]["status"], "STOP_CURRENT_RECIPE")
            self.assertFalse(metrics["CAR10"]["decision"]["checks"]["better_than_R_birth_mean_and_T10_seen"])
            self.assertTrue((root / "final_report.md").is_file())
            self.assertFalse(metrics["experiment_scope"]["mode"]["label_fidelity_verified"])
            self.assertIn("cannot guarantee CAR model/year identity", (root / "final_report.md").read_text())

    def test_fixed_A_paired_harm_is_not_assumed_zero(self):
        s = [{"y": 0, "pred": 0}, {"y": 1, "pred": 0}]
        g = [{"y": 0, "pred": 1}, {"y": 1, "pred": 1}]
        row = paired(s, g, np.ones(2, bool))
        self.assertEqual((row["repair"], row["harm"], row["delta_pp"]), (1, 1, 0.))


if __name__ == "__main__":
    unittest.main()
