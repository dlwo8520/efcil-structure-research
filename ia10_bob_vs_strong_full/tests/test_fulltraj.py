"""CPU contract checks; no diffusion model or real ViT is trained/inferred."""
from __future__ import annotations

import ast
import copy
import hashlib
import random
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

from fulltraj.common import ROOT, VENDOR, bootstrap, read_json, write_csv, write_json
bootstrap()
from data.data_manager import DummyDataset
from network.classifier import SimpleContinualLinear
from fulltraj.bob_worker import generation_plan, image_qc
from fulltraj.views import FullViews, RealOnlyManager
from fulltraj.learner import FullTrajectoryLearner, assert_matched_logs, state_hash
from fulltraj.results import paired, task_metric


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Linear(3, 3, bias=False)
        self.convnet.flag = 0
        self.fc = None

    def update_fc(self, size):
        if self.fc is None:
            self.fc = SimpleContinualLinear(3, size)
        else:
            self.fc.update(size, freeze_old=False)

    def forward(self, images, label=None):
        raw = images.mean((2, 3))
        features = self.convnet(raw)
        return {**self.fc(features), "sfeatures": features, "tfeatures": raw.detach()}


def tiny_learner(target, branch, classes=2, known=0):
    learner = object.__new__(FullTrajectoryLearner)
    learner._network = TinyNet()
    learner._network.update_fc(classes)
    learner.branch = branch
    learner._device = torch.device("cpu")
    learner._cur_task = int(known > 0)
    learner._known_classes, learner._total_classes = known, classes
    learner.task_sizes = [classes] if not known else [known, classes - known]
    learner._class_means = np.ones((known, 3), dtype=np.float32)
    learner._class_covs = None
    learner.tuned_epochs = 1
    learner.batch_size = 2
    learner.run_dir = Path(target)
    learner.current_bank, learner._active_views = [], None
    learner.args = {"scale": 20., "margin": .2, "scc_value": 5., "experiment_id": branch,
                    "fulltraj_branch": branch, "fulltraj_experiment_hash": "CPU_test_only",
                    "dataset": "IA10", "seed": 1993, "init_cls": 20, "increment": 20,
                    "expected_tasks": 10, "multi_center": False, "num_prototypes": 1,
                    "old_new_separation": False, "separation_covariance": "diagonal", "sep_lambda": 0.,
                    "sep_percentile": 95., "sep_timing": "epoch_apr_cache", "sep_boundary_policy": "birth_quantile_fixed"}
    for name in ("_local_stats", "_single_regions", "_task_source_local_stats"):
        setattr(learner, name, {})
    for name in ("_task_source_means", "_task_source_covs", "_sep_reference_means", "_sep_reference_covs"):
        setattr(learner, name, None)
    return learner


class FullTrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="bob_cpu_contract_")
        self.root = Path(self.tmp.name)
        self.paths = []
        rng = np.random.RandomState(17)
        for i in range(4):
            path = self.root / f"real{i}.png"
            Image.fromarray(rng.randint(0, 255, (80, 80, 3), dtype=np.uint8)).save(path)
            self.paths.append(str(path))
        self.transform = transforms.Compose([transforms.RandomResizedCrop(8), transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(), transforms.Normalize([.5] * 3, [.5] * 3)])
        self.base = DummyDataset(np.asarray(self.paths), np.asarray([0, 0, 1, 1]), self.transform, use_path=True)
        self.bank = [{"task": 1, "class_id": i, "accepted": "1", "generated_path": self.paths[2*i],
                      "record_id": f"bank{i}"} for i in range(2)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_anchor_views_and_global_rng_equal(self):
        s, b = FullViews(self.base, "S_full", 1), FullViews(self.base, "BOB_full", 1, self.bank)
        before = torch.get_rng_state().clone()
        for epoch in range(2):
            s.epoch = b.epoch = epoch
            for index in range(4):
                sa, ba = s[index], b[index]
                self.assertTrue(torch.equal(sa[1], ba[1]))
                self.assertEqual(int(sa[3]), int(ba[3]))
                self.assertEqual(sa[1].shape, sa[2].shape)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_solarize_gets_PIL_not_float_tensor(self):
        s = FullViews(self.base, "S_full", 1)
        operations = s.strong.transforms
        self.assertLess(next(i for i, t in enumerate(operations) if isinstance(t, transforms.RandAugment)),
                        next(i for i, t in enumerate(operations) if isinstance(t, transforms.ToTensor)))
        def forced(operation, image):
            self.assertIsInstance(image, Image.Image)
            return transforms.functional.solarize(image, 128)
        with patch.object(transforms.RandAugment, "forward", forced):
            self.assertTrue(torch.isfinite(s[0][2]).all())

    def test_old_future_and_rejected_banks_fail(self):
        for bad in ({"task": 2}, {"accepted": "0"}, {"class_id": 20}):
            changed = copy.deepcopy(self.bank)
            changed[0].update(bad)
            with self.assertRaises(AssertionError):
                FullViews(self.base, "BOB_full", 1, changed)

    def test_release_drops_bank_access(self):
        views = FullViews(self.base, "BOB_full", 1, self.bank)
        views.release()
        self.assertFalse(views.bank)
        with self.assertRaises(KeyError):
            views[0]

    def test_real_statistics_guard(self):
        source = SimpleNamespace(_train_data=self.paths, get_dataset=lambda *a, **k: self.base)
        guarded = RealOnlyManager(source)
        self.assertIs(guarded.get_dataset([0, 1], "train", "test"), self.base)
        with self.assertRaises(AssertionError):
            guarded.get_dataset([0], "train", "test", appendent=([], []))
        bad = DummyDataset(np.asarray(["unexpected_generated.png"]), np.asarray([0]), self.transform, True)
        source.get_dataset = lambda *a, **k: bad
        with self.assertRaises(AssertionError):
            guarded.get_dataset([0], "train", "test")

    def test_technical_qc_and_duplicates(self):
        policy = {"min_side": 8, "blank_std_max": 1., "reject_exact_pixel_duplicates": True}
        hashes = set()
        self.assertTrue(image_qc(self.paths[0], hashes, policy=policy)[0])
        self.assertEqual(image_qc(self.paths[0], hashes, policy=policy)[1], "exact_duplicate")
        blank = self.root / "blank.png"
        Image.new("RGB", (80, 80), "white").save(blank)
        self.assertEqual(image_qc(blank, hashes, policy=policy)[1], "blank")
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        self.assertEqual(image_qc(corrupt, hashes, policy=policy)[1], "decode_failure")
        self.assertEqual(image_qc(self.paths[1], hashes, safety=True, policy=policy)[1], "generation_safety_failure")

    def test_context_pairing_and_target_independence(self):
        records = [{"task": 1, "class_id": i, "class_name": f"class{i}", "source_id": f"real{i}",
                    "source_path": self.paths[i]} for i in range(2)]
        captions = {r["source_id"]: {"background": f"background{i}", "pose": f"pose{i}"} for i, r in enumerate(records)}
        plan = generation_plan(1, records, captions, count=12)
        self.assertEqual(len(plan), 24)
        self.assertEqual(plan, generation_plan(1, records, captions, count=12))
        for row in plan:
            i = row["context_source_id"][-1]
            self.assertEqual(row["background"], f"background{i}")
            self.assertEqual(row["pose"], f"pose{i}")
        self.assertTrue(any(r["context_source_id"] != f"real{r['class_id']}" for r in plan))

    def test_official_adapter_export_accepts_isolated_smoke_folder(self):
        from fulltraj.inputs import export_official_inputs
        folder = self.root / "feasibility"
        records = [{"task": 1, "class_id": 0, "class_name": "object_class", "source_id": f"train:{i}",
                    "source_sha256": f"sha{i}", "bob_input_path": path} for i, path in enumerate(self.paths[:2])]
        caps = [{"task": 1, "source_id": r["source_id"], "source_sha256": r["source_sha256"],
                 "background": f"bg{i}", "pose": f"pose{i}"} for i, r in enumerate(records)]
        write_csv(folder / "real_train.csv", records)
        write_csv(folder / "captions.csv", caps)
        config = export_official_inputs(1, folder)
        self.assertEqual(config["classnames"], ["object_class"])
        self.assertEqual(config["data_path"], str(folder / "real"))
        for field in ("background", "pose"):
            import csv
            with (folder / "captions" / f"{field}.tsv").open(newline="") as stream:
                rows = list(csv.reader(stream, delimiter="\t"))
            self.assertEqual(rows[0], ["path", "caption"])
            self.assertEqual([r[0] for r in rows[1:]], self.paths[:2])
            self.assertEqual([r[1] for r in rows[1:]], [c[field] for c in caps])

    def test_gpu_smoke_orchestration_with_cpu_toy_substitute(self):
        # Exercise the actual orchestration/save-load paths without allocating
        # CUDA or invoking the pretrained ViT. Actual GPU feasibility is still
        # deferred to the user's launch and is not claimed by this test.
        from fulltraj.training import gpu_smoke
        write_csv(self.root / "feasibility/bob/smoke_bank.csv", self.bank[:1])
        base = DummyDataset(np.asarray(self.paths[:2]), np.asarray([0, 0]), self.transform, True)
        fake_manager = SimpleNamespace(get_dataset=lambda *a, **k: base)
        def construct(config, bank):
            folder = Path(config["run_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            learner = tiny_learner(folder, config["fulltraj_branch"])
            learner._network.fc = None
            learner.current_bank = list(bank)
            return learner
        with patch("fulltraj.training.ROOT", self.root), patch("fulltraj.training.manager", return_value=fake_manager), \
             patch("fulltraj.training.experiment_hash", return_value="CPU_test_only"), \
             patch("fulltraj.training.FullTrajectoryLearner", side_effect=construct), patch("fulltraj.learner.progress"):
            gpu_smoke(torch.device("cpu"))
        marker = read_json(self.root / "feasibility/structure/complete.json")
        self.assertEqual(marker["optimizer_updates_each_branch"], 1)
        self.assertFalse(marker["test_evaluation"])
        for branch in ("S_full", "BOB_full"):
            self.assertTrue((self.root / "feasibility/structure" / branch / "checkpoint_latest.pt").is_file())

    def test_two_steps_and_original_checkpoint_save_load(self):
        logs = []
        for branch in ("S_full", "BOB_full"):
            torch.manual_seed(1993)
            target = self.root / branch
            target.mkdir()
            learner = tiny_learner(target, branch)
            views = FullViews(self.base, branch, 1, self.bank)
            loader = DataLoader(views, batch_size=2, shuffle=True, generator=torch.Generator())
            optimizer = torch.optim.SGD(learner._network.parameters(), lr=.01, momentum=.9)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            before = state_hash(learner._network.state_dict())
            # Keep test progress outside the experiment output.
            with patch("fulltraj.learner.progress"):
                learner._init_train(loader, None, optimizer, scheduler)
            self.assertNotEqual(before, state_hash(learner._network.state_dict()))
            cp = target / "checkpoint_latest.pt"
            loaded = torch.load(cp, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["stage"], "train_epoch")
            self.assertEqual(loaded["next_epoch"], 1)
            fresh = tiny_learner(self.root / (branch + "fresh"), branch)
            fresh._network.fc = None
            restored = fresh.load_experiment_checkpoint(cp)
            self.assertEqual(state_hash(fresh._network.state_dict()), state_hash(learner._network.state_dict()))
            self.assertEqual(fresh._network.convnet.flag, learner._network.convnet.flag)
            self.assertIsNotNone(restored["optimizer_state_dict"])
            logs.append(read_json(target / "matched_training_T01.json"))
        assert_matched_logs(*logs)
        self.assertEqual(logs[0]["epochs"][0]["optimizer_steps"], 2)
        self.assertEqual(logs[0]["epochs"][0]["aux_samples"], 4)

    def test_scc_only_real_anchor(self):
        target = self.root / "SCC"
        target.mkdir()
        learner = tiny_learner(target, "S_full", classes=4, known=2)
        base = DummyDataset(np.asarray(self.paths), np.asarray([2, 2, 3, 3]), self.transform, True)
        views = FullViews(base, "S_full", 2)
        loader = DataLoader(views, batch_size=2, generator=torch.Generator(), shuffle=True)
        optimizer = torch.optim.SGD(learner._network.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        from utils.distill_loss import LobachevskyRelativeGeomLoss
        with patch("fulltraj.learner.progress"), patch("fulltraj.learner.LobachevskyRelativeGeomLoss", wraps=LobachevskyRelativeGeomLoss) as scc:
            learner._init_train(loader, None, optimizer, scheduler)
        self.assertEqual(scc.call_count, 2)  # One per real batch, not two per view.

    def test_epoch_restart_equals_uninterrupted(self):
        def loader_for(branch):
            return DataLoader(FullViews(self.base, branch, 1, self.bank), batch_size=2,
                              generator=torch.Generator(), shuffle=True)
        torch.manual_seed(1993)
        full = tiny_learner(self.root / "uninterrupted", "BOB_full")
        full.run_dir.mkdir()
        full.tuned_epochs = 2
        initial = copy.deepcopy(full._network.state_dict())
        optimizer = torch.optim.SGD(full._network.parameters(), lr=.01, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
        with patch("fulltraj.learner.progress"):
            full._init_train(loader_for("BOB_full"), None, optimizer, scheduler)
        partial = tiny_learner(self.root / "resumed", "BOB_full")
        partial.run_dir.mkdir()
        partial._network.load_state_dict(initial)
        partial.tuned_epochs = 1
        first_optimizer = torch.optim.SGD(partial._network.parameters(), lr=.01, momentum=.9)
        first_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(first_optimizer, 2)
        with patch("fulltraj.learner.progress"):
            partial._init_train(loader_for("BOB_full"), None, first_optimizer, first_scheduler)
        checkpoint = partial.run_dir / "checkpoint_latest.pt"
        fresh = tiny_learner(partial.run_dir, "BOB_full")
        fresh._network.fc = None
        saved = fresh.load_experiment_checkpoint(checkpoint)
        fresh.tuned_epochs = 2
        second_optimizer = torch.optim.SGD(fresh._network.parameters(), lr=.01, momentum=.9)
        second_optimizer.load_state_dict(saved["optimizer_state_dict"])
        second_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(second_optimizer, 2)
        second_scheduler.load_state_dict(saved["scheduler_state_dict"])
        with patch("fulltraj.learner.progress"):
            fresh._init_train(loader_for("BOB_full"), None, second_optimizer, second_scheduler, start_epoch=1)
        self.assertEqual(state_hash(full._network.state_dict()), state_hash(fresh._network.state_dict()))
        assert_matched_logs(read_json(full.run_dir / "matched_training_T01.json"),
                            read_json(fresh.run_dir / "matched_training_T01.json"))

    def test_full_fixture_analysis_creates_five_plots(self):
        # Fabricated test fixtures never leave this temporary directory.
        from fulltraj import results
        report_root = self.root / "fixture_output"
        report_root.mkdir()
        def own_dir(branch):
            return report_root / "checkpoints" / branch
        def reference(task):
            return [{"branch": "Vanilla", "task": task, "sample_id": f"sample{b}", "y": (b-1)*20,
                     "birth_task": b, "pred": (b-1)*20, "correct": 1, "local_correct": ""} for b in range(1, task+1)]
        for branch in ("S_full", "BOB_full"):
            for task in range(1, 11):
                rows = [{**r, "branch": branch, "correct": int(branch == "BOB_full"),
                         "local_correct": int(branch == "BOB_full")} for r in reference(task)]
                write_csv(own_dir(branch) / f"predictions_T{task:02d}.csv", rows)
        with patch.object(results, "ROOT", report_root), patch.object(results, "train_dir", own_dir), \
             patch.object(results, "vanilla_rows", reference):
            self.assertTrue(results.analyze())
        self.assertEqual(len(list((report_root / "figures").glob("*.png"))), 5)
        self.assertIn("BOB_NEW_WIN_TASKS = 10 / 10", (report_root / "final_report.md").read_text())
        self.assertEqual(len(__import__("fulltraj.common", fromlist=["read_csv"]).read_csv(report_root / "repair_harm_by_task.csv")), 30)

    def test_paired_counts_macro_and_denominators(self):
        left = [{"sample_id": str(i), "y": y, "birth_task": b, "correct": c} for i, (y, b, c) in
                enumerate([(0, 1, 1), (0, 1, 0), (20, 2, 0)])]
        right = [{**r, "correct": c} for r, c in zip(left, [0, 1, 1])]
        value = paired(left, right)
        self.assertEqual((value["repair"], value["harm"], value["net"]), (2, 1, 1))
        self.assertAlmostEqual(value["delta_pp"], 100 / 3)
        metric = task_metric(right, "BOB_full", 2)
        self.assertEqual(metric["macro_accuracy_pct"], 75.)
        self.assertEqual(metric["old_accuracy_pct"], 50.)
        self.assertEqual(metric["new_accuracy_pct"], 100.)
        with self.assertRaises(AssertionError):
            paired(left, [{**r, "y": 30} for r in right])

    def test_official_arg_attribute_compatibility(self):
        # Static audit of the pinned official source; generator deps not needed.
        config_tree = ast.parse((VENDOR / "finetune/config.py").read_text())
        defined = set()
        for node in ast.walk(config_tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        defined.add(arg.value[2:].replace("-", "_"))
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == "args":
                defined.add(node.attr)
        official_tree = ast.parse((VENDOR / "finetune/finetune_t2i.py").read_text())
        used = {node.attr for node in ast.walk(official_tree) if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name) and node.value.id == "args"}
        self.assertEqual(used - defined, {"train_text_encoder", "max_train_steps"})
        # max_train_steps is computed inside official main; wrapper supplies missing flag.


if __name__ == "__main__":
    unittest.main()
