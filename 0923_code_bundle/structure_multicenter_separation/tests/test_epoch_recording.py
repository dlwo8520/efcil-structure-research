import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from models.CDS_adapter import Learner


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def forward(self, inputs, label=None):
        features = self.projection(inputs)
        return {"logits": torch.tanh(features), "sfeatures": features}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, features):
        return {"logits": self.linear(features)}


class TinyCANetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = TinyClassifier()

    def ca_forward(self, features):
        return self.fc(features)


class EpochRecordingTests(unittest.TestCase):
    def test_feature_epoch_records_sample_weighted_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = object.__new__(Learner)
            learner._network = TinyNetwork()
            learner._device = torch.device("cpu")
            learner._cur_task = 0
            learner._known_classes = 0
            learner.tuned_epochs = 1
            learner.run_dir = Path(directory)
            learner.args = {
                "experiment_id": "E0", "seed": 5, "scale": 2.0, "margin": 0.1,
                "scc": True, "scc_value": 5.0, "feature": False, "logit": False,
                "old_new_separation": False, "sep_lambda": 1.0,
                "sep_refresh_epochs": 1, "sep_grad_diagnostics": True,
                "checkpoint_every_epoch": False,
            }
            learner._compute_accuracy = lambda model, loader: 0.0
            batches = [
                (torch.tensor([0, 1]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1])),
                (torch.tensor([2]), torch.tensor([[1.0, 1.0]]), torch.tensor([0])),
            ]
            optimizer = optim.SGD(learner._network.parameters(), lr=0.1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            learner._init_train(batches, [], optimizer, scheduler)

            with (Path(directory) / "epoch_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stage"], "feature_training")
            self.assertEqual(int(rows[0]["processed_samples"]), 3)
            self.assertEqual(int(rows[0]["optimizer_steps"]), 2)
            self.assertEqual(rows[0]["scc_active"], "False")
            self.assertEqual(rows[0]["scc_raw"], "")
            self.assertEqual(rows[0]["checkpoint_path"], "")

    def test_classifier_retraining_has_a_separate_stage_row(self):
        with tempfile.TemporaryDirectory() as directory:
            learner = object.__new__(Learner)
            learner._network = TinyCANetwork()
            learner._device = torch.device("cpu")
            learner._multiple_gpus = [torch.device("cpu")]
            learner._cur_task = 1
            learner._total_classes = 2
            learner.task_sizes = [1, 1]
            learner.init_lr = 0.01
            learner.weight_decay = 0.0
            learner.logit_norm = None
            learner.run_dir = Path(directory)
            learner.test_loader = []
            learner._class_means = np.zeros((2, 2), dtype=np.float32)
            learner._class_covs = torch.stack([torch.eye(2), torch.eye(2)])
            learner.args = {
                "experiment_id": "E0", "seed": 5, "replay_samples_per_class": 4,
                "multi_center": False, "num_prototypes": 3, "scale": 1.0,
            }
            learner._compute_accuracy = lambda model, loader: 0.0
            learner._stage2_compact_classifier(task_size=1, ca_epochs=1)

            with (Path(directory) / "epoch_metrics.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["stage"], "classifier_retraining")
            self.assertEqual(int(rows[0]["processed_samples"]), 8)
            self.assertEqual(int(rows[0]["optimizer_steps"]), 2)
            self.assertEqual(rows[0]["scc_raw"], "")


if __name__ == "__main__":
    unittest.main()
