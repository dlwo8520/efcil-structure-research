from __future__ import annotations

import sys
import unittest
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.sgds_activation_guidance import (  # noqa: E402
    KNOWLEDGE_REUSE,
    NEW_SUBSPACE_ALLOCATION,
    SGDSActivationGuidance,
)


class SGDSActivationGuidanceTest(unittest.TestCase):
    def make_guidance(self) -> SGDSActivationGuidance:
        return SGDSActivationGuidance(
            feature_dim=4,
            num_classes=4,
            num_epochs=2,
            target_layer=11,
            paper_k=0.5,
            beta=0.5,
            gamma=1.0,
            exploration_ratio=0.5,
        )

    def test_eq4_to_eq6_include_current_class_itself(self) -> None:
        guidance = self.make_guidance()
        prototypes = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        rows = guidance.prepare_task(0, 0, prototypes)
        self.assertTrue(all(row["eq6_includes_self"] == 1 for row in rows))
        self.assertTrue(all(row["strategy"] == NEW_SUBSPACE_ALLOCATION for row in rows))
        self.assertAlmostEqual(rows[0]["s_old"] + rows[0]["s_new"], 1.0, places=6)

    def test_eq7_reuses_semantically_weighted_old_counter(self) -> None:
        guidance = self.make_guidance()
        guidance.prepare_task(
            0,
            0,
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]),
        )
        guidance.class_counter[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
        guidance.class_counter[1] = torch.tensor([0.0, 10.0, 0.0, 0.0])
        guidance.global_counter.copy_(guidance.class_counter[:2].sum(dim=0))
        rows = guidance.prepare_task(
            1,
            2,
            torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]),
        )
        self.assertEqual(rows[0]["strategy"], KNOWLEDGE_REUSE)
        guidance._epoch = 0
        probabilities = guidance._probabilities(torch.tensor([2]))[0]
        self.assertGreater(float(probabilities[0]), float(probabilities[2]))

    def test_eq8_topk_and_final_counter_update(self) -> None:
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(4)[:2])
        guidance.train()
        labels = torch.tensor([0, 1])
        activations = torch.ones(2, 3, 4)
        guidance.set_batch_context(labels, epoch=0)
        output = guidance(activations)
        guidance.clear_batch_context()
        self.assertEqual(output.ne(0).any(dim=1).sum(dim=1).tolist(), [2, 2])
        self.assertEqual(int(guidance.global_counter.sum()), 4)
        self.assertEqual(int(guidance.class_counter[0].sum()), 2)
        self.assertEqual(int(guidance.class_counter[1].sum()), 2)

    def test_eval_has_no_random_mask_or_counter_update(self) -> None:
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(4)[:2])
        guidance.eval()
        activation = torch.randn(2, 3, 4)
        before = guidance.global_counter.clone()
        self.assertTrue(torch.equal(guidance(activation), activation))
        self.assertTrue(torch.equal(before, guidance.global_counter))

    def test_counters_persist_at_task_boundary(self) -> None:
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(4)[:2])
        guidance.global_counter.add_(3)
        expected = guidance.global_counter.clone()
        guidance.prepare_task(1, 2, torch.eye(4)[2:])
        self.assertTrue(torch.equal(expected, guidance.global_counter))

    def test_cars_diagnostics_and_task_specific_epoch_boundary(self) -> None:
        guidance = SGDSActivationGuidance(
            feature_dim=4,
            num_classes=5,
            num_epochs=70,
            target_layer=11,
            paper_k=0.5,
            exploration_ratio=0.5,
        )
        rows = guidance.prepare_task(
            0, 0, torch.eye(4)[:2], num_epochs=70
        )
        self.assertTrue(math.isnan(rows[0]["max_old_cosine_similarity"]))
        self.assertEqual(guidance.exploration_epochs, 35)
        guidance.prepare_task(
            1,
            2,
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
            ),
            num_epochs=40,
        )
        self.assertEqual(guidance.exploration_epochs, 20)
        summary = guidance.semantic_task_summary()
        self.assertEqual(summary["old_classes"], 2)
        self.assertEqual(summary["new_classes"], 3)
        self.assertAlmostEqual(summary["mean_s_old"] + summary["mean_s_new"], 1.0)


if __name__ == "__main__":
    unittest.main()
