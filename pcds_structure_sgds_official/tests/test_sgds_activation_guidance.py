import unittest

import torch

from models.sgds_activation_guidance import SGDSActivationGuidance


class SGDSActivationGuidanceTest(unittest.TestCase):
    def make_guidance(self):
        return SGDSActivationGuidance(
            feature_dim=10,
            num_classes=4,
            num_epochs=4,
            target_layer=11,
            paper_k=0.6,
            beta=0.5,
            gamma=1.0,
            exploration_ratio=0.5,
        )

    def test_task_zero_allocation_and_eq6_self(self):
        guidance = self.make_guidance()
        rows = guidance.prepare_task(0, 0, torch.eye(2, 10))
        self.assertTrue(all(row["strategy"] == "NEW_SUBSPACE_ALLOCATION" for row in rows))
        self.assertTrue(all(row["eq6_includes_self"] == 1 for row in rows))
        self.assertAlmostEqual(sum(row["s_new"] for row in rows), 2.0, places=5)

    def test_final_mask_updates_only_survivors_and_persists(self):
        torch.manual_seed(3)
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(2, 10))
        guidance.train()
        labels = torch.tensor([0, 1])
        guidance.set_batch_context(labels, epoch=0)
        output = guidance(torch.ones(2, 3, 10))
        guidance.clear_batch_context()
        active = output.ne(0).any(dim=1).sum(dim=1)
        self.assertTrue(bool((active <= 6).all()))
        self.assertTrue(bool((guidance.global_counter == guidance.class_counter.sum(dim=0)).all()))
        before = guidance.global_counter.clone()
        guidance.prepare_task(1, 2, torch.eye(2, 10).roll(2, dims=1))
        self.assertTrue(torch.equal(before, guidance.global_counter))

    def test_eval_is_deterministic_identity(self):
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(2, 10))
        inputs = torch.randn(2, 5, 10)
        guidance.eval()
        guidance.set_batch_context(torch.tensor([0, 1]), epoch=0)
        first = guidance(inputs)
        second = guidance(inputs)
        self.assertTrue(torch.equal(first, inputs))
        self.assertTrue(torch.equal(second, inputs))

    def test_wrong_class_index_is_rejected(self):
        guidance = self.make_guidance()
        guidance.prepare_task(0, 0, torch.eye(2, 10))
        with self.assertRaises(RuntimeError):
            guidance.set_batch_context(torch.tensor([0, 2]), epoch=0)


if __name__ == "__main__":
    unittest.main()
