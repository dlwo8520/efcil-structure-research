import inspect
import unittest

import numpy as np
import torch

from oracle_audit.diagnostics import margin_rows
from oracle_audit.oracle_head import evaluate_scores, train_diagnostic_head


class OracleHeadTests(unittest.TestCase):
    def _train(self):
        generator = np.random.RandomState(7)
        weights = torch.as_tensor(generator.normal(size=(4, 5)), dtype=torch.float32)
        means = generator.normal(size=(4, 5)).astype(np.float64)
        covariances = np.stack([np.eye(5, dtype=np.float64) * 0.2 for _ in range(4)])
        class_ids = np.arange(4, dtype=np.int64)
        originals = (weights.clone(), means.copy(), covariances.copy())
        torch_state = torch.get_rng_state().clone()
        result = train_diagnostic_head(
            weights,
            means,
            covariances,
            class_ids,
            task_index=1,
            task_size=2,
            epochs=2,
            samples_per_class=3,
            init_lr=0.01,
            weight_decay=5e-4,
            scale=20.0,
            device=torch.device("cpu"),
            seed=123,
        )
        self.assertTrue(torch.equal(weights, originals[0]))
        self.assertTrue(np.array_equal(means, originals[1]))
        self.assertTrue(np.array_equal(covariances, originals[2]))
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertTrue(result.input_checksum_unchanged)
        self.assertEqual(result.optimizer_parameter_count, 1)
        self.assertFalse(np.allclose(result.weights, weights.numpy()))
        return result.weights

    def test_identical_statistics_and_seed_are_deterministic(self):
        first = self._train()
        second = self._train()
        self.assertTrue(np.array_equal(first, second))

    def test_old_new_decomposition(self):
        scores = np.asarray(
            [
                [5.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 5.0, 0.0],
                [5.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 5.0],
            ]
        )
        labels = np.arange(4, dtype=np.int64)
        class_ids = np.arange(4, dtype=np.int64)
        metrics = evaluate_scores(scores, labels, class_ids, known_before=2)
        self.assertEqual(metrics["seen_class_accuracy"], 50.0)
        self.assertEqual(metrics["old_accuracy"], 50.0)
        self.assertEqual(metrics["new_accuracy"], 50.0)
        self.assertEqual(metrics["old_only_accuracy"], 100.0)
        self.assertEqual(metrics["new_only_accuracy"], 100.0)
        self.assertEqual(metrics["old_to_new_count"], 1)
        self.assertEqual(metrics["new_to_old_count"], 1)
        rows = margin_rows({}, "TEST", scores, labels, class_ids, known_before=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["outcome"] for row in rows}, {"new_to_old", "new_to_correct_new"})

    def test_training_api_cannot_receive_backbone_or_raw_samples(self):
        parameters = set(inspect.signature(train_diagnostic_head).parameters)
        self.assertNotIn("backbone", parameters)
        self.assertNotIn("adapter", parameters)
        self.assertNotIn("data_loader", parameters)
        self.assertNotIn("raw_samples", parameters)


if __name__ == "__main__":
    unittest.main()
