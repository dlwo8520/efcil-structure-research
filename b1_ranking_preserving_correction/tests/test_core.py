"""CPU-only checks; no checkpoint, GPU, or full experiment is loaded."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import analyze
from core import Correction, descriptors, fit, invariant, normalize_fit


class CoreTests(unittest.TestCase):
    def test_descriptor_and_zero_identity(self):
        scores = torch.tensor([[1., 3., 2., 4.], [4., 2., 0., 1.]], dtype=torch.float64)
        descriptor = descriptors(scores, [2, 2])
        self.assertEqual(tuple(descriptor.shape), (2, 6))
        self.assertTrue(torch.equal(descriptor[0, :2], torch.tensor([3., 4.], dtype=torch.float64)))
        model = Correction(2, conditional=True)
        birth = torch.tensor([0, 0, 1, 1])
        self.assertTrue(torch.equal(model(scores, descriptor, birth), scores))

    def test_correction_exact_invariant_and_fitting(self):
        torch.manual_seed(3)
        scores = torch.randn(80, 6, dtype=torch.float64)
        labels = torch.randint(0, 6, (80,))
        birth = torch.tensor([0, 0, 1, 1, 2, 2])
        phi, _, _ = normalize_fit(descriptors(scores, [2, 2, 2]))
        for conditional in (False, True):
            model, status = fit(scores, phi, labels, birth, conditional)
            adjusted = model(scores, phi, birth)
            result = invariant(scores, adjusted, [2, 2, 2])
            self.assertEqual(result["local_argmax_disagreements"], 0)
            self.assertLess(result["within_task_pair_margin_max_abs"], 1e-8)
            self.assertLessEqual(status["final_objective"], status["initial_objective"] + 1e-9)

    def test_paired_tables_and_no_same_task_harm(self):
        # Labels 0/1 share task 1; 2/3 share task 2.
        scores = np.asarray([[3, 1, 0, 0], [2, 3, 4, 0], [0, 0, 3, 1], [0, 2, 1, 3]], float)
        ids = np.asarray([f"id{i}" for i in range(4)])
        labels = np.asarray([0, 1, 2, 3])
        changed = scores.copy()
        changed[1, :2] += 2  # fixed B1 repair
        trajectory = []
        for i in range(4):
            trajectory.append({"sample_id": ids[i], "true_label": str(labels[i]),
                               "error_type": "B_birth_correct_final_wrong" if i == 1 else "final_correct",
                               "birth_restriction_restored": "1" if i == 1 else "0"})
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            metrics = analyze("TEST", ids, labels, [2, 2],
                              {"C0": scores, "C1": changed, "C2": changed},
                              trajectory, Path(directory), 1)
            self.assertEqual(metrics["branches"]["C2"]["subsets"]["B1"]["repair"], 1)
            self.assertEqual(metrics["branches"]["C2"]["subsets"]["overall"]["harm"], 0)
            self.assertTrue((Path(directory) / "transition_tables.csv").exists())


if __name__ == "__main__":
    unittest.main()
