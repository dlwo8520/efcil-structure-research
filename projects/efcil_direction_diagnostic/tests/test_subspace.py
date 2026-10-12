from __future__ import annotations

import unittest

import torch

from efcil_diag.subspace import ProtectedSubspaceBank


class ProtectedSubspaceTest(unittest.TestCase):
    def test_update_is_orthonormal_and_respects_cap(self) -> None:
        torch.manual_seed(3)
        bank = ProtectedSubspaceBank(num_layers=2, feature_dim=12)
        bank.update({0: torch.randn(12, 8), 1: torch.randn(12, 7)}, rank_cap=5, tolerance=1.0e-7)
        bank.update({0: torch.randn(12, 8), 1: torch.randn(12, 7)}, rank_cap=5, tolerance=1.0e-7)
        for basis in bank.bases:
            self.assertEqual(basis.shape, (12, 5))
            identity = basis.transpose(0, 1) @ basis
            self.assertTrue(torch.allclose(identity, torch.eye(5), atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
