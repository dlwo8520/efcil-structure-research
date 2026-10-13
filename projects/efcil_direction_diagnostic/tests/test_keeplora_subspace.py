from __future__ import annotations

import unittest

import torch

from efcil_diag.keeplora_subspace import KeepLoRASubspaceBank


class _Site:
    layer_id = 0
    targets = ("q",)
    capture_group = "block_00.qkv_input"

    @staticmethod
    def site_key(target: str) -> str:
        return f"block_00.{target}"


class _Model:
    def __init__(self) -> None:
        self.site = _Site()

    def projection_sites(self):
        yield self.site, "q"


class KeepLoRASubspaceTest(unittest.TestCase):
    def test_residual_feature_update_stays_orthogonal_to_wp(self) -> None:
        bank = KeepLoRASubspaceBank(["block_00.q"], feature_dim=5)
        bank.wp_bases["block_00.q"] = torch.eye(5)[:, :1]
        features = torch.diag(torch.tensor([10.0, 4.0, 2.0, 1.0, 0.5]))
        stats, _ = bank.update_task_features(
            _Model(),
            {"block_00.qkv_input": features},
            use_wp=True,
            use_m=True,
            epsilon_f=0.99,
            tolerance=1.0e-7,
        )
        wp = bank.wp_bases["block_00.q"]
        task = bank.m_bases["block_00.q"]
        self.assertGreater(task.shape[1], 0)
        self.assertLess(float(torch.linalg.vector_norm(wp.T @ task)), 1.0e-5)
        unified = torch.cat((wp, task), dim=1)
        self.assertTrue(
            torch.allclose(unified.T @ unified, torch.eye(unified.shape[1]), atol=1.0e-5)
        )
        self.assertEqual(stats[0].unified_rank, unified.shape[1])
        self.assertGreaterEqual(stats[0].retained_energy_ratio, 0.99)


if __name__ == "__main__":
    unittest.main()
