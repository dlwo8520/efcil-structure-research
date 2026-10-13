from __future__ import annotations

import unittest

import torch
from torch import nn

from efcil_diag.keeplora_lora import KeepLoRAOutputProjection, KeepLoRAQKV


class KeepLoRAMathTest(unittest.TestCase):
    def test_paper_orientation_forward_preservation_and_merge(self) -> None:
        torch.manual_seed(13)
        dim = 8
        qkv = KeepLoRAQKV(
            nn.Linear(dim, 3 * dim, bias=True),
            layer_id=0,
            rank=3,
            alpha=1.0,
            targets=("q", "k", "v"),
        )
        output = KeepLoRAOutputProjection(
            nn.Linear(dim, dim, bias=True), layer_id=0, rank=3, alpha=1.0
        )
        x = torch.randn(2, 5, dim)
        qkv_before = qkv(x).detach()
        output_before = output(x).detach()
        protected, _ = torch.linalg.qr(torch.randn(dim, 2), mode="reduced")

        for module in (qkv, output):
            for target in module.targets:
                stats = module.initialize_branch(
                    target,
                    torch.randn(dim, dim),
                    protected,
                    tolerance=1.0e-7,
                )
                self.assertLess(stats.aq_norm, 2.0e-5)
                self.assertGreater(stats.initial_delta_norm, 0.0)

        self.assertTrue(torch.allclose(qkv_before, qkv(x), atol=3.0e-5, rtol=3.0e-5))
        self.assertTrue(
            torch.allclose(output_before, output(x), atol=3.0e-5, rtol=3.0e-5)
        )

        with torch.no_grad():
            for module in (qkv, output):
                for target, _, B in module.branch_items():
                    delta_B = torch.randn_like(B) * 0.01
                    delta_weight = module.effective_delta_from_B_delta(target, delta_B)
                    self.assertLess(
                        float(torch.linalg.vector_norm(delta_weight @ protected)), 2.0e-5
                    )
                    B.add_(delta_B)
        qkv_active = qkv(x).detach()
        output_active = output(x).detach()
        qkv.merge_and_reset()
        output.merge_and_reset()
        self.assertTrue(torch.allclose(qkv_active, qkv(x), atol=3.0e-5, rtol=3.0e-5))
        self.assertTrue(
            torch.allclose(output_active, output(x), atol=3.0e-5, rtol=3.0e-5)
        )


if __name__ == "__main__":
    unittest.main()

