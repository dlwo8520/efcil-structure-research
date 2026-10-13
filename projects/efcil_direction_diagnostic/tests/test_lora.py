from __future__ import annotations

import unittest

import torch
from torch import nn

from efcil_diag.lora import QVFixedBasisLoRA


class LoRAMathTest(unittest.TestCase):
    def test_forward_preservation_orthogonality_and_merge(self) -> None:
        torch.manual_seed(4)
        dim = 8
        base = nn.Linear(dim, 3 * dim, bias=True)
        wrapper = QVFixedBasisLoRA(base, layer_id=0, rank=3, alpha=3, targets=("q", "v"))
        inputs = torch.randn(2, 5, dim)
        before = wrapper(inputs).detach()
        protected, _ = torch.linalg.qr(torch.randn(dim, 2), mode="reduced")
        gradients = {target: torch.randn(dim, dim) for target in wrapper.targets}
        for target in wrapper.targets:
            stats = wrapper.initialize_branch(
                target,
                gradients[target],
                protected,
                orthogonal=True,
                tolerance=1.0e-7,
            )
            self.assertLess(stats.aq_norm, 1.0e-5)
        after_initialization = wrapper(inputs).detach()
        self.assertTrue(torch.allclose(before, after_initialization, atol=2.0e-5, rtol=2.0e-5))

        with torch.no_grad():
            for target, A, B in wrapper.branch_items():
                delta_B = torch.randn_like(B) * 0.01
                effective = wrapper.effective_delta_from_B_delta(target, delta_B)
                self.assertLess(float(torch.linalg.vector_norm(effective @ protected)), 1.0e-5)
                B.add_(delta_B)
        active_output = wrapper(inputs).detach()
        wrapper.merge_and_reset()
        merged_output = wrapper(inputs).detach()
        self.assertTrue(torch.allclose(active_output, merged_output, atol=2.0e-5, rtol=2.0e-5))


if __name__ == "__main__":
    unittest.main()
