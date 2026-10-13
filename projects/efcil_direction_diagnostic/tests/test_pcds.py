from __future__ import annotations

import unittest

import torch
from torch import nn

from efcil_diag.pcds import (
    FeatureStatisticsAccumulator,
    PrototypeMemory,
    adaptive_prototype_rectification,
    stable_covariance_factor,
    structural_consistency_loss,
)
from efcil_diag.pcds_experiment import METHOD_FLAGS
from efcil_diag.pcds_model import AttentionWithPCDSAdapters, MLPWithPCDSAdapter
from efcil_diag.model import ExpandingCosineClassifier
from timm.models.vision_transformer import Attention


class PCDSMathTest(unittest.TestCase):
    def test_apr_matches_equations_four_to_eight(self) -> None:
        old_means = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        old_covariances = torch.stack((torch.eye(2), 2.0 * torch.eye(2)))
        new_before = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        new_after = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        result = adaptive_prototype_rectification(
            old_means,
            old_covariances,
            new_before,
            new_after,
            gamma=2.0,
            rho=3.0,
            epsilon=1.0e-4,
        )
        similarities = old_means @ new_before.T
        weights = torch.softmax(2.0 * similarities, dim=1)
        displacements = new_after - new_before
        corrections = weights @ displacements
        expected_covariances = (
            old_covariances
            + 3.0 * torch.einsum("ki,kj->kij", corrections, corrections)
            + 1.0e-4 * torch.eye(2).unsqueeze(0)
        )
        self.assertTrue(torch.allclose(result.attention_weights, weights))
        self.assertTrue(torch.allclose(result.means, old_means + corrections))
        self.assertTrue(torch.allclose(result.covariances, expected_covariances))

    def test_scc_has_no_reference_or_anchor_gradient(self) -> None:
        current = torch.randn(4, 6, requires_grad=True)
        reference = torch.randn(4, 6, requires_grad=True)
        anchors = torch.randn(3, 6, requires_grad=True)
        loss = structural_consistency_loss(current, reference, anchors)
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNone(reference.grad)
        self.assertIsNone(anchors.grad)

    def test_zero_initialized_adapter_preserves_backbone_forward(self) -> None:
        torch.manual_seed(7)
        base = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
        wrapper = MLPWithPCDSAdapter(base, 8, bottleneck=3, dropout=0.0)
        inputs = torch.randn(2, 5, 8)
        expected = base(inputs)
        actual = wrapper(inputs)
        self.assertTrue(torch.allclose(expected, actual))
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in wrapper.adapter.parameters()))

    def test_statistics_accumulator_rejects_old_class_data(self) -> None:
        accumulator = FeatureStatisticsAccumulator(
            begin_class=10,
            num_classes=2,
            feature_dim=3,
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(RuntimeError, "possible data leakage"):
            accumulator.update(torch.randn(2, 3), torch.tensor([9, 10]))

    def test_prototype_memory_round_trip_has_only_compressed_statistics(self) -> None:
        memory = PrototypeMemory(feature_dim=2)
        memory.append(
            torch.tensor([[1.0, 2.0]]),
            torch.eye(2).unsqueeze(0),
            torch.tensor([5]),
        )
        state = memory.export()
        self.assertEqual(set(state), {"means", "covariances", "counts"})
        restored = PrototypeMemory(feature_dim=2)
        restored.load(state)
        self.assertTrue(torch.equal(restored.means, memory.means))
        self.assertTrue(torch.equal(restored.covariances, memory.covariances))

    def test_covariance_sampling_factor_is_stable(self) -> None:
        covariance = torch.tensor([[1.0, 1.2], [1.2, 1.0]])
        factor, jitter, method = stable_covariance_factor(
            covariance, 1.0e-3, max_attempts=4
        )
        self.assertEqual(method, "eigh_fallback")
        self.assertGreater(jitter, 0.0)
        reconstructed = factor @ factor.T
        self.assertTrue(torch.linalg.eigvalsh(reconstructed).min() > 0.0)

    def test_requested_ablation_switches_are_explicit(self) -> None:
        self.assertEqual(
            METHOD_FLAGS,
            {
                "baseline": {"apr": False, "scc": False, "ca": False},
                "apr_only": {"apr": True, "scc": False, "ca": False},
                "scc_only": {"apr": False, "scc": True, "ca": False},
                "apr_scc": {"apr": True, "scc": True, "ca": False},
                "pcds_full": {"apr": True, "scc": True, "ca": True},
                "pcds_structure_sgds": {"apr": True, "scc": True, "ca": True},
            },
        )

    def test_classifier_rows_follow_module_device_when_added_later(self) -> None:
        classifier = ExpandingCosineClassifier(feature_dim=4, scale=20.0).to(
            torch.device("meta")
        )
        classifier.add_classes(torch.randn(2, 4, device="cpu"))
        self.assertEqual(classifier.weight_blocks[0].device.type, "meta")

    def test_zero_initialized_attention_adapters_preserve_forward(self) -> None:
        torch.manual_seed(11)
        base = Attention(dim=8, num_heads=2, qkv_bias=True)
        wrapper = AttentionWithPCDSAdapters(
            base, feature_dim=8, bottleneck=3, dropout=0.0
        )
        wrapper.eval()
        inputs = torch.randn(2, 5, 8)
        expected = base(inputs)
        actual = wrapper(inputs)
        self.assertTrue(torch.allclose(expected, actual))
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))
        self.assertTrue(
            all(parameter.requires_grad for parameter in wrapper.key_adapter.parameters())
        )


if __name__ == "__main__":
    unittest.main()
