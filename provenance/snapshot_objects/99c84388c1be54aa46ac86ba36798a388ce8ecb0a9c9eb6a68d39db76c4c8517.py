import unittest

import numpy as np
import torch

from utils.adjust_distribution import LobachevskyPrototypeCorrector
from utils.distribution_regions import (
    SeparationCache,
    build_regions_for_class,
    build_single_region,
    deterministic_kmeans,
    empirical_covariance,
    largest_remainder_counts,
    mahalanobis_squared,
    rectify_local_regions,
    sample_gaussian_mixture,
)


CONFIG = {
    "kmeans_n_init": 4,
    "kmeans_max_iter": 30,
    "cluster_min_samples": 2,
    "stats_cov_jitter": 1e-3,
    "sep_percentile": 95.0,
    "sep_cov_eigen_floor": 1e-6,
    "sep_cholesky_initial_jitter": 1e-6,
    "sep_cholesky_jitter_multiplier": 10.0,
    "sep_cholesky_max_tries": 4,
    "sep_region_chunk_size": 2,
    "sep_tau_floor": 1e-8,
    "sep_factor_device": "cpu",
}


class DistributionRegionTests(unittest.TestCase):
    def test_single_region_preserves_baseline_mean_and_covariance(self):
        features = torch.tensor([[0.0, 1.0], [2.0, 1.0], [1.0, 4.0]])
        mean = features.mean(dim=0)
        covariance, _, _ = empirical_covariance(features, CONFIG["stats_cov_jitter"])
        region = build_single_region(features, 2, 0, "baseline", mean, covariance, CONFIG)
        self.assertTrue(torch.equal(region["mean"], mean))
        self.assertTrue(torch.equal(region["covariance"], covariance))
        self.assertEqual(region["weight"], 1.0)

    def test_feature_space_mismatch_is_rejected(self):
        region = {
            "class_id": 0,
            "mean": torch.zeros(2),
            "covariance": torch.eye(2),
            "tau_by_covariance": {"isotropic": 1.0, "diagonal": 1.0, "full": 1.0},
            "feature_space": "source",
        }
        with self.assertRaisesRegex(ValueError, "feature-space"):
            SeparationCache([region], "full", "target", CONFIG)

    def test_mahalanobis_views_match_direct_calculation(self):
        x = torch.tensor([[0.7, -1.2], [1.1, 0.2]])
        mean = torch.tensor([0.1, -0.2])
        covariance = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        difference = x - mean
        expected = {
            "isotropic": difference.square().sum(1) / (torch.trace(covariance) / 2),
            "diagonal": (difference.square() / covariance.diagonal()).sum(1),
            "full": torch.einsum("bi,ij,bj->b", difference, torch.linalg.inv(covariance), difference),
        }
        for view in expected:
            actual, _ = mahalanobis_squared(x, mean, covariance, view, 1e-7, 1e-7, 10.0, 2)
            self.assertTrue(torch.allclose(actual, expected[view], atol=1e-5, rtol=1e-5))

    def test_hinge_inside_has_gradient_and_outside_is_zero(self):
        region = {
            "class_id": 0,
            "mean": torch.zeros(2),
            "covariance": torch.eye(2),
            "tau_by_covariance": {"isotropic": 4.0, "diagonal": 4.0, "full": 4.0},
            "feature_space": "test",
        }
        cache = SeparationCache([region], "full", "test", CONFIG)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        inside = torch.stack((parameter, parameter * 0.0)).unsqueeze(0)
        loss, diagnostics = cache.loss(inside)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.75, places=5)
        self.assertTrue(torch.isfinite(parameter.grad))
        self.assertNotEqual(parameter.grad.item(), 0.0)
        self.assertTrue(diagnostics.intrusion.item())
        outside_loss, outside_diagnostics = cache.loss(torch.tensor([[3.0, 0.0]], requires_grad=True))
        self.assertEqual(outside_loss.item(), 0.0)
        self.assertFalse(outside_diagnostics.intrusion.item())
        self.assertFalse(cache.means.requires_grad)
        self.assertFalse(cache.factors.requires_grad)

    def test_kmeans_is_deterministic_and_reduces_k(self):
        features = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]])
        first = deterministic_kmeans(features, 3, 7, 3, 30, 2)
        second = deterministic_kmeans(features, 3, 7, 3, 30, 2)
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertEqual(first[2]["actual_k"], 2)
        self.assertTrue(first[2]["reduced_k"])
        self.assertEqual(len(first[2]["assignments"]), len(features))
        self.assertTrue(first[2]["fallback_applied"])

    def test_kmeans_singleton_fallback_keeps_the_sample(self):
        centers, assignments, metadata = deterministic_kmeans(
            torch.tensor([[2.0, -1.0]]), 3, 7, 2, 10, 2
        )
        self.assertEqual(centers.shape, (1, 2))
        self.assertEqual(assignments.tolist(), [0])
        self.assertIn("singleton_or_tiny_class_kept_as_one_cluster", metadata["fallback_reasons"])

    def test_replay_allocation_and_labels_preserve_total(self):
        self.assertEqual(largest_remainder_counts([0.5, 0.3, 0.2], 7), [4, 2, 1])
        features = torch.tensor(
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
        )
        regions, _ = build_regions_for_class(features, 4, 0, "test", 2, 11, CONFIG)
        replay = sample_gaussian_mixture(regions, 17, torch.device("cpu"), 1.0, CONFIG)
        self.assertEqual(replay.shape, (17, 2))
        self.assertAlmostEqual(sum(region["weight"] for region in regions), 1.0, places=7)

    def test_local_apr_uses_same_class_transform_without_accumulation(self):
        corrector = LobachevskyPrototypeCorrector(torch.device("cpu"), gamma_weight=2.0, cov=3.0)
        old_means = np.array([[0.0, 0.0]], dtype=np.float32)
        new_before = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        new_after = new_before + np.array([[0.2, 0.1], [-0.1, 0.3]], dtype=np.float32)
        new_cov = torch.stack((torch.eye(2), torch.eye(2)))
        shifts, additions = corrector.class_level_corrections(
            old_means, new_before, new_cov, new_after, new_cov
        )
        source = {
            0: [
                {
                    "class_id": 0,
                    "component_id": 0,
                    "mean": torch.tensor([0.0, 0.0]),
                    "covariance": torch.eye(2),
                    "count": 2,
                    "weight": 1.0,
                    "tau_by_covariance": {"isotropic": 1.0, "diagonal": 1.0, "full": 1.0},
                    "birth_task": 0,
                    "feature_space": "source",
                    "rank": 1,
                    "regularization": {},
                    "fallback": "none",
                    "tau_details": {},
                }
            ]
        }
        first = rectify_local_regions(source, shifts, additions, "target", 1e-6)
        second = rectify_local_regions(source, shifts, additions, "target", 1e-6)
        self.assertTrue(torch.equal(first[0][0]["mean"], second[0][0]["mean"]))
        self.assertTrue(torch.equal(first[0][0]["covariance"], second[0][0]["covariance"]))
        self.assertTrue(torch.equal(first[0][0]["mean"], source[0][0]["mean"] + shifts[0]))


if __name__ == "__main__":
    unittest.main()
