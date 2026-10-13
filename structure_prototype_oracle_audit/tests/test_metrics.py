import unittest

import numpy as np

from oracle_audit.metrics import (
    angle_degrees,
    covariance_metrics,
    mahalanobis_squared_diagonal,
    pearson,
    safe_cosine,
    spearman,
    vector_error,
)


class MetricTests(unittest.TestCase):
    def test_vector_metrics_identical(self):
        value = np.array([1.0, 2.0, 3.0])
        metrics = vector_error(value, value)
        self.assertTrue(np.isclose(metrics["cosine"], 1.0))
        self.assertTrue(np.isclose(metrics["l2"], 0.0))
        self.assertTrue(np.isclose(metrics["angle_error_deg"], 0.0))

    def test_rank_correlations_with_ties(self):
        left = np.array([1.0, 1.0, 2.0, 3.0])
        right = np.array([2.0, 2.0, 4.0, 6.0])
        self.assertTrue(np.isclose(pearson(left, right), 1.0))
        self.assertTrue(np.isclose(spearman(left, right), 1.0))

    def test_covariance_and_mahalanobis(self):
        covariance = np.diag([1.0, 4.0])
        metrics = covariance_metrics(covariance, covariance)
        self.assertTrue(np.isclose(metrics["diag_relative_l2"], 0.0))
        self.assertTrue(np.isclose(metrics["trace_ratio"], 1.0))
        self.assertTrue(np.isclose(metrics["structure_effective_rank_participation"], 25.0 / 17.0))
        self.assertTrue(np.isclose(metrics["oracle_effective_rank_participation"], 25.0 / 17.0))
        distances = mahalanobis_squared_diagonal(
            np.array([[1.0, 2.0]]), np.zeros(2), covariance
        )
        self.assertTrue(np.allclose(distances, [2.0]))


if __name__ == "__main__":
    unittest.main()
