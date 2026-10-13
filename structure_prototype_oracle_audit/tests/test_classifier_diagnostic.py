import unittest

import numpy as np

from classifier_diagnostic.pipeline import _calibration, _rank_auc, _score_components


class ClassifierDiagnosticTests(unittest.TestCase):
    def test_group_margin_and_components(self):
        scores = np.asarray([[3.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 4.0]])
        labels = np.asarray([0, 3])
        values = _score_components(scores, labels, known_before=2)
        self.assertTrue(np.allclose(values["true"], [3.0, 4.0]))
        self.assertTrue(np.allclose(values["group_margin"], [-1.0, 3.0]))
        self.assertTrue(np.allclose(values["opposite_margin"], [1.0, 3.0]))

    def test_rank_auc(self):
        values = np.asarray([-2.0, -1.0, 1.0, 2.0])
        positive = np.asarray([False, False, True, True])
        self.assertEqual(_rank_auc(values, positive), 1.0)

    def test_calibration_is_finite(self):
        scores = np.asarray([[3.0, 1.0], [0.0, 2.0]])
        labels = np.asarray([0, 1])
        result = _calibration(scores, labels)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertLess(result["nll"], 0.2)


if __name__ == "__main__":
    unittest.main()
