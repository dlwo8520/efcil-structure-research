from __future__ import annotations

import unittest

from efcil_diag.metrics import EvaluationResult, summarize_task_metrics


class MetricTest(unittest.TestCase):
    def test_task_forgetting(self) -> None:
        evaluation = EvaluationResult(
            seen_accuracy=0.65,
            class_accuracy=[0.5, 0.5, 0.8, 0.8],
            class_correct=[5, 5, 8, 8],
            class_count=[10, 10, 10, 10],
            prototypes=None,  # not used by this calculation
        )
        row, task_accuracies = summarize_task_metrics(
            task_id=1,
            classes_per_task=2,
            evaluation=evaluation,
            task_accuracy_history=[[0.9]],
        )
        self.assertEqual(task_accuracies, [0.5, 0.8])
        self.assertAlmostEqual(row["old_accuracy"], 0.5)
        self.assertAlmostEqual(row["new_accuracy"], 0.8)
        self.assertAlmostEqual(row["average_forgetting"], 0.4)

    def test_triangular_history_after_three_tasks(self) -> None:
        evaluation = EvaluationResult(
            seen_accuracy=0.6,
            class_accuracy=[0.4, 0.4, 0.6, 0.6, 0.8, 0.8],
            class_correct=[4, 4, 6, 6, 8, 8],
            class_count=[10] * 6,
            prototypes=None,
        )
        row, task_accuracies = summarize_task_metrics(
            task_id=2,
            classes_per_task=2,
            evaluation=evaluation,
            task_accuracy_history=[[0.9], [0.5, 0.85]],
        )
        self.assertEqual(task_accuracies, [0.4, 0.6, 0.8])
        self.assertAlmostEqual(row["average_forgetting"], (0.5 + 0.25) / 2)

    def test_uneven_task_sizes_use_sample_weighted_accuracy(self) -> None:
        evaluation = EvaluationResult(
            seen_accuracy=13 / 30,
            class_accuracy=[0.5, 0.5, 0.3],
            class_correct=[5, 5, 3],
            class_count=[10, 10, 10],
            prototypes=None,
        )
        row, task_accuracies = summarize_task_metrics(
            task_id=1,
            classes_per_task=[2, 1],
            evaluation=evaluation,
            task_accuracy_history=[[0.8]],
        )
        self.assertEqual(task_accuracies, [0.5, 0.3])
        self.assertEqual(row["seen_classes"], 3)
        self.assertAlmostEqual(row["old_accuracy"], 0.5)
        self.assertAlmostEqual(row["new_accuracy"], 0.3)


if __name__ == "__main__":
    unittest.main()
