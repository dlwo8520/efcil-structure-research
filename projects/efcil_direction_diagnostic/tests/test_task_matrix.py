from __future__ import annotations

import unittest

from efcil_diag.task_matrix import (
    long_format_rows,
    parse_lower_triangular_matrix,
    summarize_cil_matrix,
)


class TaskMatrixTest(unittest.TestCase):
    def test_parse_summarize_and_long_format(self) -> None:
        rows = [
            {"eval_task": "0", "task_0": "0.9", "task_1": "", "task_2": ""},
            {"eval_task": "1", "task_0": "0.7", "task_1": "0.8", "task_2": ""},
            {"eval_task": "2", "task_0": "0.5", "task_1": "0.6", "task_2": "0.7"},
        ]
        matrix = parse_lower_triangular_matrix(rows, 3)
        summary = summarize_cil_matrix(matrix)
        self.assertIsNone(summary.transfer)
        self.assertEqual(summary.row_averages, [0.9, 0.75, 0.6])
        self.assertAlmostEqual(summary.average, 0.75)
        self.assertAlmostEqual(summary.last, 0.6)
        long_rows = long_format_rows("method", 0, matrix)
        self.assertEqual(len(long_rows), 6)
        self.assertEqual(long_rows[-1]["accuracy"], 70.0)
        self.assertAlmostEqual(long_rows[-1]["seen_average"], 60.0)

    def test_rejects_future_task_accuracy(self) -> None:
        rows = [
            {"eval_task": "0", "task_0": "0.9", "task_1": "0.1"},
            {"eval_task": "1", "task_0": "0.7", "task_1": "0.8"},
        ]
        with self.assertRaisesRegex(ValueError, "Future-task"):
            parse_lower_triangular_matrix(rows, 2)


if __name__ == "__main__":
    unittest.main()
