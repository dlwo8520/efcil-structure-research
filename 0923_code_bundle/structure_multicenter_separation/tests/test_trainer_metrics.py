import unittest

from trainer import _accuracy_matrix_from_long


class TrainerMetricTests(unittest.TestCase):
    def test_accuracy_matrix_is_rebuilt_from_long_records(self):
        rows = [
            {"experiment_id": "E0", "seed": "3", "trained_task": "1", "evaluated_task": "1", "eval_stage": "task_final", "accuracy": "80.25"},
            {"experiment_id": "E0", "seed": "3", "trained_task": "2", "evaluated_task": "1", "eval_stage": "task_final", "accuracy": "70.0"},
            {"experiment_id": "E0", "seed": "3", "trained_task": "2", "evaluated_task": "2", "eval_stage": "task_final", "accuracy": "60.5"},
        ]
        self.assertEqual(
            _accuracy_matrix_from_long(rows, "E0", 3, 2, "task_final"),
            [[80.25], [70.0, 60.5]],
        )

    def test_missing_long_record_does_not_get_zero_filled(self):
        rows = [
            {"experiment_id": "E0", "seed": "3", "trained_task": "1", "evaluated_task": "1", "eval_stage": "task_final", "accuracy": "80.25"},
        ]
        self.assertIsNone(_accuracy_matrix_from_long(rows, "E0", 3, 2, "task_final"))


if __name__ == "__main__":
    unittest.main()
