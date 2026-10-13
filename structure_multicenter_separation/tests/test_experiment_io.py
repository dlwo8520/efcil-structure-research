import csv
import tempfile
import unittest
from pathlib import Path

from utils.experiment_io import upsert_csv_rows


class ExperimentIOTests(unittest.TestCase):
    def test_upsert_csv_replaces_a_resume_key_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_metrics.csv"
            key = ("experiment_id", "seed", "task_id", "stage", "epoch")
            upsert_csv_rows(
                path,
                [{"experiment_id": "E2", "seed": 7, "task_id": 2, "stage": "feature_training", "epoch": 3, "total_loss": 4.0}],
                key,
            )
            upsert_csv_rows(
                path,
                [{"experiment_id": "E2", "seed": 7, "task_id": 2, "stage": "feature_training", "epoch": 3, "total_loss": 2.5}],
                key,
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["total_loss"]), 2.5)

    def test_upsert_csv_sorts_numeric_key_fields_naturally(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            upsert_csv_rows(path, [{"epoch": 10}, {"epoch": 2}, {"epoch": 1}], ("epoch",))
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["epoch"]) for row in rows], [1, 2, 10])


if __name__ == "__main__":
    unittest.main()
