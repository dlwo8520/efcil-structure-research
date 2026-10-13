import unittest

import numpy as np

from classifier_procedure.pipeline import (
    _label_sequence_hash,
    _make_epoch_permutations,
    _replacement_balanced_bank,
    _sampling_rows,
    _split_train_validation,
)


class ClassifierProcedureTests(unittest.TestCase):
    def test_paired_label_sequence_is_deterministic(self):
        labels = np.repeat(np.arange(4), 3)
        first = _make_epoch_permutations(labels, 5, 1993)
        second = _make_epoch_permutations(labels, 5, 1993)
        self.assertEqual(_label_sequence_hash(labels, first), _label_sequence_hash(labels, second))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))

    def test_replacement_bank_is_balanced(self):
        labels = np.asarray([0, 0, 1, 1, 1, 2])
        features = np.arange(labels.size * 2, dtype=np.float32).reshape(labels.size, 2)
        bank, bank_labels, _ = _replacement_balanced_bank(
            features, labels, np.arange(3), 8, 7
        )
        self.assertEqual(bank.shape, (24, 2))
        self.assertTrue(np.array_equal(np.bincount(bank_labels), [8, 8, 8]))

    def test_validation_policy_keeps_rare_classes_in_train(self):
        labels = np.asarray([0, 0] + [1] * 5 + [2] * 10)
        features = np.arange(labels.size * 2, dtype=np.float32).reshape(labels.size, 2)
        train_x, train_y, val_x, val_y, metadata = _split_train_validation(
            features, labels, np.arange(3), 13
        )
        self.assertEqual(int((val_y == 0).sum()), 0)
        self.assertEqual(int((val_y == 1).sum()), 1)
        self.assertEqual(int((val_y == 2).sum()), 2)
        self.assertEqual(metadata["no_validation_classes"], [0])
        self.assertEqual(train_x.shape[0] + val_x.shape[0], features.shape[0])
        self.assertTrue(all((train_y == class_id).any() for class_id in range(3)))

    def test_g256_step_budget_matches_vanilla(self):
        audit, classes = _sampling_rows(
            "SYNTHETIC", total_classes=5, known_before=3,
            actual_counts=np.asarray([2, 3, 4, 5, 6]), epochs=5, batch_size=256,
        )
        rows = {row["condition"]: row for row in audit}
        self.assertEqual(rows["G256_FIXED"]["optimizer_steps"], 25)
        self.assertEqual(rows["R256_FIXED"]["optimizer_steps"], 25)
        self.assertEqual(rows["G_FIXED"]["optimizer_steps"], 15)
        r256 = [row for row in classes if row["condition"] == "R256_FIXED"]
        self.assertEqual({row["total_draws"] for row in r256}, {1280})


if __name__ == "__main__":
    unittest.main()

