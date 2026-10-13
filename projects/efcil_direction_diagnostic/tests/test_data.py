from __future__ import annotations

import unittest
from collections import Counter

from efcil_diag.data import CIFAR100_CLASSES, CIFAR100IncrementalData
from efcil_diag.utils import GOD_SEED_1993_ORDER


class SharedCIFAR100Test(unittest.TestCase):
    def test_canonical_classes_are_complete(self) -> None:
        self.assertEqual(len(CIFAR100_CLASSES), 100)
        self.assertEqual(len(set(CIFAR100_CLASSES)), 100)
        self.assertEqual(CIFAR100_CLASSES[0], "apple")
        self.assertEqual(CIFAR100_CLASSES[-1], "worm")

    def test_shared_imagefolder_uses_canonical_labels(self) -> None:
        data = CIFAR100IncrementalData(
            "/data/datasets/cifar100",
            class_order=GOD_SEED_1993_ORDER,
            image_size=224,
        )
        self.assertEqual(data.source_format, "class_directories")
        self.assertEqual(Counter(data.train_augmented.targets), Counter({i: 500 for i in range(100)}))
        self.assertEqual(Counter(data.test.targets), Counter({i: 100 for i in range(100)}))


if __name__ == "__main__":
    unittest.main()
