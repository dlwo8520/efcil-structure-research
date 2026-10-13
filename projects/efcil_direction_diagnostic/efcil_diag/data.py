from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .utils import make_generator, seed_worker


CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


class CIFAR100ImageFolder(datasets.ImageFolder):
    """Read a class-directory CIFAR-100 copy using canonical label IDs."""

    def __init__(self, root: str | Path, transform) -> None:
        super().__init__(str(root), transform=transform)
        expected = set(CIFAR100_CLASSES)
        if set(self.classes) != expected:
            raise RuntimeError(
                f"Expected the 100 canonical CIFAR-100 class directories under {root}"
            )
        canonical_id = {name: index for index, name in enumerate(CIFAR100_CLASSES)}
        folder_to_canonical = {
            self.class_to_idx[name]: canonical_id[name] for name in self.classes
        }
        self.targets = [folder_to_canonical[int(target)] for target in self.targets]
        self.samples = [
            (path, folder_to_canonical[int(target)]) for path, target in self.samples
        ]
        self.imgs = self.samples


class RemappedSubset(Dataset):
    """A view over CIFAR-100 that maps original labels to CIL label indices."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Iterable[int],
        original_to_incremental: dict[int, int],
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.original_to_incremental = original_to_incremental

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        image, original_label = self.base_dataset[self.indices[item]]
        return image, self.original_to_incremental[int(original_label)]


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    # The selected timm IN21K checkpoint uses mean/std=(0.5, 0.5, 0.5).
    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize(
            int(round(image_size / 0.9)), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def build_cds_cars_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Released CDS Cars-196 augmentation and evaluation preprocessing."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.05, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(
            int((256 / 224) * image_size),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return train_transform, test_transform


class CIFAR100IncrementalData:
    def __init__(
        self,
        root: str | Path,
        class_order: list[int],
        image_size: int,
    ) -> None:
        self.root = Path(root)
        self.class_order = class_order
        self.original_to_incremental = {
            original: incremental for incremental, original in enumerate(class_order)
        }
        train_transform, test_transform = build_transforms(image_size)
        if self._is_imagefolder_copy(self.root):
            self.source_format = "class_directories"
            self.train_augmented = CIFAR100ImageFolder(
                self.root / "train", transform=train_transform
            )
            self.train_deterministic = CIFAR100ImageFolder(
                self.root / "train", transform=test_transform
            )
            self.test = CIFAR100ImageFolder(
                self.root / "test", transform=test_transform
            )
        elif (self.root / "cifar-100-python").is_dir():
            self.source_format = "torchvision_pickle"
            self.train_augmented = datasets.CIFAR100(
                str(self.root), train=True, transform=train_transform, download=False
            )
            self.train_deterministic = datasets.CIFAR100(
                str(self.root), train=True, transform=test_transform, download=False
            )
            self.test = datasets.CIFAR100(
                str(self.root), train=False, transform=test_transform, download=False
            )
        else:
            raise FileNotFoundError(
                f"No local CIFAR-100 dataset found at {self.root}. "
                "Automatic download is intentionally disabled."
            )
        self.train_indices = self._indices_by_original_class(self.train_augmented.targets)
        self.test_indices = self._indices_by_original_class(self.test.targets)

    @staticmethod
    def _is_imagefolder_copy(root: Path) -> bool:
        return (root / "train").is_dir() and (root / "test").is_dir()

    @staticmethod
    def _indices_by_original_class(targets: list[int]) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            result[int(target)].append(index)
        return dict(result)

    @staticmethod
    def task_bounds(
        task_id: int, task_sizes: int | list[int] | tuple[int, ...]
    ) -> tuple[int, int]:
        """Return an incremental-label span for fixed or uneven task sizes."""
        if isinstance(task_sizes, int):
            begin = task_id * task_sizes
            return begin, begin + task_sizes
        sizes = [int(value) for value in task_sizes]
        if not 0 <= task_id < len(sizes):
            raise IndexError(f"task_id {task_id} outside {len(sizes)} tasks")
        begin = sum(sizes[:task_id])
        return begin, begin + sizes[task_id]

    def task_original_classes(
        self, task_id: int, task_sizes: int | list[int] | tuple[int, ...]
    ) -> list[int]:
        begin, end = self.task_bounds(task_id, task_sizes)
        return self.class_order[begin:end]

    def _make_subset(
        self,
        base: Dataset,
        original_classes: list[int],
        source_indices: dict[int, list[int]],
        per_class_limit: int | None = None,
        selection_seed: int = 0,
    ) -> RemappedSubset:
        rng = np.random.RandomState(selection_seed)
        indices: list[int] = []
        for original_class in original_classes:
            class_indices = np.asarray(source_indices[original_class], dtype=np.int64)
            if per_class_limit is not None and per_class_limit < len(class_indices):
                class_indices = rng.permutation(class_indices)[:per_class_limit]
            indices.extend(class_indices.tolist())
        return RemappedSubset(base, indices, self.original_to_incremental)

    def train_dataset(
        self,
        task_id: int,
        task_sizes: int | list[int] | tuple[int, ...],
        deterministic: bool = False,
        per_class_limit: int | None = None,
        selection_seed: int = 0,
    ) -> RemappedSubset:
        original_classes = self.task_original_classes(task_id, task_sizes)
        base = self.train_deterministic if deterministic else self.train_augmented
        return self._make_subset(
            base,
            original_classes,
            self.train_indices,
            per_class_limit=per_class_limit,
            selection_seed=selection_seed,
        )

    def seen_test_dataset(self, seen_classes: int) -> RemappedSubset:
        originals = self.class_order[:seen_classes]
        return self._make_subset(self.test, originals, self.test_indices)

    def task_test_dataset(
        self,
        task_id: int,
        task_sizes: int | list[int] | tuple[int, ...],
    ) -> RemappedSubset:
        originals = self.task_original_classes(task_id, task_sizes)
        return self._make_subset(self.test, originals, self.test_indices)


class Cars196IncrementalData(CIFAR100IncrementalData):
    """Official Stanford Cars train/test split exposed as ImageFolder trees.

    This follows the released CDS loader convention: both split directories
    contain the same 196 class directories, whose sorted ImageFolder indices
    are the original class IDs before the fixed CDS class-order permutation.
    """

    NUM_CLASSES = 196

    def __init__(
        self,
        root: str | Path,
        class_order: list[int],
        image_size: int,
    ) -> None:
        self.root = Path(root)
        if sorted(class_order) != list(range(self.NUM_CLASSES)):
            raise ValueError("Cars-196 class order must be a permutation of [0, 195]")
        self.class_order = class_order
        self.original_to_incremental = {
            original: incremental for incremental, original in enumerate(class_order)
        }
        train_root = self.root / "train"
        test_root = self.root / "test"
        if not train_root.is_dir() or not test_root.is_dir():
            raise FileNotFoundError(
                "Cars-196 official ImageFolder split is missing. Expected "
                f"{train_root}/<class>/... and {test_root}/<class>/..."
            )

        train_transform, test_transform = build_cds_cars_transforms(image_size)
        self.source_format = "official_train_test_imagefolder"
        self.train_augmented = datasets.ImageFolder(
            str(train_root), transform=train_transform
        )
        self.train_deterministic = datasets.ImageFolder(
            str(train_root), transform=test_transform
        )
        self.test = datasets.ImageFolder(str(test_root), transform=test_transform)
        if len(self.train_augmented.classes) != self.NUM_CLASSES:
            raise RuntimeError(
                f"Expected 196 Cars train classes, found {len(self.train_augmented.classes)}"
            )
        if self.train_augmented.classes != self.test.classes:
            raise RuntimeError("Cars-196 train/test class directories do not match")
        if not self.train_augmented.samples or not self.test.samples:
            raise RuntimeError("Cars-196 train/test split contains no images")
        self.class_names = list(self.train_augmented.classes)
        self.train_indices = self._indices_by_original_class(
            self.train_augmented.targets
        )
        self.test_indices = self._indices_by_original_class(self.test.targets)
        expected = set(range(self.NUM_CLASSES))
        if set(self.train_indices) != expected or set(self.test_indices) != expected:
            raise RuntimeError("Cars-196 official split is missing one or more classes")


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
