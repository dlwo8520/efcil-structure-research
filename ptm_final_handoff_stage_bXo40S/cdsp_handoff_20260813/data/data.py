"""Dataset definitions with portable roots and strict split validation."""

import logging
import os
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


def build_transform(is_train, args=None):
    input_size = 224
    if is_train:
        return [
            transforms.RandomResizedCrop(input_size, scale=(0.05, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    return [
        transforms.Resize(int((256 / 224) * input_size), interpolation=3),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ]


class iCIFAR224(iData):
    use_path = False
    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []
    class_order = np.arange(100).tolist()

    def __init__(self, root=None, expected_num_classes=100):
        self.root = str(root or "./data")
        self.expected_num_classes = int(expected_num_classes)

    def download_data(self):
        train_dataset = datasets.CIFAR100(self.root, train=True, download=True)
        test_dataset = datasets.CIFAR100(self.root, train=False, download=True)
        self.train_data = train_dataset.data
        self.train_targets = np.asarray(train_dataset.targets)
        self.test_data = test_dataset.data
        self.test_targets = np.asarray(test_dataset.targets)


class ImageFolderIncrementalData(iData):
    """Load a preprocessed CIL dataset laid out as ``root/{train,test}/class``."""

    use_path = True
    train_trsf = build_transform(True)
    test_trsf = build_transform(False)
    common_trsf = []
    dataset_name = "dataset"
    env_var = None
    default_root = None
    default_num_classes = None

    def __init__(self, root=None, expected_num_classes=None):
        env_root = os.environ.get(self.env_var, "") if self.env_var else ""
        selected = root or env_root or self.default_root
        if not selected:
            raise ValueError("No dataset root configured for {}".format(self.dataset_name))
        self.root = Path(os.path.expandvars(os.path.expanduser(str(selected)))).resolve()
        self.expected_num_classes = int(
            expected_num_classes if expected_num_classes is not None else self.default_num_classes
        )

    def _split_directories(self):
        train_dir = self.root / "train"
        test_dir = self.root / "test"
        if train_dir.is_dir() and test_dir.is_dir():
            return train_dir, test_dir

        nested = [
            path for path in self.root.iterdir()
            if path.is_dir() and (path / "train").is_dir() and (path / "test").is_dir()
        ] if self.root.is_dir() else []
        if len(nested) == 1:
            logging.info("Resolved nested %s root: %s", self.dataset_name, nested[0])
            self.root = nested[0]
            return self.root / "train", self.root / "test"

        env_hint = " Set {}=/path/to/root".format(self.env_var) if self.env_var else ""
        raise FileNotFoundError(
            "{} requires ImageFolder splits at {}/train and {}/test.{}".format(
                self.dataset_name, self.root, self.root, env_hint
            )
        )

    def download_data(self):
        train_dir, test_dir = self._split_directories()
        train_dset = datasets.ImageFolder(str(train_dir))
        test_dset = datasets.ImageFolder(str(test_dir))
        train_classes = train_dset.class_to_idx
        test_classes = test_dset.class_to_idx
        if train_classes != test_classes:
            only_train = sorted(set(train_classes) - set(test_classes))
            only_test = sorted(set(test_classes) - set(train_classes))
            raise ValueError(
                "{} train/test class mappings differ (train-only={}, test-only={})".format(
                    self.dataset_name, only_train[:10], only_test[:10]
                )
            )
        actual_classes = len(train_classes)
        if actual_classes != self.expected_num_classes:
            raise ValueError(
                "{} expected {} classes but found {} under {}".format(
                    self.dataset_name, self.expected_num_classes, actual_classes, self.root
                )
            )
        if not train_dset.imgs or not test_dset.imgs:
            raise ValueError("{} contains an empty train or test split".format(self.dataset_name))

        self.class_order = np.arange(actual_classes).tolist()
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
        logging.info(
            "Loaded %s from %s: classes=%d train=%d test=%d",
            self.dataset_name,
            self.root,
            actual_classes,
            len(self.train_targets),
            len(self.test_targets),
        )


class iImageNetR(ImageFolderIncrementalData):
    dataset_name = "ImageNet-R"
    env_var = "IMAGENETR_ROOT"
    default_root = "./data/imagenet-r"
    default_num_classes = 200


class iImageNetA(ImageFolderIncrementalData):
    dataset_name = "ImageNet-A"
    env_var = "IMAGENETA_ROOT"
    default_root = "./data/imagenet-a"
    default_num_classes = 200


class CUB(ImageFolderIncrementalData):
    dataset_name = "CUB-200"
    env_var = "CUB_ROOT"
    default_root = "./data/cub"
    default_num_classes = 200


class iCARS196_224(ImageFolderIncrementalData):
    dataset_name = "Cars-196"
    env_var = "CARS196_ROOT"
    default_root = "./data/cars196"
    default_num_classes = 196


class vtab(ImageFolderIncrementalData):
    dataset_name = "VTAB"
    env_var = "VTAB_ROOT"
    default_root = "./data/vtab"
    default_num_classes = 50


class objectnet(ImageFolderIncrementalData):
    dataset_name = "ObjectNet-CIL-200"
    env_var = "OBJECTNET_ROOT"
    default_root = "./data/objectnet"
    default_num_classes = 200


class omnibenchmark(ImageFolderIncrementalData):
    dataset_name = "OmniBenchmark-300"
    env_var = "OMNIBENCHMARK_ROOT"
    default_root = "./data/omnibenchmark"
    default_num_classes = 300
