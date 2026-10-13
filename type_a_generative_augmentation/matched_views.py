"""Matched pilot views; distinct name avoids shadowing Structure's data package."""
from __future__ import annotations

import contextlib
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@contextlib.contextmanager
def isolated_cpu_rng(seed):
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        generator = torch.Generator().manual_seed(seed)
        torch.set_rng_state(generator.get_state())
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def seed_for(task, epoch, index, stream):
    return 1993 + task * 10_000_000 + epoch * 100_000 + index * 7 + stream


def build_strong_transform(ordinary):
    """Insert PIL RandAugment before tensor conversion in the actual base pipeline.

    Structure already puts ToTensor in _train_trsf (not _common_trsf).
    RandAugment's Solarize/Posterize must not operate on the converted float
    tensor. Preserve all base transforms, then append tensor RandomErasing.
    """
    def flatten(transform):
        if isinstance(transform, transforms.Compose):
            return [operation for child in transform.transforms for operation in flatten(child)]
        return [transform]

    operations = flatten(ordinary)
    tensor_position = next((i for i, operation in enumerate(operations)
                            if isinstance(operation, (transforms.ToTensor, transforms.PILToTensor))), None)
    if tensor_position is None:
        raise ValueError("strong augmentation requires an explicit ToTensor/PILToTensor conversion")
    operations.insert(tensor_position, transforms.RandAugment(num_ops=2, magnitude=9))
    operations.append(transforms.RandomErasing(p=0.25))
    return transforms.Compose(operations)


class MatchedViews(Dataset):
    def __init__(self, base, parents, manager, branch, task, augmented):
        self.base, self.branch, self.task, self.augmented = base, branch, task, augmented
        self.epoch = 0
        self.parents = {}
        for row in parents:
            self.parents.setdefault(int(row["y"]), []).append(row)
        self.ordinary = base.trsf
        self.strong = build_strong_transform(self.ordinary)
        if augmented and any(int(y) not in self.parents for y in np.unique(base.labels)):
            raise ValueError("every current class needs a common accepted parent bank")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        with isolated_cpu_rng(seed_for(self.task, self.epoch, index, 0)):
            _, anchor, label = self.base[index]
        if not self.augmented:
            return index, anchor, label
        options = self.parents[int(label)]
        chooser = np.random.RandomState(seed_for(self.task, self.epoch, index, 1))
        position = int(chooser.randint(len(options)))
        row = options[position]
        source = row["generated_path"] if self.branch == "G" else row["parent_path"]
        with Image.open(source) as image:
            image = image.convert("RGB")
        with isolated_cpu_rng(seed_for(self.task, self.epoch, index, 2)):
            aux = (self.strong if self.branch == "S" else self.ordinary)(image)
        return index, anchor, aux, label, row["record_id"]
