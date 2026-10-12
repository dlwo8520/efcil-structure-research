from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# Reuse the verified PIL-before-ToTensor RandAugment recipe and isolated RNG.
from type_a_generative_augmentation.matched_views import (
    build_strong_transform, isolated_cpu_rng, seed_for,
)


class FullViews(Dataset):
    def __init__(self, base, branch, task, bank=()):
        self.base, self.branch, self.task = base, branch, task
        self.epoch = 0
        self.ordinary = base.trsf
        self.strong = build_strong_transform(base.trsf)
        self.bank = {}
        if branch == "BOB_full":
            for row in bank:
                if int(row["task"]) != task or row["accepted"] not in (True, "1", "True"):
                    raise AssertionError("old/future/rejected synthetic input forbidden")
                self.bank.setdefault(int(row["class_id"]), []).append(row)
            assert set(self.bank) == set(map(int, np.unique(base.labels)))
        elif branch != "S_full":
            raise ValueError(branch)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        with isolated_cpu_rng(seed_for(self.task, self.epoch, index, 0)):
            _, anchor, label = self.base[index]
        if self.branch == "S_full":
            path = str(self.base.images[index])
            aux_id = f"real:{index}"
        else:
            options = self.bank[int(label)]
            chooser = np.random.RandomState(seed_for(self.task, self.epoch, index, 1))
            row = options[int(chooser.randint(len(options)))]
            assert int(row["task"]) == self.task
            path, aux_id = row["generated_path"], row["record_id"]
        with Image.open(path) as image:
            image = image.convert("RGB")
            with isolated_cpu_rng(seed_for(self.task, self.epoch, index, 2)):
                aux = (self.strong if self.branch == "S_full" else self.ordinary)(image)
        return index, anchor, aux, label, aux_id

    def release(self):
        self.bank.clear()


class RealOnlyManager:
    """Guard all statistics/APR dataset requests: no appended/generated samples."""
    def __init__(self, base):
        self.base = base
        self.real_paths = set(map(str, base._train_data))
        self.generated_statistics_count = 0

    def __getattr__(self, key):
        return getattr(self.base, key)

    def get_dataset(self, indices, source, mode, **kwargs):
        if kwargs.get("appendent") is not None:
            raise AssertionError("replay appendent forbidden")
        result = self.base.get_dataset(indices, source, mode, **kwargs)
        if source == "train":
            dataset = result[-1] if isinstance(result, tuple) else result
            paths = set(map(str, dataset.images))
            assert paths <= self.real_paths, "generated path reached statistics/APR source"
        return result
