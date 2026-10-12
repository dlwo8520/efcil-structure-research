"""Exact IA5/IA10/CAR5/CAR10 data protocols without dependency edits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from torchvision import datasets

from data.data import iCARS196_224, iImageNetA, split_images_labels
from data.data_manager import DataManager, _map_new_class_index
from utils.experiment_io import write_json


EXPECTED_PARTITIONS = {
    "IA5": [40, 40, 40, 40, 40],
    "IA10": [20] * 10,
    "CAR5": [40, 39, 39, 39, 39],
    "CAR10": [20, 20, 20, 20, 20, 20, 19, 19, 19, 19],
}


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_partition(protocol: str, partition, total_classes: int):
    values = [int(value) for value in partition]
    expected = EXPECTED_PARTITIONS.get(str(protocol).upper())
    if expected is None:
        raise ValueError(f"unknown protocol {protocol!r}")
    if values != expected:
        raise ValueError(f"{protocol} requires {expected}, got {values}")
    if any(value <= 0 for value in values) or sum(values) != int(total_classes):
        raise ValueError(f"invalid {protocol} partition {values} for {total_classes} classes")
    return values


class SweepDataManager(DataManager):
    """DataManager with shared class orders and exact non-divisible partitions."""

    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args=None):
        self.args = args or {}
        self.dataset_name = str(dataset_name)
        self.protocol_id = str(self.args.get("protocol_id", dataset_name)).upper()
        self.source_dataset = str(self.args["source_dataset"]).lower()
        configured_order_seed = self.args.get("class_order_seed")
        self.class_order_seed = int(
            self.args["seed"] if configured_order_seed is None else configured_order_seed
        )
        self._load_source(shuffle)
        total_classes = len(self._class_order)
        expected_classes = int(self.args["expected_num_classes"])
        if total_classes != expected_classes:
            raise ValueError(f"expected {expected_classes} classes, found {total_classes}")
        self._increments = validate_partition(
            self.protocol_id, self.args["task_partition"], total_classes
        )
        if len(self._increments) != int(self.args["expected_tasks"]):
            raise ValueError("task partition length differs from expected_tasks")
        self._validate_exact_class_coverage()
        self.protocol_manifest = self._build_sweep_manifest()
        run_dir = self.args.get("run_dir")
        if run_dir:
            write_json(
                Path(run_dir) / "task_partition.json",
                {
                    "protocol": self.protocol_id,
                    "total_classes": total_classes,
                    "task_partition": self._increments,
                    "class_ids_per_task": self.protocol_manifest["class_ids_per_task"],
                    "task_partition_hash": self.protocol_manifest["task_partition_hash"],
                },
            )

    def _load_source(self, shuffle: bool):
        root = Path(self.args.get("data_root") or "").expanduser().resolve()
        train_dir = Path(self.args.get("train_dir") or root / "train").resolve()
        test_dir = Path(self.args.get("test_dir") or root / "test").resolve()
        if not train_dir.is_dir() or not test_dir.is_dir():
            raise FileNotFoundError(f"expected train/ and test/ under {root}")

        if self.source_dataset == "imageneta":
            idata = iImageNetA()
            idata.download_data({**self.args, "data_root": str(root)})
            self._train_data, self._train_targets = idata.train_data, idata.train_targets
            self._test_data, self._test_targets = idata.test_data, idata.test_targets
            self.class_names = list(idata.class_names)
            self._train_trsf, self._test_trsf = idata.train_trsf, idata.test_trsf
            self._common_trsf, self.use_path = idata.common_trsf, idata.use_path
        elif self.source_dataset == "car196":
            train_set = datasets.ImageFolder(str(train_dir))
            test_set = datasets.ImageFolder(str(test_dir))
            if train_set.class_to_idx != test_set.class_to_idx:
                raise ValueError("Cars-196 train/test class mappings differ")
            self._train_data, self._train_targets = split_images_labels(train_set.imgs)
            self._test_data, self._test_targets = split_images_labels(test_set.imgs)
            self.class_names = list(train_set.classes)
            template = iCARS196_224()
            self._train_trsf, self._test_trsf = template.train_trsf, template.test_trsf
            self._common_trsf, self.use_path = template.common_trsf, template.use_path
        else:
            raise ValueError(f"unsupported source_dataset: {self.source_dataset}")

        train_labels = set(map(int, np.unique(self._train_targets)))
        test_labels = set(map(int, np.unique(self._test_targets)))
        if train_labels != test_labels:
            raise ValueError("train/test class label sets differ")
        if train_labels != set(range(len(self.class_names))):
            raise ValueError("ImageFolder labels are not contiguous or omit a class")
        class_count = len(self.class_names)
        if shuffle:
            self._class_order = np.random.RandomState(self.class_order_seed).permutation(class_count).tolist()
        else:
            self._class_order = list(range(class_count))
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        self.split_roots = {"train": str(train_dir), "test": str(test_dir)}

    def _validate_exact_class_coverage(self):
        class_ids = list(range(len(self._class_order)))
        start = 0
        partitioned = []
        for size in self._increments:
            partitioned.extend(class_ids[start : start + size])
            start += size
        if start != len(class_ids):
            raise ValueError("task partition does not consume every class")
        if len(partitioned) != len(set(partitioned)):
            raise ValueError("task partition duplicates a mapped class")
        if set(partitioned) != set(map(int, np.unique(self._train_targets))):
            raise ValueError("train labels are not represented exactly once in task partition")
        if set(partitioned) != set(map(int, np.unique(self._test_targets))):
            raise ValueError("test labels are not represented exactly once in task partition")

    def _build_sweep_manifest(self):
        manifest = super()._build_protocol_manifest()
        class_ids_per_task = [row["mapped_class_ids"] for row in manifest["tasks"]]
        train_counts = Counter(map(int, self._train_targets))
        test_counts = Counter(map(int, self._test_targets))
        train_per_task = [sum(train_counts[item] for item in ids) for ids in class_ids_per_task]
        test_per_task = [sum(test_counts[item] for item in ids) for ids in class_ids_per_task]
        split_records = manifest.get("splits", {})
        split_hash = canonical_hash(
            {
                name: split_records.get(name, {}).get("sha256")
                for name in ("train", "test")
            }
        )
        partition_record = {
            "sizes": list(self._increments),
            "class_ids_per_task": class_ids_per_task,
        }
        manifest.update(
            {
                "protocol": self.protocol_id,
                "source_dataset": self.source_dataset,
                "total_classes": len(self._class_order),
                "num_tasks": len(self._increments),
                "task_id": list(range(1, len(self._increments) + 1)),
                "classes_per_task": list(self._increments),
                "class_ids_per_task": class_ids_per_task,
                "class_order": list(self._class_order),
                "class_order_hash": canonical_hash(self._class_order),
                "task_partition": list(self._increments),
                "task_partition_hash": canonical_hash(partition_record),
                "train_total": len(self._train_targets),
                "test_total": len(self._test_targets),
                "train_count_per_class": dict(sorted(train_counts.items())),
                "test_count_per_class": dict(sorted(test_counts.items())),
                "train_count_per_task": train_per_task,
                "test_count_per_task": test_per_task,
                "seed": int(self.args["seed"]),
                "class_order_seed": self.class_order_seed,
                "data_split_hash": split_hash,
                "source_snapshot_hash": self.args.get("source_snapshot_hash"),
                "git_commit": self.args.get("git_commit"),
                "git_diff_hash": self.args.get("git_diff_hash"),
                "backbone": self.args.get("convnet_type"),
                "adapter": {
                    "ffn_num": self.args.get("ffn_num"),
                    "placement": self.args.get("adapter_placement"),
                    "interval": self.args.get("adapter_interval"),
                },
                "optimizer": self.args.get("optimizer"),
                "lr": self.args.get("init_lr"),
                "scheduler": "cosine_annealing",
                "batch_size": self.args.get("batch_size"),
                "feature_epochs": {
                    "initial": self.args.get("init_epochs"),
                    "incremental": self.args.get("inc_epochs"),
                },
                "classifier_epochs": self.args.get("ca_epochs"),
                "replay_samples_per_class": self.args.get("replay_samples_per_class"),
                "coverage_validation": {
                    "all_class_ids_unique": True,
                    "no_missing_classes": True,
                    "no_duplicated_classes": True,
                    "all_train_test_labels_partitioned_once": True,
                },
            }
        )
        return manifest
