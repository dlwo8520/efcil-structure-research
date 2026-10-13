#!/usr/bin/env python3
"""Fast filesystem-only validation for processed CIL ImageFolder splits."""

import argparse
import os
from pathlib import Path


SPECS = {
    "imageneta": ("IMAGENETA_ROOT", "data/imagenet-a", 200),
    "cars196": ("CARS196_ROOT", "data/cars196", 196),
    "objectnet": ("OBJECTNET_ROOT", "data/objectnet", 200),
    "omnibenchmark": ("OMNIBENCHMARK_ROOT", "data/omnibenchmark", 300),
    "imagenetr": ("IMAGENETR_ROOT", "data/imagenet-r", 200),
}


def count_split(root, split):
    directory = root / split
    if not directory.is_dir():
        raise FileNotFoundError("Missing split: {}".format(directory))
    classes = sorted(path.name for path in directory.iterdir() if path.is_dir())
    files = sum(1 for path in directory.rglob("*") if path.is_file())
    return classes, files


def validate(name):
    env_name, default, expected = SPECS[name]
    root = Path(os.path.expanduser(os.environ.get(env_name, default))).resolve()
    train_classes, train_files = count_split(root, "train")
    test_classes, test_files = count_split(root, "test")
    if train_classes != test_classes:
        raise ValueError("{} train/test class names differ".format(name))
    if len(train_classes) != expected:
        raise ValueError("{} expected {} classes, found {}".format(name, expected, len(train_classes)))
    if not train_files or not test_files:
        raise ValueError("{} has an empty split".format(name))
    print("[OK] {} root={} classes={} train={} test={}".format(
        name, root, len(train_classes), train_files, test_files
    ))


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase1", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    names = ["imageneta", "cars196"] if args.phase1 else list(SPECS)
    for name in names:
        validate(name)


if __name__ == "__main__":
    main()
