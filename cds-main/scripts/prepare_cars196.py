#!/usr/bin/env python3
"""Convert the official Stanford Cars split into torchvision ImageFolder trees."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scipy.io import loadmat


EXPECTED_COUNTS = {"train": 8_144, "test": 8_041}
NUM_CLASSES = 196


def annotations(path: Path) -> list[tuple[str, int]]:
    rows = loadmat(path, squeeze_me=True)["annotations"]
    result: list[tuple[str, int]] = []
    for row in rows:
        # Stanford Cars MATLAB fields: bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        # class (one-based), fname.
        class_id = int(row[4])
        filename = str(row[5])
        if not 1 <= class_id <= NUM_CLASSES:
            raise ValueError(f"Invalid class {class_id} for {filename} in {path}")
        result.append((filename, class_id))
    return result


def build_split(root: Path, split: str, annotation_path: Path) -> dict[str, int]:
    source_dir = root / "_raw" / f"cars_{split}"
    destination_dir = root / split
    rows = annotations(annotation_path)
    if len(rows) != EXPECTED_COUNTS[split]:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNTS[split]} {split} annotations, found {len(rows)}"
        )
    if len({filename for filename, _ in rows}) != len(rows):
        raise RuntimeError(f"Duplicate filenames in {annotation_path}")

    counts: Counter[int] = Counter()
    for filename, class_id in rows:
        source = source_dir / filename
        target_dir = destination_dir / f"class_{class_id:03d}"
        target = target_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"Existing target differs from source: {target}")
        else:
            # A hard link creates the required ImageFolder view without duplicating
            # the extracted image bytes. It is safe because both paths are local.
            os.link(source, target)
        counts[class_id] += 1

    if set(counts) != set(range(1, NUM_CLASSES + 1)):
        raise RuntimeError(f"{split} does not contain all {NUM_CLASSES} classes")
    return {f"class_{key:03d}": counts[key] for key in sorted(counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cars196",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    download_dir = root / "_downloads"

    train_counts = build_split(
        root, "train", download_dir / "cars_train_annos.mat"
    )
    test_counts = build_split(
        root, "test", download_dir / "cars_test_annos_withlabels.mat"
    )
    manifest = {
        "dataset": "Stanford Cars-196",
        "root": str(root),
        "layout": "official train/test split in torchvision ImageFolder format",
        "num_classes": NUM_CLASSES,
        "train_images": sum(train_counts.values()),
        "test_images": sum(test_counts.values()),
        "train_per_class": train_counts,
        "test_per_class": test_counts,
    }
    with (root / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in manifest.items() if not key.endswith("_per_class")}, indent=2))


if __name__ == "__main__":
    main()
