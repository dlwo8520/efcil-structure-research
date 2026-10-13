"""Train-only dataset adapter to upstream BOB's directory/TSV interface."""
from __future__ import annotations

import csv
import os
from pathlib import Path

from .common import (ROOT, BASELINE, VENDOR, digest, manager, read_json, read_csv,
                     sha256, spec, task_dir, write_csv, write_json)


def class_names():
    path = ROOT / "assets/imagenet_synset_to_lemma.tsv"
    if not path.is_file():
        raise FileNotFoundError("missing ImageNet synset mapping; run bash setup_bob_env.sh")
    with path.open(encoding="utf-8", newline="") as stream:
        return {row[0]: row[1].split(",")[0].strip() for row in csv.reader(stream, delimiter="\t") if len(row) >= 2}


def current_records(data_manager, task):
    lo, hi = (task - 1) * 20, task * 20
    names = class_names()
    records = []
    # No test dataset and no future-class image is exposed to the generator.
    for index, (path, label) in enumerate(zip(data_manager._train_data, data_manager._train_targets)):
        label = int(label)
        if not lo <= label < hi:
            continue
        metadata = data_manager.get_class_metadata(label)
        wnid = metadata["class_name"]
        if wnid not in names:
            raise KeyError(f"missing semantic class name for {wnid}; never prompt with a raw WNID")
        source = Path(str(path)).resolve()
        assert source.is_relative_to(Path(data_manager.split_roots["train"]).resolve())
        records.append({"task": task, "class_id": label, "wnid": wnid,
                        "class_name": names[wnid], "source_id": f"train:{index:06d}",
                        "source_path": str(source), "source_sha256": sha256(source)})
    assert {r["class_id"] for r in records} == set(range(lo, hi))
    return records


def prepare(task):
    dm = manager()
    records = current_records(dm, task)
    target = task_dir(task)
    for row in records:
        # English class names are exactly the directory names consumed by BOB.
        name = row["class_name"]
        if "/" in name or name in (".", ".."):
            raise ValueError("unsafe class name")
        link = target / "real" / name / (row["source_id"].replace(":", "_") + Path(row["source_path"]).suffix)
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            assert link.resolve() == Path(row["source_path"])
        elif link.exists():
            raise RuntimeError(f"unexpected existing file: {link}")
        else:
            link.symlink_to(row["source_path"])
        row["bob_input_path"] = str(link.absolute())
    if len({r["class_name"] for r in records}) != 20:
        raise ValueError("semantic directory names collide")
    identity = {"task": task, "train_source_hash": digest(records),
                "original_class_order_hash": dm.protocol_manifest["class_order_hash"],
                "data_split_hash": dm.protocol_manifest["data_split_hash"],
                "source_class_order_sha256": sha256(BASELINE / "class_order.json"),
                "n_real": len(records), "n_classes": 20, "future_or_test_images": 0,
                "synset_mapping_sha256": sha256(ROOT / "assets/imagenet_synset_to_lemma.tsv")}
    path = target / "input_manifest.json"
    if path.exists() and read_json(path) != identity:
        raise RuntimeError("train inputs changed after experiment preparation")
    write_csv(target / "real_train.csv", records)
    write_json(path, identity)
    return records


def validate_captions(task, target=None):
    target = task_dir(task) if target is None else Path(target)
    records = read_csv(target / "real_train.csv")
    captions = read_csv(target / "captions.csv")
    by_id = {r["source_id"]: r for r in captions}
    assert len(by_id) == len(captions) == len(records), "all real images need both context captions"
    for row in records:
        cap = by_id[row["source_id"]]
        assert cap["source_sha256"] == row["source_sha256"]
        assert int(cap["task"]) == task
        assert cap["background"].strip() and cap["pose"].strip()
    return records, by_id


def export_official_inputs(task, target=None):
    target = task_dir(task) if target is None else Path(target)
    records, captions = validate_captions(task, target)
    capdir = target / "captions"
    capdir.mkdir(exist_ok=True)
    for field in ("background", "pose"):
        with (capdir / (field + ".tsv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["path", "caption"])
            for row in records:
                writer.writerow([row["bob_input_path"], captions[row["source_id"]][field]])
    # JSON is valid YAML; upstream yaml.safe_load consumes this without modifications.
    config = {"model": spec()["bob"]["base_local_path"], "data_path": str(target / "real"),
              "caption_path": str(capdir), "lora_path": str(target / "lora"),
              "synthetic_path": str(target / "generated"), "template": spec()["bob"]["template"],
              "dataset": "imageneta_current_task", "num_synth": spec()["bob"]["images_per_class"],
              "classnames": list(dict.fromkeys(row["class_name"] for row in records))}
    write_json(target / "official_config.yaml", config)
    return config
