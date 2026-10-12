from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
AUDIT = WORKSPACE / "structure_prototype_oracle_audit"
PROTOCOLS = ("IA10", "CAR10")
SEED = 1993
GENERATOR_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
DEFAULT_GENERATOR = WORKSPACE / ".cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots" / GENERATOR_REVISION


def bootstrap():
    for path in (WORKSPACE, AUDIT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_path(protocol, task):
    return AUDIT / "training_outputs" / protocol / "Structure_AUDIT/seed_1993/task_checkpoints" / f"task_{task:02d}.pt"


def source_config(protocol):
    value = read_json(checkpoint_path(protocol, 8).parent.parent / "resolved_config.json")
    if value["protocol_id"] != protocol or value["seed"] != SEED:
        raise ValueError("baseline protocol/seed mismatch")
    if not all(value[k] for k in ("scc", "apr", "ca")) or value["multi_center"] or value["old_new_separation"]:
        raise ValueError("expected vanilla SCC/APR/CA Structure")
    if len(value["task_partition"]) != 10:
        raise ValueError("expected ten tasks")
    return value


def data_manager(protocol):
    bootstrap()
    from oracle_audit.runtime import build_data_manager
    config = source_config(protocol)
    manager, _ = build_data_manager(protocol, config["data_root"], SEED)
    source_order = read_json(checkpoint_path(protocol, 8).parent.parent / "class_order.json")
    if manager._class_order != source_order["class_order_original_ids"]:
        raise ValueError("data manager class order differs from source")
    return manager


def sample_identity(manager, classes, source):
    bootstrap()
    from representation_relation.pipeline import _dataset_identity
    return _dataset_identity(manager, classes, source)


def run_dir(protocol, location, branch):
    return ROOT / "training" / protocol / location / branch


def default_config():
    return {"experiment": "Type A current generative augmentation", "seed": SEED,
            "protocols": list(PROTOCOLS), "locations": ["T1", "T9_T10"],
            "parent_max_per_class": 16, "parent_seed": 20261008,
            "support_seeds": [1993, 1994, 1995], "support_max_per_class": 64, "support_max_k": 5,
            "generator": {"model_id": "runwayml/stable-diffusion-v1-5",
                          "local_path": str(DEFAULT_GENERATOR), "revision": GENERATOR_REVISION,
                          "strength": 0.2, "steps": 30, "guidance": 7.5, "precision": "float16",
                          "image_size": 512, "scheduler": "checkpoint configuration"},
            "strong": {"RandAugment_num_ops": 2, "RandAugment_magnitude": 9, "RandomErasing_p": 0.25},
            "quality": {"max_sample_per_protocol": 100, "max_bad_or_uncertain_fraction": 0.05,
                        "max_no_variation_fraction": 0.5, "human_review_required": True},
            "conditions": ["V", "R", "S", "G", "GM"],
            "primary_comparison": "G-S", "secondary_comparison": "G-R", "statistics_comparison": "GM-G",
            "statistics": "R/S/G real; GM current real+generated; APR displacement remains real",
            "expansion_decision": {"birth_mean_G_minus_S_min_pp": 1.0,
                "each_birth_G_minus_S_min_pp": -0.5, "T9_seen_G_minus_S_min_pp": -0.5,
                "T9_cohort_after_T10_G_minus_S_min_pp": 0.5, "T10_seen_G_minus_S_min_pp": 0.0,
                "old_new_G_minus_S_min_pp": -1.0,
                "G_minus_R_rule": "birth mean > 0 pp AND T10 seen > 0 pp", "both_protocols_required": True},
            "test_used_for_source_or_recipe_selection": False,
            "requested_branch_name": "exp/type-a-current-generative-augmentation"}


def ensure_config():
    path = ROOT / "config.json"
    expected = default_config()
    if path.exists() and read_json(path) != expected:
        raise RuntimeError("existing config differs; use a separate experiment folder for a new recipe")
    write_json(path, expected)
    return expected
