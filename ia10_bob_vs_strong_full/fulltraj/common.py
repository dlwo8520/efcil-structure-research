from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
BASELINE = WORKSPACE / "structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993"
OFFICIAL_COMMIT = "8bc17e0537991f6bb5c0399339bddb3d1418d220"
VENDOR = ROOT / "vendor/BeyondObjects"
SEED = 1993
BRANCHES = ("S_full", "BOB_full")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refuse empty result: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat()


def spec():
    value = read_json(ROOT / "experiment_spec.json")
    assert value["protocol"] == "IA10" and value["seed"] == SEED
    assert value["task_partition"] == [20] * 10
    assert value["branches"] == list(BRANCHES)
    assert value["statistics_source"] == "real_train_only"
    assert value["bob"]["repo_commit"] == OFFICIAL_COMMIT
    return value


def experiment_hash():
    # Local implementation and upstream code are immutable after a run starts.
    files = sorted((ROOT / "fulltraj").glob("*.py")) + [ROOT / "run.py", ROOT / "experiment_spec.json", ROOT / "requirements-bob.txt"]
    files += [VENDOR / p for p in ("finetune/finetune_t2i.py", "finetune/config.py", "generation/generate.py")]
    files += [ROOT / "assets/imagenet_synset_to_lemma.tsv"]
    files += [Path(spec()["reference_trajectory"])]
    if (ROOT / "assets/model_lock.json").exists():
        files += [ROOT / "assets/model_lock.json"]
    files += [BASELINE / p for p in ("resolved_config.json", "class_order.json", "protocol_manifest.json")]
    files += [WORKSPACE / p for p in (
        "structure_multicenter_separation/models/CDS_adapter.py",
        "structure_multicenter_separation/models/base.py",
        "structure_multicenter_separation/utils/distill_loss.py",
        "structure_multicenter_separation/utils/loss.py",
        "structure_multicenter_separation/utils/adjust_distribution.py",
        "structure_multicenter_separation/network/vision_transformer_adapter.py",
        "structure_prototype_oracle_audit/oracle_audit/snapshot_learner.py",
        "type_a_generative_augmentation/matched_views.py")]
    return digest({str(p): sha256(p) for p in files})


def bootstrap():
    for p in (WORKSPACE, WORKSPACE / "structure_prototype_oracle_audit"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()
    # Always use the unique fulltraj package; never import a generic analysis.py.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def source_config():
    value = read_json(BASELINE / "resolved_config.json")
    assert value["seed"] == SEED and value["protocol_id"] == "IA10"
    assert value["task_partition"] == [20] * 10
    assert all(value[k] for k in ("apr", "scc", "ca"))
    assert not value["multi_center"] and not value["old_new_separation"]
    return value


def manager():
    bootstrap()
    from oracle_audit.runtime import build_data_manager
    value, _ = build_data_manager("IA10", source_config()["data_root"], SEED)
    old = read_json(BASELINE / "protocol_manifest.json")
    expected = read_json(BASELINE / "class_order.json")["class_order_original_ids"]
    assert value._class_order == expected, "do not generate a new class order"
    for key in ("class_order_hash", "task_partition_hash", "data_split_hash", "train_total", "test_total"):
        assert value.protocol_manifest[key] == old[key], f"baseline protocol mismatch: {key}"
    return value


def task_dir(task):
    if not 1 <= int(task) <= 10:
        raise ValueError("task must be 1..10")
    return ROOT / "bob_tasks" / f"T{int(task):02d}"


def train_dir(branch):
    if branch not in BRANCHES:
        raise ValueError(branch)
    return ROOT / "checkpoints" / branch


def progress(stage, **fields):
    value = {"updated_at": now(), "stage": stage, **fields}
    write_json(ROOT / "progress.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def configure_gpu(gpu):
    gpu = str(gpu).strip()
    if not gpu.isdigit():
        raise ValueError("enter one physical GPU number, e.g. 4")
    # Must happen BEFORE torch imports; physical GPU maps to logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu
