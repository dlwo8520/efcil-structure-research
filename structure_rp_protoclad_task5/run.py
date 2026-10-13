#!/usr/bin/env python3
"""Run one of the four fixed RP-ProtoCLAD Task-5 configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT.parent / "structure_multicenter_separation"
DEFAULT_E3 = ROOT.parent / "structure_e3_followup"
DEFAULT_SIMILARITY = ROOT.parent / "structure_similarity_aware_followup"
DEFAULT_SELECTIVE = ROOT.parent / "structure_selective_similarity_sweep"


def parse_args():
    parser = argparse.ArgumentParser(description="Four-run RP-ProtoCLAD Task-5 experiment")
    parser.add_argument("--config", required=True, help="one of configs/RP_C_*_{IA5,CAR5}.json")
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--e3-root", default=str(DEFAULT_E3))
    parser.add_argument("--similarity-root", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--selective-root", default=str(DEFAULT_SELECTIVE))
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", default="2", help="one physical GPU ID; default: 2")
    device.add_argument("--cpu", action="store_true", help="validation/sanity use only")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, choices=(1993,), default=1993)
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    return parser.parse_args()


def merge(parent, child):
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path, seen=None):
    path = Path(path).expanduser().resolve()
    visited = set() if seen is None else set(seen)
    if path in visited:
        raise ValueError(f"circular config inheritance: {path}")
    visited.add(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in config:
        return config
    return merge(load_config(path.parent / config["extends"], visited), config)


def apply_overrides(config, assignments):
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"--set expects KEY=JSON: {assignment}")
        key, raw = assignment.split("=", 1)
        config[key] = json.loads(raw)


def _parent_config(config, selective_root):
    sys.path.insert(0, str(selective_root))
    from selective.registry import protocol_config_path, resolve_method

    return merge(
        load_config(protocol_config_path(config["protocol_id"])),
        resolve_method(config["parent_run_id"]),
    )


def validate_config(config, selective_root):
    from rp.registry import PROTOCOLS, RP_DIM, RP_INPUT_DIM, RP_SEED, RUN_IDS, definition

    required = {
        "experiment_id", "parent_run_id", "protocol_id", "dataset", "source_dataset",
        "seed", "expected_num_classes", "expected_tasks", "task_partition",
        "class_order_seed", "old_new_separation", "separation_covariance", "sep_lambda",
        "selective_method", "similarity_relation_mode", "method", "top_p", "alpha",
        "lambda_clad", "rp_enabled", "rp_dim", "rp_seed", "rp_input_dim",
        "rp_activation", "rp_bias", "rp_trainable", "rp_matrix_shape",
        "rp_matrix_checksum",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"RP config missing keys: {missing}")
    protocol = str(config["protocol_id"])
    run_id = str(config["experiment_id"])
    if protocol not in PROTOCOLS or run_id not in RUN_IDS:
        raise ValueError(f"only IA5/CAR5 and the two fixed RP run IDs are allowed: {protocol}/{run_id}")
    spec = definition(protocol, run_id)
    exact = {
        "expected_tasks": 5,
        "parent_run_id": spec["parent_run_id"],
        "method": "C",
        "selective_method": "rp_proto_clad",
        "similarity_relation_mode": "rp_proto_clad",
        "top_p": 0.20,
        "alpha": None,
        "lambda_clad": spec["lambda_clad"],
        "sep_lambda": spec["lambda_clad"],
        "rp_enabled": True,
        "rp_dim": RP_DIM,
        "rp_seed": RP_SEED,
        "rp_input_dim": RP_INPUT_DIM,
        "rp_activation": "relu",
        "rp_bias": False,
        "rp_trainable": False,
        "rp_matrix_shape": [RP_INPUT_DIM, RP_DIM],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in exact.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"invalid fixed RP configuration: {mismatches}")
    if not config["old_new_separation"] or config["separation_covariance"] != "diagonal":
        raise ValueError("RP-ProtoCLAD requires the unchanged active diagonal relation cache")
    if config["seed"] != [1993]:
        raise ValueError("this experiment permits seed 1993 only")

    parent = _parent_config(config, selective_root)
    allowed_changes = {
        "prefix", "experiment_id", "method_name", "selective_method",
        "similarity_relation_mode", "parent_run_id", "rp_enabled", "rp_dim", "rp_seed",
        "rp_input_dim", "rp_activation", "rp_bias", "rp_trainable", "rp_matrix_shape",
        "rp_matrix_checksum", "rp_projection", "data_root", "output_root",
    }
    unexpected = {
        key: {"parent": value, "rp": config.get(key)}
        for key, value in parent.items()
        if key not in allowed_changes and config.get(key) != value
    }
    if unexpected:
        raise ValueError(f"RP run differs from its parent outside RP geometry: {unexpected}")

    from selective.protocol import validate_partition
    from similarity.run_compat import validate_followup_compat

    validate_partition(protocol, config["task_partition"], config["expected_num_classes"])
    validate_followup_compat(config)


def git_state(root):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
        return commit, hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return None, None


def main():
    cli = parse_args()
    roots = {
        "base": Path(cli.base_root).expanduser().resolve(),
        "e3": Path(cli.e3_root).expanduser().resolve(),
        "similarity": Path(cli.similarity_root).expanduser().resolve(),
        "selective": Path(cli.selective_root).expanduser().resolve(),
    }
    for name, root, marker in (
        ("base", roots["base"], "trainer.py"),
        ("e3", roots["e3"], "followup/learner.py"),
        ("similarity", roots["similarity"], "similarity/learner.py"),
        ("selective", roots["selective"], "selective/learner.py"),
    ):
        if not (root / marker).is_file():
            raise FileNotFoundError(f"invalid {name} dependency root {root}: missing {marker}")

    if cli.cpu or cli.validate_only:
        physical_gpus, logical_devices = [], [-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        if not cli.gpu.isdigit():
            raise ValueError("--gpu must be one non-negative physical GPU ID")
        physical_gpus = [cli.gpu]
        os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
        logical_devices = [0]

    # Highest import priority is the isolated RP extension, followed by the
    # pinned selective/similarity/E3/Structure dependency layers.
    for root in (roots["base"], roots["e3"], roots["similarity"], roots["selective"], ROOT):
        sys.path.insert(0, str(root))

    from rp.projection import make_projection, projection_metadata
    from rp.snapshot import tree_digest

    config_path = Path(cli.config).expanduser().resolve()
    config = load_config(config_path)
    apply_overrides(config, cli.set)
    config["seed"] = [cli.seed]
    matrix = make_projection(config["rp_input_dim"], config["rp_dim"], config["rp_seed"])
    metadata = projection_metadata(matrix, config["rp_seed"])
    config["rp_matrix_shape"] = metadata["shape"]
    config["rp_matrix_checksum"] = metadata["checksum_sha256"]
    config["rp_projection"] = metadata
    if cli.data_root:
        config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
    if not config.get("data_root") and not cli.validate_only:
        raise ValueError("--data-root is required for execution")
    config["output_root"] = str(Path(cli.output_root).expanduser().resolve())
    source_hash, source_count = tree_digest(ROOT)
    commit, diff_hash = git_state(ROOT)
    config.update(
        {
            "config": str(config_path),
            "base_root": str(roots["base"]),
            "extension_root": str(roots["e3"]),
            "e3_root": str(roots["e3"]),
            "similarity_extension_root": str(roots["similarity"]),
            "selective_sweep_root": str(roots["selective"]),
            "rp_extension_root": str(ROOT),
            "selective_sweep_extension": True,
            "rp_protoclad_extension": True,
            "device": logical_devices,
            "physical_gpus": physical_gpus,
            "launch_command": " ".join(shlex.quote(value) for value in sys.argv),
            "source_snapshot_hash": source_hash,
            "source_file_count": source_count,
            "git_commit": commit,
            "git_diff_hash": diff_hash,
        }
    )
    validate_config(config, roots["selective"])
    if cli.resume:
        config["resume"] = str(Path(cli.resume).expanduser().resolve())
    if cli.validate_only:
        print(
            f"VALID protocol={config['protocol_id']} run_id={config['experiment_id']} "
            f"parent={config['parent_run_id']} rp_shape={config['rp_matrix_shape']} "
            f"checksum={config['rp_matrix_checksum']} seed={config['seed']}"
        )
        return

    import torch
    import trainer
    from rp.diagnostics import export_rp_run_diagnostics
    from rp.learner import RPProtoCLADLearner
    from selective.diagnostics import export_run_diagnostics
    from selective.protocol import SweepDataManager

    original_factory = trainer.model_factory.get_model
    trainer.DataManager = SweepDataManager

    def get_model(model_name, args):
        if bool(args.get("rp_protoclad_extension", False)):
            return RPProtoCLADLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_model
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible physical GPU: {physical_gpus[0]}")
    print(f"--- Starting {config['protocol_id']}/{config['experiment_id']} ---", flush=True)
    trainer.CDS_train(config)
    run_dir = (
        Path(config["output_root"])
        / config["dataset"]
        / config["experiment_id"]
        / f"seed_{cli.seed}"
    )
    per_seed = dict(config, seed=cli.seed)
    export_run_diagnostics(run_dir, per_seed)
    export_rp_run_diagnostics(run_dir, per_seed)
    try:
        from analysis.build_report import build_report

        build_report()
    except Exception as error:
        print(f"WARNING: report refresh failed without invalidating training: {error}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"--- Finished {config['protocol_id']}/{config['experiment_id']} ---", flush=True)


if __name__ == "__main__":
    main()
