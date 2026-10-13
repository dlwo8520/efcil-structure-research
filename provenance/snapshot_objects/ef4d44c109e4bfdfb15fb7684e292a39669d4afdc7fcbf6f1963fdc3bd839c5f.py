#!/usr/bin/env python3
"""Run one protocol/method on a selectable physical GPU (GPU 1 by default)."""

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


def parse_args():
    parser = argparse.ArgumentParser(description="Similarity-selective Structure sweep")
    parser.add_argument("--protocol", choices=("IA5", "IA10", "CAR5", "CAR10"), required=True)
    parser.add_argument(
        "--run-id", required=True,
        help="Structure, C0, A_P005, B_P010_A050, ...",
    )
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--e3-root", default=str(DEFAULT_E3))
    parser.add_argument("--similarity-root", default=str(DEFAULT_SIMILARITY))
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", default="1", help="physical GPU ID(s); default: 1")
    device.add_argument("--cpu", action="store_true", help="CPU mode for diagnostics/smoke only")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", action="append")
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="test-only config override")
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
    return config if "extends" not in config else merge(load_config(path.parent / config["extends"], visited), config)


def parse_seeds(values):
    return [int(item) for value in values for item in value.split(",") if item.strip()]


def apply_overrides(config, assignments):
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"--set expects KEY=JSON: {assignment}")
        key, raw = assignment.split("=", 1)
        config[key] = json.loads(raw)


def validate_config(config):
    required = {
        "experiment_id", "protocol_id", "dataset", "source_dataset", "seed",
        "expected_num_classes", "expected_tasks", "task_partition", "class_order_seed",
        "multi_center", "num_prototypes", "old_new_separation", "separation_covariance",
        "sep_lambda", "sep_percentile", "sep_tau_floor", "sep_reduction", "sep_timing",
        "sep_boundary_policy", "sep_refresh_epochs", "replay_samples_per_class",
        "selective_method", "method", "top_p", "alpha", "lambda_clad",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if config["sep_reduction"] != "max_relative_intrusion" or config["sep_timing"] != "epoch_apr_cache":
        raise ValueError("the inherited controlled baseline requires max_relative_intrusion/epoch_apr_cache")
    if config["sep_boundary_policy"] != "birth_quantile_fixed":
        raise ValueError("only the fixed birth-quantile boundary is supported")
    if config["separation_covariance"] != "diagonal" or config["multi_center"]:
        raise ValueError("this sweep requires single-center diagonal regions")
    from selective.protocol import validate_partition
    validate_partition(config["protocol_id"], config["task_partition"], config["expected_num_classes"])
    method = config["method"]
    if method not in {"Structure", "C0", "A", "B", "C"}:
        raise ValueError(f"invalid method {method}")
    if method in {"A", "B", "C"} and not 0.0 < float(config["top_p"]) <= 1.0:
        raise ValueError("A/B/C require top_p in (0,1]")
    if method == "B" and float(config["alpha"]) not in {0.25, 0.5, 1.0}:
        raise ValueError("invalid Method-B alpha")
    if method == "C" and float(config["lambda_clad"]) not in {0.1, 0.3, 1.0, 3.0}:
        raise ValueError("invalid Proto-CLAD coefficient")
    if method == "Structure":
        expected = {
            "old_new_separation": False,
            "adaptive_margin_mode": "none",
            "selective_method": "none",
            "similarity_relation_mode": "none",
            "similarity_scc_enabled": False,
            "top_p": None,
            "alpha": None,
            "lambda_clad": None,
        }
        mismatches = {
            key: {"expected": value, "actual": config.get(key)}
            for key, value in expected.items()
            if config.get(key) != value
        }
        if float(config["sep_lambda"]) != 0.0:
            mismatches["sep_lambda"] = {
                "expected": 0.0,
                "actual": config.get("sep_lambda"),
            }
        if mismatches:
            raise ValueError(f"invalid vanilla Structure overrides: {mismatches}")


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
    base_root = Path(cli.base_root).expanduser().resolve()
    e3_root = Path(cli.e3_root).expanduser().resolve()
    similarity_root = Path(cli.similarity_root).expanduser().resolve()
    for path, marker in (
        (base_root, "trainer.py"), (e3_root, "followup/learner.py"),
        (similarity_root, "similarity/learner.py"),
    ):
        if not (path / marker).is_file():
            raise FileNotFoundError(f"invalid dependency root {path}: missing {marker}")

    if cli.cpu or cli.validate_only:
        physical_gpus, logical_devices = [], [-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        physical_gpus = [value.strip() for value in cli.gpu.split(",") if value.strip()]
        if not physical_gpus or any(not value.isdigit() for value in physical_gpus):
            raise ValueError("--gpu must contain non-negative physical GPU IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
        logical_devices = list(range(len(physical_gpus)))

    # Import priority: this extension, similarity layer, E3 layer, Structure base.
    sys.path.insert(0, str(base_root))
    sys.path.insert(0, str(e3_root))
    sys.path.insert(0, str(similarity_root))
    sys.path.insert(0, str(ROOT))
    from selective.registry import protocol_config_path, resolve_method
    from selective.snapshot import tree_digest
    from similarity.run_compat import validate_followup_compat

    config_path = protocol_config_path(cli.protocol)
    config = merge(load_config(config_path), resolve_method(cli.run_id))
    apply_overrides(config, cli.set)
    config["seed"] = parse_seeds(cli.seed) if cli.seed else list(config["seed"])
    if cli.data_root:
        config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
    if not config.get("data_root") and not cli.validate_only:
        raise ValueError("--data-root is required for execution")
    config["output_root"] = str(Path(cli.output_root).expanduser().resolve())
    source_hash, _ = tree_digest(ROOT)
    commit, diff_hash = git_state(ROOT)
    config.update(
        {
            "config": str(config_path), "base_root": str(base_root),
            "extension_root": str(e3_root), "e3_root": str(e3_root),
            "similarity_extension_root": str(similarity_root),
            "selective_sweep_root": str(ROOT), "selective_sweep_extension": True,
            "device": logical_devices, "physical_gpus": physical_gpus,
            "launch_command": " ".join(shlex.quote(value) for value in sys.argv),
            "source_snapshot_hash": source_hash, "git_commit": commit, "git_diff_hash": diff_hash,
        }
    )
    validate_config(config)
    validate_followup_compat(config)
    if cli.resume:
        if len(config["seed"]) != 1:
            raise ValueError("--resume accepts exactly one seed")
        config["resume"] = str(Path(cli.resume).expanduser().resolve())
    if cli.validate_only:
        print(
            f"VALID protocol={config['protocol_id']} run_id={config['experiment_id']} "
            f"method={config['method']} partition={config['task_partition']} seeds={config['seed']}"
        )
        return

    import torch
    import trainer
    from selective.diagnostics import export_run_diagnostics
    from selective.exports import export_all, export_protocol
    from selective.learner import SelectiveSimilarityLearner
    from selective.protocol import SweepDataManager

    original_factory = trainer.model_factory.get_model
    trainer.DataManager = SweepDataManager

    def get_model(model_name, args):
        if bool(args.get("selective_sweep_extension", False)):
            return SelectiveSimilarityLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_model
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible physical GPUs: {physical_gpus if physical_gpus else 'CPU'}")
    print(f"--- Starting {config['protocol_id']}/{config['experiment_id']} ---", flush=True)
    trainer.CDS_train(config)
    for seed in config["seed"]:
        run_dir = Path(config["output_root"]) / config["dataset"] / config["experiment_id"] / f"seed_{seed}"
        per_seed = dict(config, seed=int(seed))
        export_run_diagnostics(run_dir, per_seed)
    export_protocol(config["output_root"], config["protocol_id"])
    export_all(config["output_root"], ("IA5", "IA10", "CAR5", "CAR10"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"--- Finished {config['protocol_id']}/{config['experiment_id']} ---", flush=True)


if __name__ == "__main__":
    main()
