#!/usr/bin/env python3
"""Run C0-C4 on physical GPU 2 by default, keeping dependencies read-only."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT.parent / "structure_multicenter_separation"
DEFAULT_E3 = ROOT.parent / "structure_e3_followup"
DEFAULT_SIMILARITY = ROOT.parent / "structure_similarity_aware_followup"


def parse_args():
    parser = argparse.ArgumentParser(description="Similarity-selective controls C0-C4")
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--base-root", default=str(DEFAULT_BASE))
    parser.add_argument("--e3-root", default=str(DEFAULT_E3))
    parser.add_argument("--similarity-root", default=str(DEFAULT_SIMILARITY))
    parser.add_argument("--gpu", default="2", help="physical GPU IDs; default: 2")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", action="append")
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
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
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "extends" not in config:
        return config
    return merge(load_config(path.parent / config["extends"], visited), config)


def parse_seeds(values):
    if not values:
        return None
    return [int(item) for value in values for item in value.split(",") if item.strip()]


def main():
    cli = parse_args()
    base_root = Path(cli.base_root).expanduser().resolve()
    e3_root = Path(cli.e3_root).expanduser().resolve()
    similarity_root = Path(cli.similarity_root).expanduser().resolve()
    for path, marker in (
        (base_root, "trainer.py"),
        (e3_root, "followup/learner.py"),
        (similarity_root, "similarity/learner.py"),
    ):
        if not (path / marker).is_file():
            raise FileNotFoundError(f"invalid dependency root: {path} (missing {marker})")

    if cli.cpu or cli.validate_only:
        physical_gpus, logical_devices = [], [-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        physical_gpus = [item.strip() for item in cli.gpu.split(",") if item.strip()]
        if not physical_gpus or any(not item.isdigit() for item in physical_gpus):
            raise ValueError("--gpu must contain non-negative integer IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
        logical_devices = list(range(len(physical_gpus)))

    sys.path.insert(0, str(base_root))
    sys.path.insert(0, str(e3_root))
    sys.path.insert(0, str(similarity_root))
    sys.path.insert(0, str(ROOT))
    from control.run_compat import validate_control_config
    from main import validate_config as validate_base_config
    from similarity.run_compat import validate_followup_compat

    seeds = parse_seeds(cli.seed)
    resolved = []
    for config_name in cli.config:
        config_path = Path(config_name).expanduser().resolve()
        config = load_config(config_path)
        validate_base_config(config)
        validate_followup_compat(config)
        validate_control_config(config)
        config.update(
            {
                "config": str(config_path),
                "base_root": str(base_root),
                "extension_root": str(e3_root),
                "e3_root": str(e3_root),
                "similarity_extension_root": str(similarity_root),
                "control_extension_root": str(ROOT),
                "control_extension": True,
                "device": logical_devices,
                "physical_gpus": physical_gpus,
                "launch_command": " ".join(shlex.quote(value) for value in sys.argv),
            }
        )
        if cli.data_root:
            config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
        output = Path(cli.output_root or config["output_root"]).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        config["output_root"] = str(output.resolve())
        if seeds is not None:
            config["seed"] = seeds
        if cli.resume:
            config["resume"] = str(Path(cli.resume).expanduser().resolve())
        resolved.append(config)
    if cli.resume and (len(resolved) != 1 or len(resolved[0]["seed"]) != 1):
        raise ValueError("--resume accepts one method and one seed")
    if cli.validate_only:
        for config in resolved:
            print(
                f"VALID {config['experiment_id']}: seeds={config['seed']} "
                f"selection={config['control_selection_mode']} "
                f"fraction={config['control_fraction']} weak={config['control_weak_weight']}"
            )
        return

    import torch
    import trainer
    from control.learner import ControlLearner
    from similarity.exports import export_output_tree

    original_factory = trainer.model_factory.get_model

    def get_control_model(model_name, args):
        if bool(args.get("control_extension", False)):
            return ControlLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_control_model
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible physical GPUs: {physical_gpus if physical_gpus else 'CPU'}")
    for config in resolved:
        print(f"--- Starting {config['experiment_id']}: {config['config']} ---")
        trainer.CDS_train(config)
        export_output_tree(Path(config["output_root"]) / config["dataset"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--- Finished {config['experiment_id']} ---")


if __name__ == "__main__":
    main()

