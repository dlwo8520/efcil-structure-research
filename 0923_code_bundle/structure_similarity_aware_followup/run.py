#!/usr/bin/env python3
"""Run similarity-aware Structure experiments without editing either dependency."""

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
VALID_RELATION_MODES = {
    "none",
    "legacy_absolute",
    "similarity_tau",
    "hard_relative",
    "soft_hybrid",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Structure similarity-aware follow-up")
    parser.add_argument("--config", nargs="+", required=True, help="JSON config(s)")
    parser.add_argument("--base-root", default=str(DEFAULT_BASE), help="read-only Structure tree")
    parser.add_argument("--e3-root", default=str(DEFAULT_E3), help="read-only E3 extension tree")
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--gpu", help="physical GPU IDs, e.g. 1 or 1,2")
    device.add_argument("--cpu", action="store_true", help="CPU mode for validation/tests")
    parser.add_argument("--data-root", help="ImageNet-A root containing train/ and test/")
    parser.add_argument("--output-root", help="override output root")
    parser.add_argument("--seed", action="append", help="repeat or comma-separate seed overrides")
    parser.add_argument("--resume", help="checkpoint_latest.pt (one method/seed only)")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _merge(parent, child):
    merged = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path, seen=None):
    config_path = Path(path).expanduser().resolve()
    visited = set() if seen is None else set(seen)
    if config_path in visited:
        raise ValueError(f"circular config inheritance: {config_path}")
    visited.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "extends" not in config:
        return config
    return _merge(load_config(config_path.parent / config["extends"], visited), config)


def parse_seeds(values):
    if not values:
        return None
    return [int(item) for value in values for item in value.split(",") if item.strip()]


def validate_similarity_config(config):
    required = {
        "similarity_relation_mode",
        "similarity_alpha",
        "relative_margin",
        "similarity_beta_source",
        "similarity_beta_quantile",
        "similarity_scc_enabled",
        "similarity_scc_alpha",
        "similarity_scc_min_weight",
        "soft_gate_temperature",
        "top_k_similar_old",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"similarity config is missing: {missing}")
    mode = str(config["similarity_relation_mode"])
    if mode not in VALID_RELATION_MODES:
        raise ValueError(f"invalid similarity_relation_mode: {mode}")
    active = mode not in {"none"}
    if active and mode != "none" and mode != "legacy_absolute":
        if not bool(config["old_new_separation"]):
            raise ValueError(f"{mode} requires old_new_separation=true")
    if mode == "legacy_absolute" and str(config["adaptive_margin_mode"]) != "drift":
        raise ValueError("B1 legacy_absolute must use AM-Drift")
    if bool(config["similarity_scc_enabled"]) and mode != "hard_relative":
        raise ValueError("similarity-aware SCC is defined only on top of S2 hard_relative")
    if not 0.0 <= float(config["similarity_alpha"]) <= 1.0:
        raise ValueError("similarity_alpha must be in [0,1]")
    if float(config["relative_margin"]) < 0:
        raise ValueError("relative_margin must be non-negative")
    if not 0.0 < float(config["similarity_beta_quantile"]) < 1.0:
        raise ValueError("similarity_beta_quantile must be in (0,1)")
    if str(config["similarity_beta_source"]) not in {"S0", "task_quantile", "fixed"}:
        raise ValueError("similarity_beta_source must be S0, task_quantile, or fixed")
    if str(config["similarity_beta_source"]) == "fixed":
        beta = config.get("similarity_beta_fixed")
        if beta is None or not -1.0 <= float(beta) <= 1.0:
            raise ValueError("fixed beta requires similarity_beta_fixed in [-1,1]")
    if not 0.0 <= float(config["similarity_scc_alpha"]) <= 1.0:
        raise ValueError("similarity_scc_alpha must be in [0,1]")
    if not 0.0 < float(config["similarity_scc_min_weight"]) <= 1.0:
        raise ValueError("similarity_scc_min_weight must be in (0,1]")
    if float(config["soft_gate_temperature"]) <= 0:
        raise ValueError("soft_gate_temperature must be positive")
    if int(config["top_k_similar_old"]) <= 0:
        raise ValueError("top_k_similar_old must be positive")
    if bool(config.get("multi_center", False)):
        raise ValueError("similarity-aware study requires multi_center=false")


def main():
    cli = parse_args()
    base_root = Path(cli.base_root).expanduser().resolve()
    e3_root = Path(cli.e3_root).expanduser().resolve()
    if not (base_root / "trainer.py").is_file():
        raise FileNotFoundError(f"invalid Structure root: {base_root}")
    if not (e3_root / "followup" / "learner.py").is_file():
        raise FileNotFoundError(f"invalid E3 extension root: {e3_root}")

    if cli.gpu:
        physical_gpus = [value.strip() for value in cli.gpu.split(",") if value.strip()]
        if not physical_gpus or any(not value.isdigit() for value in physical_gpus):
            raise ValueError("--gpu must contain non-negative integer IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
        logical_devices = list(range(len(physical_gpus)))
    elif cli.cpu or cli.validate_only:
        physical_gpus = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logical_devices = [-1]
    else:
        raise ValueError("choose --gpu or --cpu")

    # New extension first, then the E3 dependency, then the Structure base.
    sys.path.insert(0, str(base_root))
    sys.path.insert(0, str(e3_root))
    sys.path.insert(0, str(ROOT))
    from main import validate_config as validate_base_config
    from similarity.run_compat import validate_followup_compat

    seeds = parse_seeds(cli.seed)
    resolved = []
    for config_name in cli.config:
        config_path = Path(config_name).expanduser().resolve()
        config = load_config(config_path)
        validate_base_config(config)
        validate_followup_compat(config)
        validate_similarity_config(config)
        config.update(
            {
                "config": str(config_path),
                "base_root": str(base_root),
                # Parent E3 snapshot records this dependency independently.
                "extension_root": str(e3_root),
                "e3_root": str(e3_root),
                "similarity_extension_root": str(ROOT),
                "device": logical_devices,
                "physical_gpus": physical_gpus,
                "launch_command": " ".join(shlex.quote(value) for value in sys.argv),
            }
        )
        if cli.data_root:
            config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
        configured_output = Path(cli.output_root or config["output_root"]).expanduser()
        if not configured_output.is_absolute():
            configured_output = ROOT / configured_output
        config["output_root"] = str(configured_output.resolve())
        if seeds is not None:
            config["seed"] = seeds
        if cli.resume:
            config["resume"] = str(Path(cli.resume).expanduser().resolve())
        resolved.append(config)

    if cli.resume and (len(resolved) != 1 or len(resolved[0]["seed"]) != 1):
        raise ValueError("--resume accepts exactly one config and one seed")
    if cli.validate_only:
        for config in resolved:
            print(
                f"VALID {config['experiment_id']}: seeds={config['seed']} "
                f"mode={config['similarity_relation_mode']} "
                f"scc_sa={config['similarity_scc_enabled']} "
                f"beta={config['similarity_beta_source']}"
            )
        return

    import torch
    import trainer
    from similarity.exports import export_output_tree
    from similarity.learner import SimilarityAwareLearner

    original_factory = trainer.model_factory.get_model

    def get_similarity_model(model_name, args):
        if bool(args.get("similarity_aware_extension", False)):
            return SimilarityAwareLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_similarity_model
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
