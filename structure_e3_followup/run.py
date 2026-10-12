#!/usr/bin/env python3
"""Run follow-up methods while importing the base Structure tree read-only."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = ROOT.parent / "structure_multicenter_separation"


def parse_args():
    parser = argparse.ArgumentParser(description="Structure E3 follow-up experiments")
    parser.add_argument("--config", nargs="+", required=True, help="JSON config(s), with optional extends")
    parser.add_argument("--base-root", default=str(DEFAULT_BASE), help="read-only Structure repository")
    device = parser.add_mutually_exclusive_group(required=False)
    device.add_argument("--gpu", help="physical GPU IDs, for example 0 or 2,3")
    device.add_argument("--cpu", action="store_true", help="CPU mode (validation/small diagnostics)")
    parser.add_argument("--data-root", help="ImageNet-A root containing train/ and test/")
    parser.add_argument("--output-root", help="override output root")
    parser.add_argument("--seed", action="append", help="override seed(s); repeat or use commas")
    parser.add_argument("--resume", help="checkpoint_latest.pt (one config and seed only)")
    parser.add_argument("--validate-only", action="store_true", help="resolve and validate without training")
    return parser.parse_args()


def _merge(parent, child):
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, seen=None):
    config_path = Path(path).expanduser().resolve()
    seen = set() if seen is None else set(seen)
    if config_path in seen:
        raise ValueError(f"circular config inheritance: {config_path}")
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if "extends" not in config:
        return config
    parent_path = (config_path.parent / config["extends"]).resolve()
    return _merge(load_config(parent_path, seen), config)


def parse_seeds(values):
    if not values:
        return None
    return [int(item) for value in values for item in value.split(",") if item.strip()]


def validate_followup(config):
    required = {
        "pic_enabled",
        "ahn_enabled",
        "normalized_diagonal",
        "adaptive_margin_mode",
        "ceos_replay_fraction",
        "diagnostic_grad_interval",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"follow-up config is missing: {missing}")
    diag = bool(config["old_new_separation"]) and float(config["sep_lambda"]) > 0
    pic = bool(config["pic_enabled"])
    ahn = bool(config["ahn_enabled"])
    ceos = float(config["ceos_replay_fraction"])
    adaptive = str(config["adaptive_margin_mode"])
    if config["separation_covariance"] != "diagonal" and diag:
        raise ValueError("follow-up E3 experiments support diagonal covariance only")
    if pic and diag:
        raise ValueError("FCS PIC comparison must disable E3 diagonal separation")
    if ceos and (diag or pic or ahn):
        raise ValueError("CEOS isolation requires diagonal/PIC/AHN losses to be disabled")
    if not 0.0 <= ceos <= 1.0:
        raise ValueError("ceos_replay_fraction must be in [0,1]")
    if adaptive not in {"none", "variance", "age", "drift"}:
        raise ValueError("invalid adaptive_margin_mode")
    if adaptive != "none" and not diag:
        raise ValueError("adaptive E3 margin requires diagonal separation")
    if bool(config["normalized_diagonal"]) and not diag:
        raise ValueError("normalized_diagonal requires diagonal separation")
    if ahn and not diag:
        raise ValueError("the planned AHN comparison is E3 + AHN and requires E3")
    if int(config["diagnostic_grad_interval"]) < 0:
        raise ValueError("diagnostic_grad_interval must be non-negative")


def main():
    cli = parse_args()
    base_root = Path(cli.base_root).expanduser().resolve()
    if not (base_root / "trainer.py").is_file():
        raise FileNotFoundError(f"invalid Structure base root: {base_root}")
    if cli.gpu:
        physical_gpus = [item.strip() for item in cli.gpu.split(",") if item.strip()]
        if not physical_gpus or any(not item.isdigit() for item in physical_gpus):
            raise ValueError("--gpu must contain non-negative integer IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
        logical_devices = list(range(len(physical_gpus)))
    elif cli.cpu or cli.validate_only:
        physical_gpus = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logical_devices = [-1]
    else:
        raise ValueError("choose --gpu or --cpu (validation-only defaults to CPU)")

    sys.path.insert(0, str(base_root))
    sys.path.insert(0, str(ROOT))
    from main import validate_config as validate_base_config

    seeds = parse_seeds(cli.seed)
    resolved = []
    for config_name in cli.config:
        config_path = Path(config_name).expanduser().resolve()
        config = load_config(config_path)
        validate_base_config(config)
        validate_followup(config)
        config["config"] = str(config_path)
        config["base_root"] = str(base_root)
        config["extension_root"] = str(ROOT)
        config["device"] = logical_devices
        config["physical_gpus"] = physical_gpus
        config["launch_command"] = " ".join(shlex.quote(value) for value in sys.argv)
        if cli.data_root:
            config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
        if cli.output_root:
            config["output_root"] = str(Path(cli.output_root).expanduser().resolve())
        else:
            configured_output = Path(config["output_root"]).expanduser()
            if not configured_output.is_absolute():
                configured_output = ROOT / configured_output
            config["output_root"] = str(configured_output.resolve())
        if seeds is not None:
            config["seed"] = seeds
        if cli.resume:
            config["resume"] = str(Path(cli.resume).expanduser().resolve())
        resolved.append(config)
    if cli.resume and (
        len(resolved) != 1 or len(resolved[0]["seed"]) != 1
    ):
        raise ValueError("--resume accepts exactly one config and one seed")
    if cli.validate_only:
        for config in resolved:
            print(
                f"VALID {config['experiment_id']}: seeds={config['seed']} "
                f"diag={config['old_new_separation']} pic={config['pic_enabled']} "
                f"ahn={config['ahn_enabled']} ceos={config['ceos_replay_fraction']} "
                f"adaptive={config['adaptive_margin_mode']}"
            )
        return

    import torch
    import trainer
    from followup.learner import FollowupLearner

    original_factory = trainer.model_factory.get_model

    def get_followup_model(model_name, args):
        if bool(args.get("followup_extension", False)):
            return FollowupLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_followup_model
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible physical GPUs: {physical_gpus if physical_gpus else 'CPU'}")
    for config in resolved:
        print(f"--- Starting {config['experiment_id']}: {config['config']} ---")
        trainer.CDS_train(config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--- Finished {config['experiment_id']} ---")


if __name__ == "__main__":
    main()
