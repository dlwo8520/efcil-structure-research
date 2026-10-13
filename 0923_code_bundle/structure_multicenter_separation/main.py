"""CLI entry point with CUDA visibility configured before importing torch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Structure E0-E7 experiments")
    parser.add_argument("--config", nargs="+", required=True, help="one or more complete JSON configs")
    device = parser.add_mutually_exclusive_group(required=True)
    device.add_argument("--gpu", help="physical GPU IDs, for example 0 or 2,3")
    device.add_argument("--cpu", action="store_true", help="CPU mode (diagnostics only)")
    parser.add_argument("--data-root", help="dataset root containing train/ and test/")
    parser.add_argument("--output-root", help="override output_root from the JSON config")
    parser.add_argument("--seed", action="append", help="override seeds; repeat or use commas")
    parser.add_argument("--resume", help="checkpoint_latest.pt (one config and one seed only)")
    return parser.parse_args()


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_seeds(values):
    if not values:
        return None
    return [int(item) for value in values for item in value.split(",") if item.strip()]


def validate_config(config):
    required = {
        "experiment_id", "dataset", "seed", "init_cls", "increment", "expected_tasks",
        "multi_center", "num_prototypes", "old_new_separation", "separation_covariance",
        "sep_lambda", "sep_percentile", "sep_tau_floor", "sep_reduction", "sep_timing",
        "sep_boundary_policy", "sep_refresh_epochs", "replay_samples_per_class",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing required fields: {missing}")
    if config["sep_reduction"] != "max_relative_intrusion":
        raise ValueError("only max_relative_intrusion is implemented")
    if config["sep_timing"] != "epoch_apr_cache":
        raise ValueError("only epoch_apr_cache is implemented")
    if config["sep_boundary_policy"] != "birth_quantile_fixed":
        raise ValueError("only birth_quantile_fixed is implemented")
    if config["separation_covariance"] not in {"isotropic", "diagonal", "full"}:
        raise ValueError("invalid separation_covariance")
    if int(config["init_cls"]) != 40 or int(config["increment"]) != 40:
        raise ValueError("this ImageNet-A five-task protocol requires init_cls=increment=40")


def main():
    cli = parse_args()
    if cli.gpu:
        physical_gpus = [item.strip() for item in cli.gpu.split(",") if item.strip()]
        if not physical_gpus or any(not item.isdigit() for item in physical_gpus):
            raise ValueError("--gpu must contain non-negative integer IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical_gpus)
        logical_devices = list(range(len(physical_gpus)))
    else:
        physical_gpus = []
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logical_devices = [-1]

    import torch
    from trainer import CDS_train

    seeds = parse_seeds(cli.seed)
    if cli.resume and (len(cli.config) != 1 or (seeds is not None and len(seeds) != 1)):
        raise ValueError("--resume accepts exactly one config and one seed")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Visible physical GPUs: {physical_gpus if physical_gpus else 'CPU'}")
    for config_path in cli.config:
        config = load_config(config_path)
        validate_config(config)
        config["config"] = str(Path(config_path).resolve())
        config["device"] = logical_devices
        config["physical_gpus"] = physical_gpus
        if cli.data_root:
            config["data_root"] = str(Path(cli.data_root).expanduser().resolve())
        if cli.output_root:
            config["output_root"] = str(Path(cli.output_root).expanduser().resolve())
        if seeds is not None:
            config["seed"] = seeds
        if cli.resume:
            config["resume"] = str(Path(cli.resume).expanduser().resolve())
        print(f"--- Starting {config['experiment_id']}: {config_path} ---")
        CDS_train(config)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"--- Finished {config['experiment_id']} ---")


if __name__ == "__main__":
    main()
