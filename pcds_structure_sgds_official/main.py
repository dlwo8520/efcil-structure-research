import json
import os
import argparse
from trainer import CDS_train
import gc
import torch

def run_single_experiment(config_path, raise_errors=False):
    """
    Run a single CDS training experiment with the given config file path.

    Args:
        config_path (str): Path to the experiment configuration JSON file.
    """
    print(f"--- Starting experiment: {config_path} ---")

    try:
        config_from_json = load_json(config_path)
        args = argparse.Namespace(config=config_path)
        merged_config = merge_configs(args, config_from_json)
        CDS_train(merged_config)

        print(f"--- Experiment finished: {config_path} ---\n")

    except FileNotFoundError:
        print(f"[Error] Config file not found: {config_path}, skipping.")
        if raise_errors:
            raise
    except Exception as e:
        print(f"[Error] Unknown error running {config_path}: {e}, skipping.")
        if raise_errors:
            raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def load_json(settings_path):
    """Load JSON file from the specified path."""
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def merge_configs(args, config):
    """Merge argparse arguments with config loaded from json."""
    merged_config = vars(args)
    merged_config.update(config)
    return merged_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Official PCDS / PCDS+SGDS runner")
    parser.add_argument(
        "--config",
        action="append",
        help="JSON config to run; repeat for sequential experiments",
    )
    cli = parser.parse_args()

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")

    experiment_files = cli.config or [
        './exps_jsons/adapter_car196.json',
        './exps_jsons/adapter_cifar224.json',
    ]

    for config_file in experiment_files:
        run_single_experiment(config_file, raise_errors=cli.config is not None)

    print("====== All experiments completed ======")
