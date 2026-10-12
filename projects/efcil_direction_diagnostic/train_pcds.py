from __future__ import annotations

import argparse
from pathlib import Path

from efcil_diag.pcds_experiment import METHODS, run_experiment
from efcil_diag.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="PCDS on CIFAR-100 single-head EFCIL")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), default="pcds_full")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_experiment(
        load_config(args.config),
        method=args.method,
        repo_root=Path(__file__).resolve().parent,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

