from __future__ import annotations

import argparse
from pathlib import Path

from efcil_diag.experiment import METHODS, run_experiment
from efcil_diag.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EFCIL LoRA direction diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--resume", action="store_true", help="resume a task-boundary checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    run_experiment(
        load_config(args.config),
        method=args.method,
        repo_root=repo_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
