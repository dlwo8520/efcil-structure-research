#!/usr/bin/env python3
"""Run C0--C4 sequentially; physical GPU 2 is the default."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORDER = ("C0", "C1", "C3", "C4", "C2")
DEFAULT_DATA = Path("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--gpu", default="2", help="physical GPU ID; default: 2")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--methods", nargs="+", choices=ORDER, default=list(ORDER))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def metadata(run_dir):
    path = run_dir / "run_metadata.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    cli = parse_args()
    if not str(cli.gpu).isdigit():
        raise ValueError("--gpu must be one non-negative physical GPU ID")
    seeds = (1993,) if cli.stage == "pilot" else (1993, 1996, 1997)
    data_root = Path(cli.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {data_root}")
    output_root = Path(cli.output_root).expanduser().resolve()
    selected = [method for method in ORDER if method in cli.methods]
    for seed in seeds:
        for method in selected:
            run_dir = output_root / "imageneta" / method / f"seed_{seed}"
            state = metadata(run_dir)
            if state.get("status") == "complete" and int(state.get("completed_tasks", 0)) == 5:
                print(f"SKIP complete {method} seed={seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--config",
                str(ROOT / "configs" / f"{method}.json"),
                "--gpu",
                str(cli.gpu),
                "--data-root",
                str(data_root),
                "--output-root",
                str(output_root),
                "--seed",
                str(seed),
            ]
            checkpoint = run_dir / "checkpoint_latest.pt"
            if state and checkpoint.is_file() and not cli.no_resume:
                command.extend(("--resume", str(checkpoint)))
            print("RUN", " ".join(command), flush=True)
            if not cli.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
