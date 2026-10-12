#!/usr/bin/env python3
"""Run E0-E7 sequentially while letting the caller choose physical GPU IDs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="physical GPU ID(s), e.g. 3 or 2,3")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--stage", choices=("pilot", "remaining", "all"), default="pilot")
    parser.add_argument("--experiments", default="E0,E1,E2,E3,E4,E5,E6,E7")
    parser.add_argument(
        "--resume-incomplete",
        action="store_true",
        help="resume checkpoint_latest.pt and skip runs whose summary status is complete",
    )
    args = parser.parse_args()
    experiments = [item.strip().upper() for item in args.experiments.split(",") if item.strip()]
    invalid = sorted(set(experiments) - {f"E{i}" for i in range(8)})
    if invalid:
        raise ValueError(f"unknown experiments: {invalid}")
    stages = [args.stage] if args.stage != "all" else ["pilot", "remaining"]
    for stage in stages:
        seeds = [1993] if stage == "pilot" else [1996, 1997]
        for experiment in experiments:
            for seed in seeds:
                run_dir = Path(args.output_root).resolve() / "imageneta" / experiment / f"seed_{seed}"
                summary_path = run_dir / "summary.json"
                if args.resume_incomplete and summary_path.is_file():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("status") == "complete":
                        print("SKIP complete", run_dir, flush=True)
                        continue
                command = [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "--config",
                    str(ROOT / "exps_jsons" / "imageneta_5task" / f"{experiment}.json"),
                    "--gpu",
                    args.gpu,
                    "--data-root",
                    args.data_root,
                    "--output-root",
                    args.output_root,
                    "--seed",
                    str(seed),
                ]
                checkpoint = run_dir / "checkpoint_latest.pt"
                if args.resume_incomplete and checkpoint.is_file():
                    command.extend(["--resume", str(checkpoint)])
                print("RUN", " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
