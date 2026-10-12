#!/usr/bin/env python3
"""Sequential training/audit queue with an exclusive lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from oracle_audit.config import PROTOCOLS, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--protocol", action="append", choices=PROTOCOLS)
    parser.add_argument("--phase", choices=("train", "audit", "all"), default="all")
    parser.add_argument("--covariance-mode", choices=("diagonal", "full", "both"), default="diagonal")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument(
        "--oracle-head-eval-tasks", choices=("final", "all", "none"), default="final"
    )
    return parser.parse_args()


def _complete_training(protocol: str) -> bool:
    expected = 5 if protocol.endswith("5") else 10
    run_dir = ROOT / "training_outputs" / protocol / "Structure_AUDIT" / "seed_1993"
    snapshots = sorted((run_dir / "task_checkpoints").glob("task_*.pt"))
    summary_path = run_dir / "summary.json"
    if len(snapshots) != expected or not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary.get("status") == "complete" and int(summary.get("completed_tasks", 0)) == expected


def _resume_checkpoint(protocol: str) -> Path | None:
    run_dir = ROOT / "training_outputs" / protocol / "Structure_AUDIT" / "seed_1993"
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "complete":
            raise RuntimeError(
                f"{protocol} says complete but lacks the full task snapshot series; "
                "preserving it without overwrite. Inspect the run manually."
            )
    checkpoint = run_dir / "checkpoint_latest.pt"
    return checkpoint if checkpoint.is_file() else None


def main() -> None:
    cli = parse_args()
    protocols = cli.protocol or list(PROTOCOLS)
    lock_path = ROOT / "logs" / "queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another oracle-audit queue holds {lock_path}") from error
        lock.write(str(os.getpid()))
        lock.flush()
        for protocol in protocols:
            if cli.phase in {"train", "all"}:
                if _complete_training(protocol):
                    print(f"SKIP complete training {protocol}", flush=True)
                else:
                    command = [
                        sys.executable, str(ROOT / "run_structure.py"),
                        "--protocol", protocol, "--gpu", cli.gpu,
                    ]
                    resume = _resume_checkpoint(protocol)
                    if resume is not None:
                        command.extend(["--resume", str(resume)])
                    print("RUN", " ".join(command), flush=True)
                    subprocess.run(command, cwd=ROOT, check=True)
            if cli.phase in {"audit", "all"}:
                command = [
                    sys.executable, str(ROOT / "run_audit.py"),
                    "--protocol", protocol, "--gpu", cli.gpu,
                    "--covariance-mode", cli.covariance_mode,
                    "--oracle-head-eval-tasks", cli.oracle_head_eval_tasks,
                ]
                if cli.skip_visualizations:
                    command.append("--skip-visualizations")
                print("RUN", " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
