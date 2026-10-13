#!/usr/bin/env python3
"""Sequential two-run queue; only the new StoredAPR output tree is writable."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

from run import DATA_ROOTS, EXPERIMENT_ID, ROOT


def status(run_dir: Path) -> str:
    summary = run_dir / "summary.json"
    metadata = run_dir / "run_metadata.json"
    try:
        s = json.loads(summary.read_text(encoding="utf-8"))
        m = json.loads(metadata.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        s = m = {}
    if (
        s.get("status") == "complete"
        and int(s.get("completed_tasks", 0)) == 5
        and m.get("status") == "complete"
        and (run_dir / "checkpoint_latest.pt").is_file()
        and (run_dir / "stored_rp_birth_prototypes.pt").is_file()
    ):
        return "COMPLETE"
    if (run_dir / "checkpoint_latest.pt").is_file():
        return "RESUME"
    if run_dir.exists() and any(run_dir.iterdir()):
        return "PARTIAL_NO_CHECKPOINT"
    return "NEW"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="one physical GPU ID")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()
    if not cli.gpu.isdigit():
        raise ValueError("--gpu must be one physical GPU ID")
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "queue.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another StoredAPR queue is running") from error
        for index, protocol in enumerate(("IA5", "CAR5"), 1):
            run_dir = ROOT / "outputs" / protocol / EXPERIMENT_ID / "seed_1993"
            state = status(run_dir)
            label = f"{protocol}/{EXPERIMENT_ID}"
            if state == "COMPLETE":
                print(f"[{index}/2] SKIP complete {label}", flush=True)
                continue
            if state == "PARTIAL_NO_CHECKPOINT":
                raise RuntimeError(f"{label}: partial artifacts but no checkpoint; inspect before retry")
            command = [
                sys.executable, str(ROOT / "run.py"), "--protocol", protocol,
                "--gpu", cli.gpu, "--data-root", str(DATA_ROOTS[protocol]),
            ]
            if state == "RESUME":
                command.extend(("--resume", str(run_dir / "checkpoint_latest.pt")))
            print(f"[{index}/2] {state} {label}", flush=True)
            if cli.dry_run:
                print(" ".join(command), flush=True)
                continue
            with (logs / f"{protocol}__{EXPERIMENT_ID}__seed_1993.log").open(
                "a", encoding="utf-8"
            ) as handle:
                result = subprocess.run(
                    command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False,
                )
            if result.returncode:
                print(f"[{index}/2] FAIL exit={result.returncode} {label}", flush=True)
                raise SystemExit(result.returncode)
            print(f"[{index}/2] DONE {label}", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "analysis" / "build_report.py")],
                cwd=ROOT, check=False,
            )
        print(
            "STOREDAPR DRY RUN FINISHED" if cli.dry_run
            else "STOREDAPR QUEUE FINISHED failures=0",
            flush=True,
        )


if __name__ == "__main__":
    main()
