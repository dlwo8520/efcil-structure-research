#!/usr/bin/env python3
"""Run exactly the four fixed RP jobs, dynamically skipping completed work."""

from __future__ import annotations

import argparse
import csv
import fcntl
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rp.audit import DATA_ROOTS, RP_OUTPUT_ROOT, SELECTIVE_OUTPUT_ROOT, audit_rp


def parse_args():
    parser = argparse.ArgumentParser(description="Run the fixed RP-ProtoCLAD IA5/CAR5 queue")
    parser.add_argument("--gpu", required=True, help="one physical GPU ID (use 2 for this study)")
    parser.add_argument("--imageneta-root", default=str(DATA_ROOTS["IA5"]))
    parser.add_argument("--cars-root", default=str(DATA_ROOTS["CAR5"]))
    parser.add_argument("--output-root", default=str(RP_OUTPUT_ROOT))
    parser.add_argument("--selective-output-root", default=str(SELECTIVE_OUTPUT_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    cli = parse_args()
    if not cli.gpu.isdigit():
        raise ValueError("--gpu must be one non-negative physical GPU ID")
    data_roots = {
        "IA5": Path(cli.imageneta_root).expanduser().resolve(),
        "CAR5": Path(cli.cars_root).expanduser().resolve(),
    }
    for protocol, path in data_roots.items():
        if not path.is_dir():
            raise FileNotFoundError(f"{protocol} dataset root missing: {path}")
    output_root = Path(cli.output_root).expanduser().resolve()
    selective_output_root = Path(cli.selective_output_root).expanduser().resolve()
    log_root = ROOT / "logs" / "rp_task5"
    log_root.mkdir(parents=True, exist_ok=True)
    with (log_root / "queue.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another RP queue is already running") from error

        initial = audit_rp(output_root, selective_output_root)
        planned = [row for row in initial if row["action"] not in {"KEEP_COMPLETE", "SKIP_FAILED"}]
        print(f"RP TASK5 QUEUE jobs={len(planned)} gpu={cli.gpu} seed=1993", flush=True)
        failures = deferred = 0
        for index, planned_row in enumerate(planned, 1):
            current = next(
                row for row in audit_rp(output_root, selective_output_root)
                if row["protocol"] == planned_row["protocol"] and row["run_id"] == planned_row["run_id"]
            )
            label = f"{current['protocol']}/{current['run_id']}"
            if current["action"] == "KEEP_COMPLETE":
                print(f"[{index}/{len(planned)}] SKIP complete {label}", flush=True)
                continue
            if current["action"] == "WAIT_PARENT":
                print(f"[{index}/{len(planned)}] DEFER {label}: {current['reason']}", flush=True)
                deferred += 1
                continue
            if current["action"] == "SKIP_FAILED":
                print(f"[{index}/{len(planned)}] SKIP failed {label}: diagnose first", flush=True)
                failures += 1
                continue
            command = [
                sys.executable,
                str(ROOT / "run.py"),
                "--config", str(current["config"]),
                "--gpu", cli.gpu,
                "--data-root", str(data_roots[current["protocol"]]),
                "--output-root", str(output_root),
                "--seed", "1993",
            ]
            if current["action"] == "RESUME":
                command.extend(("--resume", current["resume_from"]))
            print(f"[{index}/{len(planned)}] {current['action']} {label}", flush=True)
            print(" ".join(command), flush=True)
            if cli.dry_run:
                continue
            log_path = log_root / f"{current['protocol']}__{current['run_id']}__seed_1993.log"
            started = datetime.now(timezone.utc).isoformat()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n===== LAUNCH {started} action={current['action']} =====\n")
                handle.flush()
                result = subprocess.run(
                    command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            ended = datetime.now(timezone.utc).isoformat()
            append_csv(
                log_root / "exit_codes.csv",
                {
                    "protocol": current["protocol"], "run_id": current["run_id"],
                    "seed": 1993, "action": current["action"],
                    "started_at_utc": started, "ended_at_utc": ended,
                    "exit_code": result.returncode, "log": str(log_path),
                    "output_dir": current["output_dir"],
                },
            )
            if result.returncode:
                failures += 1
                print(f"[{index}/{len(planned)}] FAIL exit={result.returncode} {label}", flush=True)
            else:
                print(f"[{index}/{len(planned)}] DONE {label}", flush=True)

        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_rp_plan.py")], cwd=ROOT, check=False)
        subprocess.run([sys.executable, str(ROOT / "analysis" / "build_report.py")], cwd=ROOT, check=False)
        print(f"RP TASK5 QUEUE FINISHED failures={failures} deferred={deferred}", flush=True)
        if not cli.dry_run and (failures or deferred):
            raise SystemExit(1 if failures else 2)


if __name__ == "__main__":
    main()

