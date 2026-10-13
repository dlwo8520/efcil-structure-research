#!/usr/bin/env python3
"""Run only unfinished selected IA5/CAR5 jobs from the reduced matrix."""

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
from selective.reduced import DEFAULT_DATA_ROOTS, DEFAULT_OUTPUT_ROOT, audit_all
from selective.registry import REDUCED_PROTOCOLS, REDUCED_RUN_IDS


def parse_args():
    parser = argparse.ArgumentParser(description="Run the reduced IA5/CAR5 Stage-1 queue")
    parser.add_argument("--gpu", required=True, help="one physical GPU ID")
    parser.add_argument("--imageneta-root", default=str(DEFAULT_DATA_ROOTS["IA5"]))
    parser.add_argument("--cars-root", default=str(DEFAULT_DATA_ROOTS["CAR5"]))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, choices=(1993,), default=1993)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def append_exit_record(path: Path, row: dict):
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
    output_root = Path(cli.output_root).expanduser().resolve()
    data_roots = {
        "IA5": Path(cli.imageneta_root).expanduser().resolve(),
        "IA10": Path(cli.imageneta_root).expanduser().resolve(),
        "CAR5": Path(cli.cars_root).expanduser().resolve(),
        "CAR10": Path(cli.cars_root).expanduser().resolve(),
    }
    for protocol in REDUCED_PROTOCOLS:
        if not data_roots[protocol].is_dir():
            raise FileNotFoundError(f"{protocol} dataset root missing: {data_roots[protocol]}")

    log_root = ROOT / "logs" / "reduced_stage1"
    log_root.mkdir(parents=True, exist_ok=True)
    lock_path = log_root / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another reduced queue holds {lock_path}") from error

        initial = audit_all(output_root, cli.seed, data_roots)
        selected = {
            (row.protocol, row.run_id): row
            for row in initial
            if row.protocol in REDUCED_PROTOCOLS and row.run_id in REDUCED_RUN_IDS
        }
        planned = [
            selected[(protocol, run_id)]
            for protocol in REDUCED_PROTOCOLS
            for run_id in REDUCED_RUN_IDS
            if selected[(protocol, run_id)].action in {"RUN_NEW", "RESUME"}
        ]
        print(
            f"REDUCED QUEUE jobs={len(planned)} gpu={cli.gpu} seed={cli.seed} "
            f"protocols={','.join(REDUCED_PROTOCOLS)}",
            flush=True,
        )
        failures = 0
        for index, planned_row in enumerate(planned, 1):
            current_rows = audit_all(output_root, cli.seed, data_roots)
            current = next(
                row for row in current_rows
                if row.protocol == planned_row.protocol and row.run_id == planned_row.run_id
            )
            if current.action == "KEEP_COMPLETE":
                print(
                    f"[{index}/{len(planned)}] SKIP newly complete "
                    f"{current.protocol}/{current.run_id}",
                    flush=True,
                )
                continue
            if current.action not in {"RUN_NEW", "RESUME"}:
                print(
                    f"[{index}/{len(planned)}] SKIP action={current.action} "
                    f"status={current.current_status} {current.protocol}/{current.run_id}",
                    flush=True,
                )
                continue
            command = [
                sys.executable, str(ROOT / "run.py"),
                "--protocol", current.protocol,
                "--run-id", current.run_id,
                "--gpu", cli.gpu,
                "--data-root", str(data_roots[current.protocol]),
                "--output-root", str(output_root),
                "--seed", str(cli.seed),
            ]
            if current.action == "RESUME":
                command.extend(("--resume", current.resume_from))
            print(
                f"[{index}/{len(planned)}] {current.action} "
                f"{current.protocol}/{current.run_id}",
                flush=True,
            )
            if cli.dry_run:
                print(" ".join(command), flush=True)
                continue
            log_path = log_root / f"{current.protocol}__{current.run_id}__seed_{cli.seed}.log"
            started = datetime.now(timezone.utc).isoformat()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n===== LAUNCH {started} action={current.action} =====\n")
                handle.flush()
                result = subprocess.run(
                    command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
                )
            ended = datetime.now(timezone.utc).isoformat()
            append_exit_record(
                log_root / "exit_codes.csv",
                {
                    "protocol": current.protocol,
                    "run_id": current.run_id,
                    "seed": cli.seed,
                    "action": current.action,
                    "started_at_utc": started,
                    "ended_at_utc": ended,
                    "exit_code": result.returncode,
                    "log": str(log_path),
                    "output_dir": current.output_dir,
                },
            )
            if result.returncode:
                failures += 1
                print(
                    f"[{index}/{len(planned)}] FAIL exit={result.returncode}; "
                    f"preserved and continuing: {log_path}",
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(planned)}] DONE {current.protocol}/{current.run_id}",
                    flush=True,
                )
        print(f"REDUCED QUEUE FINISHED failures={failures}", flush=True)
        raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

