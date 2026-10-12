#!/usr/bin/env python3
"""Sequential, resumable Stage-1/Stage-2 queue runner."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective.registry import PROTOCOL_ORDER, method_registry


DEFAULT_IMAGENETA = Path("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a")
DEFAULT_CARS = Path("${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196")


def parse_args():
    parser = argparse.ArgumentParser(description="Run selective-similarity sweeps")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--gpu", default="1", help="physical GPU ID; default: 1")
    parser.add_argument("--imageneta-root", default=str(DEFAULT_IMAGENETA))
    parser.add_argument("--cars-root", default=str(DEFAULT_CARS))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOL_ORDER, default=list(PROTOCOL_ORDER))
    parser.add_argument("--run-ids", nargs="+", help="optional Stage-1 subset")
    parser.add_argument("--selection", default=str(ROOT / "stage2_selection.json"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def complete(run_dir, expected_tasks):
    summary = read_json(run_dir / "summary.json", {})
    metadata = read_json(run_dir / "run_metadata.json", {})
    return (
        summary.get("status") == "complete"
        and int(summary.get("completed_tasks", 0)) == expected_tasks
        and metadata.get("status") == "complete"
    )


def queue(cli):
    registry = method_registry()
    if cli.stage == "stage1":
        run_ids = list(registry)
        if cli.run_ids:
            unknown = sorted(set(cli.run_ids) - set(registry))
            if unknown:
                raise ValueError(f"unknown run IDs: {unknown}")
            run_ids = [run_id for run_id in run_ids if run_id in set(cli.run_ids)]
        for protocol in PROTOCOL_ORDER:
            if protocol not in cli.protocols:
                continue
            for run_id in run_ids:
                yield protocol, run_id, 1993
        return

    selection = read_json(cli.selection)
    if not selection:
        raise FileNotFoundError(
            f"Stage-2 selection missing: {cli.selection}; run scripts/select_stage2.py first"
        )
    for protocol in PROTOCOL_ORDER:
        if protocol not in cli.protocols:
            continue
        chosen = ["C0"]
        for family in ("A", "B", "C"):
            values = selection.get(protocol, {}).get(family, [])
            if len(values) != 2:
                raise ValueError(f"{protocol}/{family} needs exactly two frozen Stage-2 run IDs")
            chosen.extend(values)
        for seed in (1996, 1997):
            for run_id in chosen:
                if run_id not in registry:
                    raise ValueError(f"selection contains unknown run ID: {run_id}")
                yield protocol, run_id, seed


def append_exit_record(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    cli = parse_args()
    if not str(cli.gpu).isdigit():
        raise ValueError("--gpu must be one non-negative physical GPU ID")
    data_roots = {
        "IA5": Path(cli.imageneta_root).expanduser().resolve(),
        "IA10": Path(cli.imageneta_root).expanduser().resolve(),
        "CAR5": Path(cli.cars_root).expanduser().resolve(),
        "CAR10": Path(cli.cars_root).expanduser().resolve(),
    }
    for protocol in cli.protocols:
        if not data_roots[protocol].is_dir():
            raise FileNotFoundError(f"{protocol} dataset root missing: {data_roots[protocol]}")
    output_root = Path(cli.output_root).expanduser().resolve()
    log_root = ROOT / "logs" / cli.stage
    log_root.mkdir(parents=True, exist_ok=True)
    exit_path = log_root / "exit_codes.csv"
    failures = 0
    jobs = list(queue(cli))
    print(f"QUEUE stage={cli.stage} jobs={len(jobs)} gpu={cli.gpu}", flush=True)
    for index, (protocol, run_id, seed) in enumerate(jobs, start=1):
        expected_tasks = 5 if protocol.endswith("5") else 10
        run_dir = output_root / protocol / run_id / f"seed_{seed}"
        if complete(run_dir, expected_tasks):
            print(f"[{index}/{len(jobs)}] SKIP complete {protocol}/{run_id}/seed_{seed}", flush=True)
            continue
        command = [
            sys.executable, str(ROOT / "run.py"), "--protocol", protocol, "--run-id", run_id,
            "--gpu", str(cli.gpu), "--data-root", str(data_roots[protocol]),
            "--output-root", str(output_root), "--seed", str(seed),
        ]
        metadata = read_json(run_dir / "run_metadata.json", {})
        checkpoint = run_dir / "checkpoint_latest.pt"
        if metadata and checkpoint.is_file() and not cli.no_resume:
            command.extend(("--resume", str(checkpoint)))
        log_path = log_root / f"{protocol}__{run_id}__seed_{seed}.log"
        print(f"[{index}/{len(jobs)}] RUN {' '.join(command)}", flush=True)
        if cli.dry_run:
            continue
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n===== LAUNCH {started} =====\n")
            handle.flush()
            result = subprocess.run(
                command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        ended = datetime.now(timezone.utc).isoformat()
        append_exit_record(
            exit_path,
            {
                "stage": cli.stage, "protocol": protocol, "run_id": run_id, "seed": seed,
                "started_at_utc": started, "ended_at_utc": ended,
                "exit_code": result.returncode, "log": str(log_path), "output_dir": str(run_dir),
            },
        )
        if result.returncode:
            failures += 1
            print(f"[{index}/{len(jobs)}] FAIL exit={result.returncode}; continuing: {log_path}", flush=True)
        else:
            print(f"[{index}/{len(jobs)}] DONE {protocol}/{run_id}/seed_{seed}", flush=True)
    print(f"QUEUE FINISHED jobs={len(jobs)} failures={failures}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
