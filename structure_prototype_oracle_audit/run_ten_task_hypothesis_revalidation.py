#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from oracle_audit.config import ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential frozen-checkpoint 5-task versus 10-task hypothesis revalidation."
    )
    parser.add_argument("--gpu", default="4", help="Physical GPU ID exposed to the process.")
    parser.add_argument("--protocol", action="append", choices=("IA10", "CAR10"))
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--phases", default="1,2,3", help="Comma-separated subset of 1,2,3.")
    parser.add_argument(
        "--output-root", default=str(ROOT / "ten_task_hypothesis_revalidation")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _complete(path: Path, phases: tuple[int, ...], protocols: tuple[str, ...]) -> bool:
    manifest = path / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "complete"
        and set(phases).issubset(set(value.get("phases", [])))
        and set(protocols).issubset(set(value.get("protocols_executed", [])))
    )


def main() -> None:
    cli = parse_args()
    phases = tuple(sorted({int(item.strip()) for item in cli.phases.split(",") if item.strip()}))
    if not phases or any(phase not in {1, 2, 3} for phase in phases):
        raise ValueError("--phases must be a comma-separated subset of 1,2,3")
    protocols = tuple(cli.protocol or ("IA10", "CAR10"))
    physical, visible = configure_device(cli.gpu, False)
    import torch

    from ten_task_revalidation.pipeline import RevalidationOptions, run_revalidation

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable; requested physical GPU={physical}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"sequential runner requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another 10-task revalidation queue holds {lock_path}") from error
        if _complete(output_root, phases, protocols) and not cli.force:
            print(f"[skip complete] {output_root / 'ten_task_hypothesis_revalidation_final_report.md'}", flush=True)
            return
        print(
            f"physical GPU={physical[0]} -> process-local cuda:{visible[0]}; "
            f"sequential protocol order={list(protocols)}; phases={list(phases)}",
            flush=True,
        )
        result = run_revalidation(
            RevalidationOptions(
                output_root=output_root,
                device=torch.device("cuda:0"),
                physical_gpu=physical[0],
                seed=cli.seed,
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
                phases=phases,
                protocols=protocols,
                random_trials=cli.random_trials,
                force=cli.force,
            )
        )
        print(
            f"[complete] elapsed={result['elapsed_seconds']:.1f}s "
            f"report={result['reports']['final']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
