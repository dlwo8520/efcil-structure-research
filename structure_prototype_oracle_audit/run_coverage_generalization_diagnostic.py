#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from oracle_audit.config import ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Coverage-source and pretrained-vs-final frozen representation diagnostic."
    )
    parser.add_argument("--gpu", default="4", help="Physical GPU ID exposed to this process.")
    parser.add_argument("--protocol", action="append", choices=("IA10", "CAR10", "IA5", "CAR5"))
    parser.add_argument("--include-five-task", action="store_true")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--gaussian-draws", type=int, default=3)
    parser.add_argument("--support-k", type=int, default=5)
    parser.add_argument("--bank-cap", type=int, default=0, help="0 keeps every unique actual reference.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--support-chunk-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-lrs", default="0.01,0.03,0.1")
    parser.add_argument(
        "--output-root", default=str(ROOT / "coverage_generalization_diagnostic")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _complete(root: Path, protocols: tuple[str, ...]) -> bool:
    manifest = root / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("status") == "complete"
        and set(protocols).issubset(set(value.get("protocols", [])))
        and (root / "coverage_generalization_final_report.md").is_file()
    )


def main() -> None:
    cli = parse_args()
    if cli.gaussian_draws < 3:
        raise ValueError("--gaussian-draws must be at least 3")
    if cli.support_k <= 0:
        raise ValueError("--support-k must be positive")
    if cli.bank_cap < 0:
        raise ValueError("--bank-cap must be non-negative")
    learning_rates = tuple(float(value.strip()) for value in cli.probe_lrs.split(",") if value.strip())
    if not learning_rates or any(value <= 0 for value in learning_rates):
        raise ValueError("--probe-lrs must contain positive comma-separated values")
    protocols = list(cli.protocol or ("IA10", "CAR10"))
    if cli.include_five_task:
        protocols.extend(("IA5", "CAR5"))
    protocols = tuple(dict.fromkeys(protocols))
    physical, visible = configure_device(cli.gpu, False)
    import torch

    from coverage_generalization.pipeline import CoverageOptions, run_diagnostic

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
            raise RuntimeError(f"another coverage diagnostic holds {lock_path}") from error
        if _complete(output_root, protocols) and not cli.force:
            print(f"[skip complete] {output_root / 'coverage_generalization_final_report.md'}", flush=True)
            return
        print(
            f"physical GPU={physical[0]} -> process-local cuda:{visible[0]}; "
            f"sequential protocol order={list(protocols)}",
            flush=True,
        )
        result = run_diagnostic(
            CoverageOptions(
                output_root=output_root,
                device=torch.device("cuda:0"),
                physical_gpu=physical[0],
                protocols=protocols,
                seed=cli.seed,
                gaussian_draws=cli.gaussian_draws,
                support_k=cli.support_k,
                bank_cap=None if cli.bank_cap == 0 else cli.bank_cap,
                batch_size=cli.batch_size,
                support_chunk_size=cli.support_chunk_size,
                num_workers=cli.num_workers,
                probe_epochs=cli.probe_epochs,
                probe_batch_size=cli.probe_batch_size,
                probe_lrs=learning_rates,
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
