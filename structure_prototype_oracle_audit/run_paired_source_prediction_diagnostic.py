#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path

from oracle_audit.config import ROOT, configure_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched A/B/C paired source prediction diagnostic")
    parser.add_argument("--gpu", default="4", help="Physical GPU ID")
    parser.add_argument("--protocol", action="append", choices=("IA10", "CAR10", "IA5", "CAR5"))
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--output-root", type=Path, default=ROOT / "paired_source_prediction_diagnostic")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_repeats < 100:
        parser.error("--bootstrap-repeats must be >= 100")
    protocols = tuple(dict.fromkeys(args.protocol or ("IA10", "CAR10")))
    physical, visible = configure_device(args.gpu, False)
    import torch
    from paired_source_prediction.pipeline import Options, run

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA GPU: requested physical {physical}")
    output = args.output_root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "queue.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another paired source queue holds {output / 'queue.lock'}") from error
        print(f"physical GPU {physical[0]} -> process cuda:{visible[0]}; sequential {protocols}", flush=True)
        result = run(Options(
            output_root=output, device=torch.device("cuda:0"),
            physical_gpu=physical[0], protocols=protocols, seed=args.seed,
            bootstrap_repeats=args.bootstrap_repeats, force=args.force,
        ))
        print(f"[complete] {result['report']} ({result['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
