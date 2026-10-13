#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", action="append", choices=("IA5", "CAR5"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--task", action="append", type=int)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root", default=str(ROOT / "classifier_bottleneck_diagnostic")
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    configure_device(cli.gpu, False)
    import torch

    from classifier_diagnostic.pipeline import DiagnosticOptions, run_protocol
    from classifier_diagnostic.report import generate_combined_report, generate_protocol_report

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocols = cli.protocol or ["IA5", "CAR5"]
    output_root = Path(cli.output_root).expanduser().resolve()
    for protocol in protocols:
        expected = 5
        tasks = ({expected} if cli.final_only else set(cli.task)) if (cli.final_only or cli.task) else None
        output_dir = output_root / protocol / f"seed_{cli.seed}"
        scientific_dir = ROOT / "scientific_audit" / protocol / f"seed_{cli.seed}"
        training_dir = ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{cli.seed}"
        manifest = run_protocol(
            DiagnosticOptions(
                protocol=protocol,
                gpu_device=torch.device("cuda:0"),
                output_dir=output_dir,
                scientific_dir=scientific_dir,
                training_dir=training_dir,
                data_root=Path(DEFAULT_DATA_ROOTS[protocol]),
                seed=cli.seed,
                tasks=tasks,
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
                force=cli.force,
            )
        )
        report = generate_protocol_report(output_dir)
        print(f"complete {protocol} tasks={manifest['completed_tasks']} report={report}")
    print(generate_combined_report(output_root))


if __name__ == "__main__":
    main()
