#!/usr/bin/env python3
"""Read-only post-hoc oracle audit for one vanilla Structure protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=("IA5", "IA10", "CAR5", "CAR10"))
    parser.add_argument("--gpu", default="1", help="physical GPU ID; default: 1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--task", action="append", type=int)
    parser.add_argument("--splits", default="train,test", help="train,test or one split")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--covariance-mode", choices=("diagonal", "full", "both"), default="diagonal")
    parser.add_argument("--max-classes", type=int, help="smoke only")
    parser.add_argument("--max-samples-per-class", type=int, help="smoke only")
    parser.add_argument("--visualization-samples-per-class", type=int, default=50)
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument(
        "--oracle-head-eval-tasks",
        choices=("final", "all", "none"),
        default="final",
        help="run the post-hoc A-D classifier replacement on the final task (default), all tasks, or none",
    )
    parser.add_argument("--oracle-head-epochs", type=int, help="sanity/debug override; production uses checkpoint CA epochs")
    parser.add_argument(
        "--oracle-head-samples-per-class",
        type=int,
        help="sanity/debug override; production uses checkpoint replay sample count",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    physical, logical = configure_device(cli.gpu, cli.cpu)
    import torch
    from oracle_audit.pipeline import AuditOptions, audit_protocol
    from oracle_audit.report import generate_combined_report, generate_protocol_report

    protocol = cli.protocol.upper()
    run_dir = Path(cli.run_dir).expanduser().resolve() if cli.run_dir else (
        ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{cli.seed}"
    )
    output_root = Path(cli.output_root).expanduser().resolve()
    output_dir = output_root / protocol / f"seed_{cli.seed}"
    splits = tuple(item.strip() for item in cli.splits.split(",") if item.strip())
    if not splits or any(item not in {"train", "test"} for item in splits):
        raise ValueError("--splits must be train,test or one of them")
    device = torch.device("cpu" if cli.cpu else "cuda:0")
    if not cli.cpu and not torch.cuda.is_available():
        raise RuntimeError("GPU audit requested but CUDA is unavailable")
    print(f"audit protocol={protocol} physical_gpu={physical or 'CPU'} run_dir={run_dir}")
    manifest = audit_protocol(
        AuditOptions(
            protocol=protocol,
            run_dir=run_dir,
            output_dir=output_dir,
            data_root=Path(cli.data_root or DEFAULT_DATA_ROOTS[protocol]).expanduser().resolve(),
            device=device,
            explicit_checkpoints=cli.checkpoint,
            splits=splits,
            batch_size=cli.batch_size,
            num_workers=cli.num_workers,
            covariance_mode=cli.covariance_mode,
            sample_cap=cli.max_samples_per_class,
            class_cap=cli.max_classes,
            task_filter=set(cli.task) if cli.task else None,
            visualization_samples_per_class=cli.visualization_samples_per_class,
            seed=cli.seed,
            skip_visualizations=cli.skip_visualizations,
            oracle_head_eval_tasks=cli.oracle_head_eval_tasks,
            oracle_head_epochs=cli.oracle_head_epochs,
            oracle_head_samples_per_class=cli.oracle_head_samples_per_class,
        )
    )
    report = generate_protocol_report(output_dir)
    combined = generate_combined_report(output_root)
    print(f"audit complete tasks={manifest['audited_tasks']} report={report} combined={combined}")


if __name__ == "__main__":
    main()
