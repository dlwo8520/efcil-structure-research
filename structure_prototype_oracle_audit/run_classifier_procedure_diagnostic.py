#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from oracle_audit.config import ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", action="append", choices=("IA5", "CAR5"))
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--max-converged-epochs", type=int, default=100)
    parser.add_argument("--linear-max-iter", type=int, default=100)
    parser.add_argument(
        "--output-root", default=str(ROOT / "classifier_procedure_diagnostic")
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    configure_device(cli.gpu, False)
    import torch

    from classifier_procedure.pipeline import ProcedureOptions, copy_sampling_audit, run_protocol
    from classifier_procedure.report import generate_combined_report, generate_protocol_report

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    protocols = cli.protocol or ["IA5", "CAR5"]
    output_root = Path(cli.output_root).expanduser().resolve()
    copy_sampling_audit(ROOT / "ca_sampling_audit.md", output_root)
    for protocol in protocols:
        training_dir = ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{cli.seed}"
        output_dir = output_root / f"{protocol}_T5"
        print(f"[classifier procedure] {protocol} T5", flush=True)
        manifest = run_protocol(
            ProcedureOptions(
                protocol=protocol,
                checkpoint_path=training_dir / "task_checkpoints" / "task_05.pt",
                bottleneck_dir=ROOT / "classifier_bottleneck_diagnostic" / protocol / f"seed_{cli.seed}",
                scientific_dir=ROOT / "scientific_audit" / protocol / f"seed_{cli.seed}",
                output_dir=output_dir,
                gpu_device=torch.device("cuda:0"),
                seed=cli.seed,
                max_converged_epochs=cli.max_converged_epochs,
                linear_max_iter=cli.linear_max_iter,
            )
        )
        report = generate_protocol_report(output_dir)
        print(
            f"complete {protocol} elapsed={manifest['elapsed_seconds']:.1f}s report={report}",
            flush=True,
        )
    print(generate_combined_report(output_root), flush=True)


if __name__ == "__main__":
    main()

