#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from oracle_audit.config import ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen T4→T5 same-sample and boundary-risk ranking diagnostics."
    )
    parser.add_argument("--protocol", action="append", choices=("IA5", "CAR5"))
    parser.add_argument("--gpu", default="4", help="Physical GPU ID.")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument(
        "--output-root", default=str(ROOT / "pair_boundary_risk_diagnostic")
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _complete(directory: Path) -> bool:
    path = directory / "run_manifest.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    cli = parse_args()
    physical, _ = configure_device(cli.gpu, False)
    import torch

    from pair_boundary_risk.pipeline import BoundaryRiskOptions, run_protocol
    from pair_boundary_risk.report import generate_combined_report, generate_protocol_report

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable; requested physical GPU(s)={physical}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("this sequential runner requires exactly one visible GPU")
    protocols = cli.protocol or ["IA5", "CAR5"]
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "pair_boundary_risk_diagnostic_plan.md",
        output_root / "pair_boundary_risk_diagnostic_plan.md",
    )
    print(
        f"physical GPU={physical[0]} -> process-local cuda:0; protocol order={protocols}",
        flush=True,
    )
    for protocol in protocols:
        checkpoint_dir = (
            ROOT / "training_outputs" / protocol / "Structure_AUDIT"
            / f"seed_{cli.seed}" / "task_checkpoints"
        )
        output_dir = output_root / f"{protocol}_T4_T5"
        options = BoundaryRiskOptions(
            protocol=protocol,
            t4_checkpoint=checkpoint_dir / "task_04.pt",
            t5_checkpoint=checkpoint_dir / "task_05.pt",
            t4_feature_cache=(
                ROOT / "representation_relation_diagnostic" / f"{protocol}_T5"
                / ".cache" / "t4_full_t5_class_features.npz"
            ),
            t5_feature_cache=(
                ROOT / "classifier_bottleneck_diagnostic" / protocol
                / f"seed_{cli.seed}" / ".cache" / "task_05_features.npz"
            ),
            output_dir=output_dir,
            device=torch.device("cuda:0"),
            seed=cli.seed,
        )
        required = (
            options.t4_checkpoint, options.t5_checkpoint,
            options.t4_feature_cache, options.t5_feature_cache,
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{protocol} missing required input: " + ", ".join(map(str, missing))
            )
        if _complete(output_dir) and not cli.force:
            report = generate_protocol_report(output_dir)
            print(f"[skip complete] {protocol}: {report}", flush=True)
            continue
        print(f"[start] {protocol} T4→T5 pair-boundary diagnostic", flush=True)
        manifest = run_protocol(options)
        report = generate_protocol_report(output_dir)
        print(
            f"[complete] {protocol}: elapsed={manifest['elapsed_seconds']:.1f}s report={report}",
            flush=True,
        )
    print(f"[complete] combined={generate_combined_report(output_root)}", flush=True)


if __name__ == "__main__":
    main()
