#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT, configure_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen Structure representation/class-relation diagnostic."
    )
    parser.add_argument(
        "--protocol",
        action="append",
        choices=("IA5", "CAR5"),
        help="Repeat to select protocols. Default: IA5 then CAR5.",
    )
    parser.add_argument("--gpu", default="2", help="Physical GPU ID exposed to this process.")
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "representation_relation_diagnostic"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute a protocol even when a complete manifest already exists.",
    )
    return parser.parse_args()


def _complete(path: Path) -> bool:
    manifest = path / "run_manifest.json"
    if not manifest.is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    cli = parse_args()
    physical, visible = configure_device(cli.gpu, False)
    import torch

    from representation_relation.pipeline import RelationOptions, run_protocol
    from representation_relation.report import generate_combined_report, generate_protocol_report

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is required; requested physical GPU(s)={physical}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"this sequential diagnostic expects one visible GPU, got {torch.cuda.device_count()}"
        )
    protocols = cli.protocol or ["IA5", "CAR5"]
    output_root = Path(cli.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"physical GPU={physical[0]} -> process-local cuda:{visible[0]}; "
        f"protocol order={protocols}",
        flush=True,
    )
    for protocol in protocols:
        training_dir = (
            ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{cli.seed}"
        )
        checkpoint_dir = training_dir / "task_checkpoints"
        output_dir = output_root / f"{protocol}_T5"
        t4_checkpoint = checkpoint_dir / "task_04.pt"
        t5_checkpoint = checkpoint_dir / "task_05.pt"
        feature_cache = (
            ROOT
            / "classifier_bottleneck_diagnostic"
            / protocol
            / f"seed_{cli.seed}"
            / ".cache"
            / "task_05_features.npz"
        )
        missing = [path for path in (t4_checkpoint, t5_checkpoint, feature_cache) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{protocol} required input missing: " + ", ".join(map(str, missing))
            )
        if _complete(output_dir) and not cli.force:
            report = generate_protocol_report(output_dir)
            print(f"[skip complete] {protocol}: {report}", flush=True)
            continue
        print(f"[start] {protocol} T5 representation relation diagnostic", flush=True)
        manifest = run_protocol(
            RelationOptions(
                protocol=protocol,
                t4_checkpoint=t4_checkpoint,
                t5_checkpoint=t5_checkpoint,
                t5_feature_cache=feature_cache,
                output_dir=output_dir,
                data_root=DEFAULT_DATA_ROOTS[protocol],
                gpu_device=torch.device("cuda:0"),
                seed=cli.seed,
                batch_size=cli.batch_size,
                num_workers=cli.num_workers,
            )
        )
        report = generate_protocol_report(output_dir)
        print(
            f"[complete] {protocol}: elapsed={manifest['elapsed_seconds']:.1f}s report={report}",
            flush=True,
        )
    combined = generate_combined_report(output_root)
    print(f"[complete] combined report={combined}", flush=True)


if __name__ == "__main__":
    main()
