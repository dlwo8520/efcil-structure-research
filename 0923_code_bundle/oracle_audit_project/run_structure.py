#!/usr/bin/env python3
"""Train exact vanilla Structure while retaining every task-end checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle_audit.config import (
    ROOT,
    bootstrap_paths,
    configure_device,
    git_commit,
    launch_command,
    protocol_config,
    tree_sha256,
    validate_protocol_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, choices=("IA5", "IA10", "CAR5", "CAR10"))
    parser.add_argument("--gpu", default="1", help="physical GPU ID; default: 1")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--data-root")
    parser.add_argument("--output-root", default=str(ROOT / "training_outputs"))
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=JSON",
        help="explicit smoke/testing override; production protocol values should not be changed",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    physical, logical = configure_device(cli.gpu, cli.cpu or cli.validate_only)
    bootstrap_paths()
    config = protocol_config(cli.protocol, cli.data_root)
    for assignment in cli.set:
        if "=" not in assignment:
            raise ValueError(f"--set expects KEY=JSON: {assignment}")
        key, raw = assignment.split("=", 1)
        config[key] = json.loads(raw)
    config.update(
        {
            "seed": [cli.seed],
            "device": logical,
            "physical_gpus": physical,
            "output_root": str(Path(cli.output_root).expanduser().resolve()),
            "oracle_audit_extension": True,
            "oracle_audit_root": str(ROOT),
            "source_snapshot_hash": tree_sha256(ROOT),
            "git_commit": git_commit(ROOT),
            "launch_command": launch_command(),
        }
    )
    if cli.resume:
        config["resume"] = str(Path(cli.resume).expanduser().resolve())
    validate_protocol_config(config)
    if cli.validate_only:
        print(
            f"VALID {config['protocol_id']} partition={config['task_partition']} "
            f"gpu={physical or 'CPU'} data={config['data_root']}"
        )
        return

    import torch
    import trainer
    from oracle_audit.snapshot_learner import AuditSnapshotLearner
    from selective.protocol import SweepDataManager

    trainer.DataManager = SweepDataManager
    original_factory = trainer.model_factory.get_model

    def get_model(model_name, args):
        if args.get("oracle_audit_extension"):
            return AuditSnapshotLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_model
    print(f"PyTorch={torch.__version__} CUDA={torch.cuda.is_available()} physical_gpu={physical}")
    print(f"--- vanilla Structure audit-ready training: {cli.protocol} ---", flush=True)
    trainer.CDS_train(config)
    print(f"--- finished: {cli.protocol} ---", flush=True)


if __name__ == "__main__":
    main()
