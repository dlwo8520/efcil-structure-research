#!/usr/bin/env python3
"""Run one isolated StoredAPR RP-ProtoCLAD five-task experiment."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT_RP = ROOT.parent / "structure_rp_protoclad_task5"
ROOTS = {
    "base": ROOT.parent / "structure_multicenter_separation",
    "e3": ROOT.parent / "structure_e3_followup",
    "similarity": ROOT.parent / "structure_similarity_aware_followup",
    "selective": ROOT.parent / "structure_selective_similarity_sweep",
    "rp": PARENT_RP,
}
DATA_ROOTS = {
    "IA5": Path("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a"),
    "CAR5": Path("${RESEARCH_WORKSPACE}/cds-main/data/cars196"),
}
EXPERIMENT_ID = "StoredAPR_RP_C_P020_L030"


def merge(parent, child):
    result = dict(parent)
    for key, value in child.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path, seen=None):
    path = Path(path).resolve()
    seen = set() if seen is None else set(seen)
    if path in seen:
        raise ValueError(f"circular config inheritance: {path}")
    seen.add(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if "extends" not in config:
        return config
    return merge(load_config(path.parent / config["extends"], seen), config)


def validate_variant(config, parent, protocol, checksum):
    allowed = {
        "prefix", "experiment_id", "method_name", "stored_apr_rp_enabled",
        "stored_apr_rp_definition",
    }
    changes = {
        key: (parent.get(key), config.get(key))
        for key in set(parent) | set(config)
        if key not in allowed and parent.get(key) != config.get(key)
    }
    if changes:
        raise ValueError(f"StoredAPR differs from old RP outside its variant identity: {changes}")
    expected = {
        "protocol_id": protocol,
        "dataset": protocol,
        "experiment_id": EXPERIMENT_ID,
        "parent_run_id": "C_P020_L030",
        "expected_tasks": 5,
        "seed": [1993],
        "top_p": 0.2,
        "lambda_clad": 0.3,
        "sep_lambda": 0.3,
        "rp_input_dim": 768,
        "rp_dim": 2048,
        "rp_seed": 1993,
        "rp_activation": "relu",
        "rp_trainable": False,
        "rp_bias": False,
        "rp_enabled": True,
        "stored_apr_rp_enabled": True,
        "rp_matrix_checksum": checksum,
    }
    mismatches = {key: (value, config.get(key)) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"invalid fixed StoredAPR RP config: {mismatches}")
    if config["method"] != "C" or config["selective_method"] != "rp_proto_clad":
        raise ValueError("old RP method and Top-P selection must remain unchanged")
    from selective.protocol import validate_partition
    from similarity.run_compat import validate_followup_compat
    validate_partition(protocol, config["task_partition"], config["expected_num_classes"])
    validate_followup_compat(config)


def main():
    parser = argparse.ArgumentParser(description="StoredAPR RP-ProtoCLAD (IA5/CAR5 only)")
    parser.add_argument("--protocol", choices=("IA5", "CAR5"), required=True)
    parser.add_argument("--gpu", required=True, help="one physical GPU ID")
    parser.add_argument("--data-root")
    parser.add_argument("--resume")
    parser.add_argument("--validate-only", action="store_true")
    cli = parser.parse_args()
    if not cli.gpu.isdigit():
        raise ValueError("--gpu must be one physical GPU ID")
    for name, root in ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(f"missing {name} dependency: {root}")
        sys.path.insert(0, str(root))
    sys.path.insert(0, str(ROOT))
    for name, root, marker in (
        ("base", ROOTS["base"], "trainer.py"),
        ("e3", ROOTS["e3"], "followup/learner.py"),
        ("similarity", ROOTS["similarity"], "similarity/learner.py"),
        ("selective", ROOTS["selective"], "selective/learner.py"),
        ("rp", ROOTS["rp"], "rp/learner.py"),
    ):
        if not (root / marker).is_file():
            raise FileNotFoundError(f"invalid {name} root: {root}")
    if not cli.validate_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from rp.projection import make_projection, projection_metadata
    from stored_apr_rp.snapshot import tree_digest

    config_path = ROOT / "configs" / f"{cli.protocol}.json"
    parent_path = PARENT_RP / "configs" / f"RP_C_P020_L030_{cli.protocol}.json"
    config, parent = load_config(config_path), load_config(parent_path)
    matrix = make_projection(768, 2048, 1993)
    rp_metadata = projection_metadata(matrix, 1993)
    config["rp_matrix_shape"] = rp_metadata["shape"]
    config["rp_matrix_checksum"] = rp_metadata["checksum_sha256"]
    config["rp_projection"] = rp_metadata
    parent["rp_matrix_shape"] = rp_metadata["shape"]
    parent["rp_matrix_checksum"] = rp_metadata["checksum_sha256"]
    parent["rp_projection"] = rp_metadata
    validate_variant(config, parent, cli.protocol, rp_metadata["checksum_sha256"])
    data_root = Path(cli.data_root or DATA_ROOTS[cli.protocol]).resolve()
    if not cli.validate_only and not data_root.is_dir():
        raise FileNotFoundError(f"dataset root missing: {data_root}")
    output_root = ROOT / "outputs"
    run_dir = output_root / cli.protocol / EXPERIMENT_ID / "seed_1993"
    if cli.resume:
        checkpoint = Path(cli.resume).resolve()
        if checkpoint != run_dir / "checkpoint_latest.pt" or not checkpoint.is_file():
            raise ValueError("--resume must be this variant's existing checkpoint_latest.pt")
        config["resume"] = str(checkpoint)
    elif run_dir.exists() and any(run_dir.iterdir()) and not cli.validate_only:
        raise RuntimeError("existing run artifacts found; inspect and pass --resume rather than overwriting")

    source_hash, source_count = tree_digest(ROOT)
    config.update({
        "config": str(config_path),
        "base_root": str(ROOTS["base"]),
        "extension_root": str(ROOTS["e3"]),
        "e3_root": str(ROOTS["e3"]),
        "similarity_extension_root": str(ROOTS["similarity"]),
        "selective_sweep_root": str(ROOTS["selective"]),
        "rp_extension_root": str(ROOTS["rp"]),
        "stored_apr_rp_root": str(ROOT),
        "selective_sweep_extension": True,
        "rp_protoclad_extension": True,
        "device": [-1] if cli.validate_only else [0],
        "physical_gpus": [] if cli.validate_only else [cli.gpu],
        "data_root": str(data_root),
        "output_root": str(output_root),
        "launch_command": " ".join(shlex.quote(value) for value in sys.argv),
        "source_snapshot_hash": source_hash,
        "source_file_count": source_count,
    })
    if cli.validate_only:
        print(f"VALID {cli.protocol}/{EXPERIMENT_ID} R={rp_metadata['checksum_sha256']}")
        return

    import torch
    import trainer
    from rp.diagnostics import export_rp_run_diagnostics
    from selective.diagnostics import export_run_diagnostics
    from selective.protocol import SweepDataManager
    from stored_apr_rp.learner import StoredAPRRPProtoCLADLearner

    trainer.DataManager = SweepDataManager
    original_factory = trainer.model_factory.get_model

    def get_model(model_name, args):
        if args.get("stored_apr_rp_enabled", False):
            return StoredAPRRPProtoCLADLearner(args)
        return original_factory(model_name, args)

    trainer.model_factory.get_model = get_model
    print(f"Visible physical GPU: {cli.gpu}", flush=True)
    print(f"--- Starting {cli.protocol}/{EXPERIMENT_ID} ---", flush=True)
    trainer.CDS_train(config)
    per_seed = dict(config, seed=1993)
    export_run_diagnostics(run_dir, per_seed)
    export_rp_run_diagnostics(run_dir, per_seed)
    try:
        from analysis.build_report import build_report
        build_report()
    except Exception as error:
        print(f"WARNING: report refresh failed without invalidating training: {error}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"--- Finished {cli.protocol}/{EXPERIMENT_ID} ---", flush=True)


if __name__ == "__main__":
    main()
