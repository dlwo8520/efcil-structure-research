"""Sequential T8 -> T9 -> T10 two-view Control/JS pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT.parent / "structure_prototype_oracle_audit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def run_one(protocol: str, branch: str, gpu: str, seed: int):
    from oracle_audit.config import DEFAULT_DATA_ROOTS, protocol_config
    from oracle_audit.runtime import torch_load
    from oracle_audit.snapshot_learner import AuditSnapshotLearner
    from selective.protocol import SweepDataManager
    from pilot_learner import VariationPilotLearner
    import trainer
    from utils import model_factory

    source = AUDIT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{seed}" / "task_checkpoints" / "task_08.pt"
    if not source.is_file():
        raise FileNotFoundError(f"T8 checkpoint required: {source}")
    run_dir = ROOT / protocol / branch / f"seed_{seed}"
    final = run_dir / "task_checkpoints" / "task_10.pt"
    if final.is_file():
        print(f"[{protocol} {branch}] already complete; preserving {final}", flush=True)
        return
    resume = run_dir / "task_checkpoints" / "task_09.pt"
    if not resume.is_file():
        resume = source
    checkpoint = torch_load(source)
    if checkpoint["stage"] != "task_end" or int(checkpoint["cur_task"]) != 7:
        raise RuntimeError("source must be end-of-T8 checkpoint")
    config = protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])
    for key in ("scc", "apr", "ca"):
        if not config.get(key):
            raise RuntimeError(f"pilot requires {key}")
    if config["old_new_separation"] or int(config["expected_tasks"]) != 10:
        raise RuntimeError("pilot requires vanilla 10-task Structure")
    config.update({
        "experiment_id": branch, "method_name": f"Structure two-view {branch}",
        "pilot_branch": branch, "seed": [seed], "device": [0],
        "physical_gpus": [gpu], "output_root": str(ROOT),
        "resume": str(resume), "oracle_audit_extension": True,
        "oracle_audit_root": str(AUDIT), "num_workers": 8,
        "checkpoint_every_epoch": False,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": protocol, "branch": branch, "seed": seed,
        "start_task": 8, "trained_tasks": [9, 10],
        "source_checkpoint": str(source), "source_checkpoint_sha256": _sha256(source),
        "resume_checkpoint": str(resume), "source_git_commit": _git_commit(AUDIT),
        "base_git_commit": _git_commit(ROOT.parent / "structure_multicenter_separation"),
        "lambda_js": .1 if branch == "js" else 0.0, "temperature": 1.0,
        "consistency_candidate_set": "current-only",
        "augmentation": "original Structure train transform applied independently twice",
        "optimizer": config["optimizer"], "init_lr": config["init_lr"],
        "inc_epochs": config["inc_epochs"], "batch_size": config["batch_size"],
        "ca_epochs": config["ca_epochs"],
        "ca_exposure_per_class": config["replay_samples_per_class"],
        "scc": config["scc"], "apr": config["apr"], "ca": config["ca"],
        "physical_gpu": gpu,
    }
    (run_dir / "pilot_start_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    trainer.DataManager = SweepDataManager
    original_factory = model_factory.get_model

    def factory(model_name, args):
        return VariationPilotLearner(args) if args.get("pilot_branch") in {"control", "js"} else original_factory(model_name, args)

    model_factory.get_model = factory
    try:
        trainer.CDS_train(config)
    finally:
        model_factory.get_model = original_factory
    if not final.is_file():
        raise RuntimeError(f"pilot finished without final task snapshot: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True, help="physical GPU index")
    parser.add_argument("--protocol", choices=("IA10", "CAR10", "all"), default="all")
    parser.add_argument("--branch", choices=("control", "js", "all"), default="all")
    parser.add_argument("--seed", type=int, default=1993)
    args = parser.parse_args()
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != args.gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES conflicts with --gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import sys
    sys.path.insert(0, str(AUDIT))
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()
    sys.path.insert(0, str(ROOT))
    for protocol in (("IA10", "CAR10") if args.protocol == "all" else (args.protocol,)):
        for branch in (("control", "js") if args.branch == "all" else (args.branch,)):
            print(f"=== {protocol} {branch} GPU {args.gpu} ===", flush=True)
            run_one(protocol, branch, args.gpu, args.seed)


if __name__ == "__main__":
    main()
