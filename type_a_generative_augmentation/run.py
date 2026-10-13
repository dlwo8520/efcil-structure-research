"""Staged Type A audit, generation feasibility, and matched acquisition pilot."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "prepare", "audit", "generate", "auto-sanity", "quality-check", "train", "report"), required=True)
    parser.add_argument("--gpu", type=int, help="physical GPU, required for audit/generate/train")
    args = parser.parse_args()
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be a nonnegative physical GPU number")
    if args.stage in ("audit", "generate", "train") and (args.gpu is None or args.gpu < 0):
        parser.error("this stage requires a nonnegative --gpu")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from type_a_generative_augmentation.common import (bootstrap, PROTOCOLS, ensure_config, checkpoint_path,
                                                       read_json, write_json, ROOT as output)
    bootstrap()
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ensure_config()
    if args.stage == "preflight":
        import importlib.util
        from type_a_generative_augmentation.common import DEFAULT_GENERATOR, AUDIT, source_config, data_manager
        from data.data import iCARS196_224, iImageNetA
        from data.data_manager import DummyDataset
        from type_a_generative_augmentation.learner import AugmentationLearner
        print("Structure data/learner imports PASS")
        train_cache_missing = 0
        for protocol in PROTOCOLS:
            source_config(protocol)
            manager = data_manager(protocol)
            print(f"{protocol} data/class-order PASS ({len(manager._class_order)} classes)")
            del manager
            for task in range(1, 11):
                if not checkpoint_path(protocol, task).is_file():
                    raise FileNotFoundError(checkpoint_path(protocol, task))
            for task in range(1, 10):
                path = AUDIT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / f"model_{task:02d}_on_task_{task:02d}.npz"
                import numpy as np
                with np.load(path, allow_pickle=False) as values:
                    if not {"test_features", "test_labels", "test_sample_ids"}.issubset(values.files):
                        raise RuntimeError(f"incomplete birth feature cache: {path}")
                    train_cache_missing += int(not {"train_features", "train_labels"}.issubset(values.files))
        print("Baseline/checkpoint/birth-cache preflight PASS")
        print("Birth current-train banks requiring frozen feature extraction:", train_cache_missing)
        print("Pinned generator:", DEFAULT_GENERATOR)
        print("Generator model_index exists:", (DEFAULT_GENERATOR / "model_index.json").is_file())
        print("Generation deps in current interpreter (wrapper checks .venv_generator separately):",
              {name: importlib.util.find_spec(name) is not None for name in ("diffusers", "transformers", "accelerate")})
        return
    if args.stage == "prepare":
        from type_a_generative_augmentation.generation import prepare_parents
        rows = prepare_parents()
        from type_a_generative_augmentation.parent_distribution import report_parent_distribution
        report_parent_distribution(rows)
        print(f"Fixed train-only parents: {len(rows)}")
        return
    if args.stage == "quality-check":
        from type_a_generative_augmentation.generation import quality_gate
        accepted = quality_gate()
        mode = "READY_EXPLORATORY (label fidelity NOT verified)" if (output / "exploratory_config.json").exists() else "Quality gates PASS"
        print(f"{mode}; common parent count: {len(accepted)}")
        return
    if args.stage == "report":
        from type_a_generative_augmentation.results import build_report
        build_report()
        return
    if args.stage == "auto-sanity" and args.gpu is None:
        from type_a_generative_augmentation.automatic_sanity import run_sanity
        torch.set_num_threads(4)
        run_sanity(torch.device("cpu"))
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"GPU {args.gpu} unavailable or not isolated")
    device = torch.device("cuda:0")
    # Initialize the allocator before any stage resets peak memory counters.
    torch.empty(1, device=device)
    manifest_path = output / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"stages": {}}
    manifest["stages"][args.stage] = {"status": "running", "physical_gpu": args.gpu}
    write_json(manifest_path, manifest)
    try:
        if args.stage == "audit":
            from type_a_generative_augmentation.audit import run_audit
            from type_a_generative_augmentation.common import read_csv, write_csv
            for protocol in PROTOCOLS:
                run_audit(protocol, device)
            write_csv(output / "birth_support_audit.csv", [row for protocol in PROTOCOLS for row in
                      read_csv(output / "audit" / protocol / "birth_support_audit.csv")])
        elif args.stage == "generate":
            from type_a_generative_augmentation.generation import generate
            generate(device)
        elif args.stage == "auto-sanity":
            from type_a_generative_augmentation.automatic_sanity import run_sanity
            run_sanity(device)
        elif args.stage == "train":
            if any(not (output / "audit" / p / "summary.json").exists() for p in PROTOCOLS):
                raise RuntimeError("complete birth audit before matched training; it does not select training parents")
            from type_a_generative_augmentation.training import run_training
            run_training(device)
            from type_a_generative_augmentation.results import build_report
            build_report()
        manifest["stages"][args.stage]["status"] = "complete"
    except Exception as error:
        manifest["stages"][args.stage].update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
