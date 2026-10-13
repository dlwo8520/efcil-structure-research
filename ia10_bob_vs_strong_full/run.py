from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from fulltraj.common import (ROOT, VENDOR, OFFICIAL_COMMIT, BRANCHES, bootstrap, configure_gpu,
                             digest, experiment_hash, now, progress, read_json, spec, task_dir,
                             train_dir, write_json)


def child(command, stage, task=None, branch=None):
    start = time.perf_counter()
    print("RUN", " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=ROOT, check=False)
    attempt = {"stage": stage, "task": task, "branch": branch, "seconds": time.perf_counter() - start,
               "exit_code": result.returncode, "recorded_at": now(),
               "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
    path = ROOT / "process_attempts.json"
    rows = read_json(path) if path.exists() else []
    write_json(path, rows + [attempt])
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def preflight(require_generator=False, generator_python=None):
    import hashlib
    import numpy as np
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VENDOR, text=True).strip()
    assert commit == OFFICIAL_COMMIT, "official repository commit changed"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=VENDOR, text=True).strip()
    from fulltraj.common import BASELINE, manager, read_csv, sha256
    dm = manager()
    reference = Path(spec()["reference_trajectory"])
    assert reference.is_file()
    assert len(list((BASELINE / "task_checkpoints").glob("task_*.pt"))) == 10
    expected = read_json(BASELINE / "resolved_config.json")
    assert expected["scc"] and expected["apr"] and expected["ca"]
    # Verify the saved vanilla trajectory against actual split identities and
    # labels, not only a filename or a number copied from an earlier report.
    paths, labels, _ = dm.get_dataset(np.arange(200), "test", "test", ret_data=True)
    ids = [f"test:{i:06d}:{hashlib.sha256(str(p).encode()).hexdigest()[:20]}" for i, p in enumerate(paths)]
    rows = read_csv(reference)
    by_id = {r["sample_id"]: r for r in rows}
    assert len(rows) == len(by_id) == len(ids) and by_id.keys() == set(ids)
    for identity, label in zip(ids, labels):
        row = by_id[identity]
        assert row["protocol"] == "IA10" and int(row["true_label"]) == int(label)
        birth = int(label) // 20 + 1
        assert int(row["birth_task"]) == birth
        for task in range(birth, 11):
            prediction = int(row[f"pred_T{task:02d}"])
            assert 0 <= prediction < task * 20
            assert int(row[f"correct_T{task:02d}"]) == int(prediction == int(label))
    vanilla_seen = [100 * sum(int(r[f"correct_T{t:02d}"]) for r in rows if int(r["birth_task"]) <= t)
                   / sum(int(r["birth_task"]) <= t for r in rows) for t in range(1, 11)]
    if require_generator:
        if not Path(generator_python).is_file():
            raise FileNotFoundError("BOB environment missing; first run bash setup_bob_env.sh")
        if not (ROOT / "assets/model_lock.json").is_file():
            raise FileNotFoundError("frozen caption model missing; first run bash setup_bob_env.sh")
        child([generator_python, "-m", "fulltraj.bob_worker", "logging-check"], "generator_logging_check")
        child([generator_python, "-m", "fulltraj.bob_worker", "doctor"], "generator_doctor")
    info = {"completed_at": now(), "status": "PASS", "experiment_hash": experiment_hash(),
            "official_commit": commit, "vanilla_reference_sha256": sha256(reference),
            "class_order_hash": dm.protocol_manifest["class_order_hash"],
            "data_split_hash": dm.protocol_manifest["data_split_hash"],
            "n_real_train": len(dm._train_targets), "n_test": len(dm._test_targets),
            "vanilla_reference_sample_identity_label_and_ten_task_coverage": "PASS",
            "vanilla_reference_stage": "task_end_post_after_task",
            "vanilla_seen_trajectory_pct": vanilla_seen,
            "vanilla_final_seen_pct": vanilla_seen[-1],
            "vanilla_average_incremental_pct": sum(vanilla_seen) / 10,
            "training_executed": False, "generator_executed": False,
            "estimated_lora_updates_per_task": [spec()["bob"]["finetune_epochs"] * ((n + 7) // 8) for n in dm.protocol_manifest["train_count_per_task"]],
            "estimated_candidates": 10 * 20 * spec()["bob"]["images_per_class"]}
    write_json(ROOT / "preflight.json", info)
    return info


def main():
    parser = argparse.ArgumentParser(description="IA10 two independent cumulative trajectories, no vanilla retraining")
    parser.add_argument("--gpu", help="physical GPU number; always mapped to logical cuda:0")
    parser.add_argument("--stage", choices=("full", "preflight", "smoke", "gpu-smoke", "prepare", "train", "analyze", "status"), default="full")
    parser.add_argument("--task", type=int, choices=range(1, 11))
    parser.add_argument("--branch", choices=BRANCHES)
    parser.add_argument("--generator-python", default=str(ROOT / ".bob_env/bin/python"))
    args = parser.parse_args()
    if args.stage == "status":
        for filename in ("progress.json", "run_manifest.json"):
            path = ROOT / filename
            print(path.name, read_json(path) if path.exists() else "not started")
        for branch in BRANCHES:
            completed = sorted(p.stem for p in train_dir(branch).glob("completed_T*.json"))
            print(branch, len(completed), "/10", completed)
        return
    if args.stage in ("full", "train", "gpu-smoke"):
        if args.gpu is None:
            parser.error("--gpu required; input a physical number at launch")
        configure_gpu(args.gpu)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    bootstrap()
    if args.stage == "smoke":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    if args.stage == "preflight":
        print(preflight(), flush=True)
        return
    if args.stage == "prepare":
        if args.task is None:
            parser.error("--task required")
        from fulltraj.inputs import prepare
        prepare(args.task)
        return
    if args.stage == "analyze":
        from fulltraj.results import analyze
        print("Full trajectory complete:", analyze(), flush=True)
        return
    if args.stage in ("train", "gpu-smoke"):
        if args.stage == "train" and (args.branch is None or args.task is None):
            parser.error("--branch and --task required")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("selected physical GPU is unavailable")
        torch.cuda.init()
        from fulltraj.training import train_task, gpu_smoke
        if args.stage == "gpu-smoke":
            gpu_smoke(torch.device("cuda:0"))
        else:
            train_task(args.branch, args.task, torch.device("cuda:0"))
        return
    # A single queue lock prevents accidental double runs without touching other jobs.
    with (ROOT / "queue.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another IA10 BOB queue is active; do not remove its lock") from error
        # Do not freeze an incomplete model identity when setup was forgotten.
        # Otherwise installing the model afterwards would make this empty run
        # appear to be a recipe change and prevent a legitimate first launch.
        if not Path(args.generator_python).is_file() or not (ROOT / "assets/model_lock.json").is_file():
            raise FileNotFoundError("first run bash setup_bob_env.sh; no experiment has been started")
        manifest_path = ROOT / "run_manifest.json"
        identity = experiment_hash()
        prior_manifest = read_json(manifest_path) if manifest_path.exists() else {}
        if manifest_path.exists():
            assert prior_manifest["experiment_hash"] == identity, "code/spec changed after queue started"
        write_json(manifest_path, {**prior_manifest, "status": "running", "updated_at": now(), "experiment_hash": identity,
                                  "physical_gpu": args.gpu, "branches": list(BRANCHES),
                                  "vanilla_retrained": False, "full_cumulative": True, "error": None})
        try:
            preflight(True, args.generator_python)
            child([sys.executable, ROOT / "run.py", "--stage", "smoke"], "CPU_smoke")
            child([sys.executable, ROOT / "run.py", "--stage", "prepare", "--task", 1], "prepare_feasibility", 1)
            child([args.generator_python, "-m", "fulltraj.bob_worker", "feasibility"], "BOB_GPU_smoke", 1)
            child([sys.executable, ROOT / "run.py", "--stage", "gpu-smoke", "--gpu", args.gpu], "Structure_GPU_smoke", 1)
            for branch in BRANCHES:
                for task in range(1, 11):
                    if not (train_dir(branch) / f"completed_T{task:02d}.json").exists() and branch == "BOB_full":
                        child([sys.executable, ROOT / "run.py", "--stage", "prepare", "--task", task], "prepare", task)
                        for stage in ("caption", "lora", "generate"):
                            child([args.generator_python, "-m", "fulltraj.bob_worker", stage, "--task", task], stage, task)
                    child([sys.executable, ROOT / "run.py", "--stage", "train", "--branch", branch,
                           "--task", task, "--gpu", args.gpu], "structure_training", task, branch)
                    # Analysis subprocess cannot advance the next training RNG.
                    child([sys.executable, ROOT / "run.py", "--stage", "analyze"], "analysis", task, branch)
            write_json(manifest_path, {**read_json(manifest_path), "status": "complete", "completed_at": now()})
            progress("complete", S_full_tasks=10, BOB_full_tasks=10)
        except BaseException as error:
            write_json(manifest_path, {**read_json(manifest_path), "status": "interrupted_or_failed",
                                      "updated_at": now(), "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
