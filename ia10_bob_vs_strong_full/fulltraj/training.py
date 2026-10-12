from __future__ import annotations

import gc
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from .common import (ROOT, SEED, digest, experiment_hash, manager, now, read_json,
                     source_config, spec, train_dir, write_csv, write_json, sha256)
from .learner import FullTrajectoryLearner, assert_matched_logs, state_hash
from .views import RealOnlyManager


def evaluate(learner, dm, branch, task):
    data, labels, dataset = dm.get_dataset(np.arange(task * 20), "test", "test", ret_data=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    ids = [f"test:{i:06d}:{hashlib.sha256(str(p).encode()).hexdigest()[:20]}" for i, p in enumerate(data)]
    scores_all, observed = [], []
    learner._network.eval()
    with torch.no_grad():
        for _, images, targets in loader:
            output = learner._network(images.to(learner._device))["logits"].detach().cpu()
            assert output.shape[1] == task * 20 and torch.isfinite(output).all()
            scores_all.append(output.numpy())
            observed.extend(targets.tolist())
    assert list(map(int, labels)) == observed and len(ids) == len(observed)
    scores = np.concatenate(scores_all)
    rows = []
    for i, (identity, y) in enumerate(zip(ids, observed)):
        birth = y // 20 + 1
        start = (birth - 1) * 20
        prediction = int(scores[i].argmax())
        local = start + int(scores[i, start:start + 20].argmax())
        rows.append({"branch": branch, "task": task, "sample_id": identity, "y": y,
                     "birth_task": birth, "pred": prediction, "local_pred": local,
                     "correct": int(prediction == y), "local_correct": int(local == y),
                     "evaluation_stage": "task_end_post_after_task"})
    target = train_dir(branch)
    write_csv(target / f"predictions_T{task:02d}.csv", rows)
    # Saved logits support future diagnostics without repeat training/inference.
    np.savez_compressed(target / f"logits_T{task:02d}.npz", scores=scores, labels=np.asarray(observed),
                        sample_ids=np.asarray(ids), candidate_classes=np.arange(task * 20), scale=1.0)
    return rows


def train_task(branch, task, device):
    from oracle_audit.runtime import torch_load
    from .bob_worker import verify_bank
    target = train_dir(branch)
    target.mkdir(parents=True, exist_ok=True)
    cp = target / "task_checkpoints" / f"task_{task:02d}.pt"
    completed = target / f"completed_T{task:02d}.json"
    identity = experiment_hash()
    if completed.exists():
        value = read_json(completed)
        assert value["experiment_hash"] == identity and cp.is_file()
        assert sha256(target / f"predictions_T{task:02d}.csv") == value["predictions_sha256"]
        print(f"Preserve completed {branch} T{task}", flush=True)
        return
    bank = verify_bank(task) if branch == "BOB_full" and not cp.exists() else []
    dm = RealOnlyManager(manager())
    config = source_config()
    config.update({"experiment_id": branch, "fulltraj_branch": branch,
                   "fulltraj_experiment_hash": identity, "device": [device],
                   "run_dir": str(target), "output_root": str(ROOT / "checkpoints"),
                   "num_workers": 0, "checkpoint_every_epoch": True})
    write_json(target / "resolved_config.json", {**config, "device": [str(device)]})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    learner = FullTrajectoryLearner(config, bank)
    initial_hash = state_hash(learner._network.state_dict())
    if task == 1:
        start_path = target / "start_manifest.json"
        if start_path.exists():
            assert read_json(start_path)["initial_network_sha256"] == initial_hash
        write_json(start_path, {"initial_network_sha256": initial_hash, "model_seed": SEED,
                               "source": "pretrained initialization, not vanilla task checkpoint",
                               "experiment_hash": identity})
        if branch == "BOB_full":
            assert read_json(train_dir("S_full") / "start_manifest.json")["initial_network_sha256"] == initial_hash
    resume = None
    if cp.exists():
        resume = cp  # Recover missing evaluation/completion files only.
    elif (target / "checkpoint_latest.pt").is_file():
        info = torch_load(target / "checkpoint_latest.pt")
        if info["stage"] == "train_epoch" and int(info["cur_task"]) + 1 == task:
            resume = target / "checkpoint_latest.pt"
        del info
    if resume is None and task > 1:
        resume = target / "task_checkpoints" / f"task_{task - 1:02d}.pt"
        if not resume.is_file():
            raise FileNotFoundError(f"cumulative trajectory needs its OWN previous task: {resume}")
    if resume:
        loaded = learner.load_experiment_checkpoint(resume)
        recovering = cp.exists()
        expected_task = task if recovering or loaded["stage"] == "train_epoch" else task - 1
        assert int(loaded["cur_task"]) + 1 == expected_task
        if not recovering and loaded["stage"] == "task_end":
            assert learner._known_classes == (task - 1) * 20
        del loaded
    else:
        recovering = False
        assert task == 1 and learner._cur_task == -1
    learner._network.to(device)
    if not recovering:
        # Same FC expansion and model RNG policy in both independent trajectories.
        torch.manual_seed(SEED + task * 10000)
        torch.cuda.manual_seed_all(SEED + task * 10000)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter()
        learner.incremental_train(dm)
        assert dm.generated_statistics_count == 0
        learner.release_synthetic_access()
        learner.after_task()
        rows = evaluate(learner, dm, branch, task)
        learner.save_experiment_checkpoint(target / "checkpoint_latest.pt", stage="task_end")
        write_json(target / f"resource_T{task:02d}.json", {
            "seconds_this_attempt": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "generated_statistics_count": 0, "synthetic_access_released_before_next_task": True,
            "source_checkpoint": str(resume) if resume else None,
            "source_is_own_branch": resume is None or target in resume.parents})
    else:
        evaluate(learner, dm, branch, task)
    if branch == "BOB_full":
        assert_matched_logs(read_json(train_dir("S_full") / f"matched_training_T{task:02d}.json"),
                            read_json(target / f"matched_training_T{task:02d}.json"))
    write_json(completed, {"branch": branch, "task": task, "completed_at": now(),
                           "experiment_hash": identity, "checkpoint_path": str(cp),
                           "predictions_sha256": sha256(target / f"predictions_T{task:02d}.csv"),
                           "generated_statistics_count": 0, "old_synthetic_access": False,
                           "matched_acquisition_checked": branch == "BOB_full",
                           "data_split_hash": dm.protocol_manifest["data_split_hash"]})
    del learner
    gc.collect()
    torch.cuda.empty_cache()


def gpu_smoke(device):
    """Real ViT adapter forward/backward + original checkpoint save/load, no eval."""
    from .common import read_csv
    from .views import FullViews
    target = ROOT / "feasibility/structure"
    marker = target / "complete.json"
    identity = experiment_hash()
    if marker.exists():
        assert read_json(marker)["experiment_hash"] == identity
        return
    bank = read_csv(ROOT / "feasibility/bob/smoke_bank.csv")
    class_id = int(bank[0]["class_id"])
    dm = manager()
    dataset = dm.get_dataset(np.asarray([class_id]), source="train", mode="train")
    from data.data_manager import DummyDataset
    dataset = DummyDataset(dataset.images[:2], dataset.labels[:2], dataset.trsf, use_path=True)
    logs = []
    for branch in ("S_full", "BOB_full"):
        folder = target / branch
        config = source_config()
        config.update({"device": [device], "run_dir": str(folder), "experiment_id": branch,
                       "fulltraj_branch": branch, "fulltraj_experiment_hash": identity,
                       "fulltraj_smoke": True, "batch_size": 2, "num_workers": 0})
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        learner = FullTrajectoryLearner(config, bank if branch == "BOB_full" else [])
        learner._network.update_fc(20)
        learner._network.to(device)
        learner._cur_task, learner._known_classes, learner._total_classes = 0, 0, 20
        learner.task_sizes = [20]
        learner.tuned_epochs = 1
        views = FullViews(dataset, branch, 1, bank)
        loader = torch.utils.data.DataLoader(views, batch_size=2, shuffle=True, generator=torch.Generator())
        optimizer = torch.optim.SGD(learner._network.parameters(), lr=.01, momentum=.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1)
        learner._init_train(loader, None, optimizer, scheduler)
        trained_hash = state_hash(learner._network.state_dict())
        del learner, optimizer, scheduler
        gc.collect(); torch.cuda.empty_cache()
        fresh = FullTrajectoryLearner(config, [])
        loaded = fresh.load_experiment_checkpoint(folder / "checkpoint_latest.pt")
        assert state_hash(fresh._network.state_dict()) == trained_hash
        assert loaded["stage"] == "train_epoch" and loaded["next_epoch"] == 1
        logs.append(read_json(folder / "matched_training_T01.json"))
        del loaded, fresh
        gc.collect(); torch.cuda.empty_cache()
    assert_matched_logs(*logs)
    write_json(marker, {"experiment_hash": identity, "actual_ViT_forward_backward": True,
                        "optimizer_updates_each_branch": 1, "checkpoint_roundtrip": "PASS",
                        "test_evaluation": False, "smoke_only_not_full_result": True})
