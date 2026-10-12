from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .common import (ROOT, PROTOCOLS, SEED, checkpoint_path, source_config, data_manager,
                     run_dir, sample_identity, write_json, write_csv, read_csv, sha256)
from .generation import quality_gate
from .learner import AugmentationLearner, state_hash
from utils.experiment_io import capture_rng_state, restore_rng_state


def evaluate_task(learner, manager, branch, target):
    task = learner._cur_task + 1
    ids, expected = sample_identity(manager, np.arange(learner._total_classes), "test")
    rows = []
    known = sum(learner.task_sizes[:-1])
    birth = np.repeat(np.arange(1, len(learner.task_sizes) + 1), learner.task_sizes)
    fixed = {r["sample_id"]: r for r in read_csv(ROOT.parent / "error_trajectory_diagnostic" /
                                               learner.args["protocol_id"] / "sample_trajectory.csv")}
    learner._network.eval()
    pointer = 0
    with torch.no_grad():
        for _, inputs, labels in learner.test_loader:
            logits = learner._network(inputs.to(learner._device))["logits"]
            scores = logits.cpu().numpy() * float(learner.args["scale"])
            predicted = scores.argmax(1)
            labels = labels.numpy()
            for i, label in enumerate(labels):
                sample_id = str(ids[pointer])
                if int(expected[pointer]) != int(label) or int(fixed[sample_id]["true_label"]) != int(label):
                    raise AssertionError("evaluation identity/class mapping mismatch")
                own_group = birth[label]
                local_classes = np.flatnonzero(birth == own_group)
                local_pred = int(local_classes[scores[i, local_classes].argmax()])
                rival = scores[i].copy()
                rival[label] = -np.inf
                local_rival = rival[local_classes].max()
                fixed_row = fixed[sample_id]
                kind = fixed_row["error_type"]
                if kind == "B_birth_correct_final_wrong":
                    kind = "B1" if fixed_row["birth_restriction_restored"] == "1" else "B2"
                elif kind.startswith("A_"):
                    kind = "A"
                elif kind.startswith("C_"):
                    kind = "C"
                rows.append({"sample_id": sample_id, "y": int(label), "birth_task": int(own_group),
                             "branch": branch, "evaluated_after_task": task, "pred": int(predicted[i]),
                             "local_pred": local_pred, "fixed_type": kind,
                             "margin_all": float(scores[i, label] - rival.max()),
                             "margin_within": float(scores[i, label] - local_rival),
                             "head_CE": float(torch.nn.functional.cross_entropy(
                                 torch.as_tensor(scores[i:i + 1]), torch.tensor([int(label)])))})
                pointer += 1
    if pointer != len(ids):
        raise AssertionError("evaluation sample count mismatch")
    write_csv(target / f"predictions_T{task:02d}.csv", rows)
    return rows


def save_gm_fork(learner, manager, location):
    pending = learner._gm_state
    if pending is None:
        raise RuntimeError("missing GM fork")
    original_dir, original_args = learner.run_dir, learner.args.copy()
    original_head = copy.deepcopy(learner._network.fc.state_dict())
    means, covs = learner._class_means, learner._class_covs
    current_rng = capture_rng_state()
    original_sidecar = getattr(learner, "_audit_pre_ca_sidecar", None)
    gm_dir = run_dir(learner.args["protocol_id"], location, "GM")
    gm_dir.mkdir(parents=True, exist_ok=True)
    try:
        learner._network.fc.load_state_dict(pending["head"])
        learner._class_means, learner._class_covs = pending["means"], pending["covs"]
        restore_rng_state(pending["rng"])
        learner.run_dir = gm_dir
        learner.args = {**original_args, "experiment_id": "GM", "augmentation_branch": "GM", "run_dir": str(gm_dir)}
        learner._audit_pre_ca_sidecar = pending.get("sidecar")
        evaluate_task(learner, manager, "GM", gm_dir)
        learner.save_experiment_checkpoint(gm_dir / "checkpoint_latest.pt", stage="task_end")
        write_json(gm_dir / "fork_manifest.json", {"shared_G_representation": True,
                   "source_pre_CA_bundle": str(original_dir / "pre_ca_bundle.pt"),
                   "shared_representation_sha256": state_hash(learner._network.convnet.state_dict()),
                   "branching_task": learner._cur_task + 1})
    finally:
        learner._network.fc.load_state_dict(original_head)
        learner._class_means, learner._class_covs = means, covs
        learner.run_dir, learner.args = original_dir, original_args
        learner._audit_pre_ca_sidecar = original_sidecar
        restore_rng_state(current_rng)
        learner._gm_state = None


def train_one(protocol, location, branch, parents, device):
    from oracle_audit.runtime import torch_load
    target = run_dir(protocol, location, branch)
    last_task = 1 if location == "T1" else 10
    final = target / "task_checkpoints" / f"task_{last_task:02d}.pt"
    if final.exists() and (target / f"predictions_T{last_task:02d}.csv").exists():
        print(f"Preserve completed {protocol} {location} {branch}", flush=True)
        return
    if branch == "GM" and location == "T1":
        raise RuntimeError("GM T1 must have been saved by the G statistics fork")
    target.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = source_config(protocol)
    config.update({"experiment_id": branch, "augmentation_branch": branch, "device": [device],
                   "run_dir": str(target), "output_root": str(ROOT / "training"),
                   "num_workers": 0, "checkpoint_every_epoch": False})
    if (ROOT / "exploratory_config.json").exists():
        from .common import read_json
        config["exploratory_pilot"] = read_json(ROOT / "exploratory_config.json")
        config["automatic_sanity_policy_sha256"] = sha256(ROOT / "sanity_policy.json")
        config["automatic_sanity_summary_sha256"] = sha256(ROOT / "automatic_sanity_summary.json")
    write_json(target / "resolved_config.json", {**config, "device": [str(device)]})
    manager = data_manager(protocol)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    import random
    random.seed(SEED)
    learner = AugmentationLearner(config, parents)
    initialized_hash = state_hash(learner._network.state_dict())
    source = None
    if location == "T9_T10":
        own_t9 = target / "task_checkpoints/task_09.pt"
        source = own_t9 if own_t9.exists() else checkpoint_path(protocol, 8)
        if branch == "GM" and not own_t9.exists():
            raise RuntimeError("GM needs G's shared-representation T9 fork; do not train GM separately")
        checkpoint = learner.load_experiment_checkpoint(source)
        if int(checkpoint["cur_task"]) not in (7, 8) or checkpoint["stage"] != "task_end":
            raise RuntimeError("continuation must start at T8/T9 task end")
        del checkpoint
        initialized_hash = state_hash(learner._network.state_dict())
    if not (target / "start_manifest.json").exists():
        write_json(target / "start_manifest.json", {"source": str(source) if source else "pretrained initialization",
               "source_sha256": sha256(source) if source else None,
               "initial_network_sha256": initialized_hash,
               "accepted_parent_manifest_sha256": sha256(ROOT / "accepted_parent_manifest.csv"),
               "protocol": protocol, "location": location, "branch": branch,
                   "original_optimizer_state_at_task_boundary": "task_end has no optimizer; original recipe creates new task optimizer"})
    start_task = learner._cur_task + 2
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for task in range(start_task, last_task + 1):
        if location == "T1" and task != 1:
            raise AssertionError("T1 acquisition only")
        if task == 10:
            learner.parents = []  # Generated bank is not learner state for the next task.
        learner.incremental_train(manager)
        if branch == "G" and task == 1:
            learner.save_experiment_checkpoint(target / "pre_ca_bundle.pt", stage="pre_ca")
            learner.prepare_t1_gm()
        evaluate_task(learner, manager, branch, target)
        learner.after_task()
        learner.save_experiment_checkpoint(target / "checkpoint_latest.pt", stage="task_end")
        if branch == "G" and task in (1, 9):
            save_gm_fork(learner, manager, location)
    write_json(target / "resource_usage.json", {"seconds": time.perf_counter() - started,
               "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "source_checkpoint_bytes": source.stat().st_size if source else 0,
               "generated_bank_read_tasks": [1] if location == "T1" and branch == "G" else
                   ([9] if branch == "G" else []), "trained_last_task": last_task})
    del learner
    torch.cuda.empty_cache()


def run_training(device):
    parents = quality_gate()
    parent_hash = sha256(ROOT / "accepted_parent_manifest.csv")
    for path in (ROOT / "training").glob("*/*/*/start_manifest.json"):
        from .common import read_json
        if read_json(path)["accepted_parent_manifest_sha256"] != parent_hash:
            raise RuntimeError("accepted parent set changed after training began")
    for protocol in PROTOCOLS:
        own = [r for r in parents if r["protocol"] == protocol]
        for location in ("T1", "T9_T10"):
            for branch in ("R", "S", "G", "GM"):
                if branch == "GM" and location == "T1":
                    if not (run_dir(protocol, location, branch) / "task_checkpoints/task_01.pt").exists():
                        raise RuntimeError("G failed to create GM T1 statistics fork")
                    continue
                train_one(protocol, location, branch, own, device)
