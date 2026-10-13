"""Read-only, task-end Structure error trajectories for IA10 and CAR10.

Only inference is performed.  The source checkpoint tree is never written to.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = ROOT.parent
AUDIT_ROOT = HOME / "structure_prototype_oracle_audit"
SOURCE_ROOT = AUDIT_ROOT / "training_outputs"
SEED = 1993
PROTOCOLS = ("IA10", "CAR10")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_ids(ids, labels) -> str:
    digest = hashlib.sha256()
    for sample_id, label in zip(ids, labels):
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(int(label)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_audit(manifest: dict) -> None:
    lines = [
        "# Final Error Trajectory — consistency audit", "",
        "이 문서는 read-only 진단 실행 중 갱신된다. 원본 checkpoint는 수정하지 않는다.", "",
        f"- Seed: {manifest['seed']}",
        f"- Physical evaluation GPU: {manifest['physical_gpu']} (process-local cuda:0)",
        "- Evaluation: frozen task-end network, post-CA at T2–T10; T1 has no CA sidecar",
        "- Candidate set: each task's mapped classes 0..seen-1",
        "- Sample IDs: `representation_relation.pipeline._dataset_identity` on official test split",
        "- Score source: direct checkpoint inference; older feature-cache logits are not used",
        "- No training, new loss, replay, or test-label tuning", "",
    ]
    for protocol in PROTOCOLS:
        info = manifest.get("protocols", {}).get(protocol)
        if not info:
            lines.extend([f"## {protocol}", "", "아직 시작하지 않음.", ""])
            continue
        lines += [f"## {protocol}", "", f"- Status: {info['status']}",
                  f"- Source training branch: `{info['source_run_dir']}`",
                  f"- Original class order: `{info['original_class_order']}`",
                  f"- Task partition: `{info['task_partition']}`",
                  f"- Final official test samples: {info['final_test_samples']}",
                  f"- Original official test split SHA-256: `{info['official_test_split_sha256']}`",
                  f"- Test split hash mode: `{info['official_test_split_hash_mode']}`",
                  f"- Final sample-ID/label SHA-256: `{info['sample_id_label_sha256']}`",
                  f"- Test preprocessing and normalization: `{info['test_transform']}`",
                  f"- Training-recorded git commit: `{info['source_git_commit_recorded_at_training']}`",
                  f"- Training-recorded source snapshot hash: `{info['source_snapshot_hash_at_training']}`",
                  f"- Current baseline source git commit: `{info['current_base_git_commit']}`", "",
                  "| Task | Checkpoint SHA-256 | State | Candidate classes | Test N | Correct | Historical correct Δ | ID-map SHA-256 |",
                  "|---|---|---|---:|---:|---:|---:|---|"]
        for entry in info["checkpoints"]:
            lines.append(f"| T{entry['task']} | `{entry['sha256']}` | {entry['ca_state']} | "
                         f"0..{entry['candidate_count']-1} | {entry['sample_count']} | "
                         f"{entry['top1_correct']} | {entry['historical_correct_delta']} | "
                         f"`{entry['sample_id_label_sha256']}` |")
        lines += ["", "Historical correct Δ compares task-end checkpoint inference with the "
                  "original task-summary evaluation, which ran immediately before `after_task()`; "
                  "nonzero Δ is recorded, not silently treated as exact parity.", ""]
        lines += ["", "Source-code SHA-256:", ""]
        for path, digest in info["source_code_sha256"].items():
            lines.append(f"- `{path}`: `{digest}`")
        lines.append("")
    temporary = (ROOT / "audit.md").with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, ROOT / "audit.md")


def source_preflight(protocol: str, expected_config: dict) -> tuple[Path, list[Path], dict]:
    """Check the exact baseline branch and all ten snapshot names before GPU inference."""
    from oracle_audit.config import validate_protocol_config

    run_dir = SOURCE_ROOT / protocol / "Structure_AUDIT" / f"seed_{SEED}"
    snapshot_dir = run_dir / "task_checkpoints"
    manifest_path = snapshot_dir / "manifest.json"
    config_path = run_dir / "resolved_config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"baseline snapshot manifest/config missing: {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_protocol_config(config)
    if manifest.get("training_behavior_modified") is not False:
        raise ValueError(f"not an unmodified baseline snapshot manifest: {manifest_path}")
    for key, expected in (
        ("seed", SEED), ("protocol_id", protocol), ("dataset", protocol),
        ("experiment_id", "Structure_AUDIT"), ("method", "Structure"),
        ("expected_tasks", 10), ("ca", True),
    ):
        if config.get(key) != expected:
            raise ValueError(f"{protocol}: resolved_config[{key}]={config.get(key)!r}, expected {expected!r}")
    if Path(config.get("run_dir", "")).resolve() != run_dir.resolve():
        raise ValueError(f"{protocol}: config run_dir points to a different training branch")
    if [int(x) for x in config["task_partition"]] != [int(x) for x in expected_config["task_partition"]]:
        raise ValueError(f"{protocol}: source and evaluation task partitions disagree")
    if Path(config["data_root"]).resolve() != Path(expected_config["data_root"]).resolve():
        raise ValueError(f"{protocol}: evaluation data root differs from training config")
    entries = manifest.get("entries", [])
    if len(entries) != 10 or [int(e["task_id"]) for e in entries] != list(range(1, 11)):
        raise ValueError(f"{protocol}: expected exactly ordered T1..T10 snapshots")
    paths = []
    for task, entry in enumerate(entries, start=1):
        path = snapshot_dir / f"task_{task:02d}.pt"
        if not path.is_file() or Path(entry["path"]).resolve() != path.resolve():
            raise FileNotFoundError(f"{protocol} T{task}: wrong or missing checkpoint {path}")
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError(f"{protocol} T{task}: snapshot size changed")
        if entry.get("state_id") != f"task_end:task_{task}" or int(entry["network_flag_after_task"]) != task:
            raise ValueError(f"{protocol} T{task}: manifest is not the post-task state")
        sidecar = entry.get("pre_ca_classifier_sidecar")
        if task == 1 and sidecar is not None:
            raise ValueError(f"{protocol}: unexpected T1 pre-CA sidecar")
        if task > 1 and (not sidecar or not Path(sidecar).is_file()):
            raise FileNotFoundError(f"{protocol} T{task}: missing pre-CA sidecar; cannot establish CA sequence")
        paths.append(path)
    return run_dir, paths, config


def validate_checkpoint(checkpoint: dict, path: Path, protocol: str, task: int,
                        partition: list[int], source_config: dict) -> None:
    cfg = checkpoint.get("config", {})
    expected_seen = sum(partition[:task])
    for key, expected in (
        ("stage", "task_end"), ("state_id", f"task_end:task_{task}"),
        ("cur_task", task - 1), ("total_classes", expected_seen),
    ):
        if checkpoint.get(key) != expected:
            raise ValueError(f"{path}: {key}={checkpoint.get(key)!r}, expected {expected!r}")
    if [int(x) for x in checkpoint.get("task_sizes", [])] != partition[:task]:
        raise ValueError(f"{path}: task sizes do not match {protocol} partition")
    if int(checkpoint.get("known_classes", -1)) != expected_seen:
        raise ValueError(f"{path}: task-end known class count differs from seen classes")
    for key in ("seed", "protocol_id", "dataset", "experiment_id", "method", "ca", "ca_epochs",
                "task_partition", "data_root", "run_dir"):
        if cfg.get(key) != source_config.get(key):
            raise ValueError(f"{path}: checkpoint config {key} differs from resolved training config")
    if "network_state_dict" not in checkpoint:
        raise ValueError(f"{path}: no saved final network state")


def load_prediction_cache(path: Path, signature: str, ids, labels, seen: int):
    import numpy as np

    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if str(stored["signature"].item()) != signature:
            raise ValueError(f"stale prediction cache (source/code/transform changed): {path}; delete it manually")
        if not np.array_equal(stored["sample_ids"], ids) or not np.array_equal(stored["labels"], labels):
            raise ValueError(f"sample identity changed since cache creation: {path}")
        scores = np.asarray(stored["logits"], dtype=np.float32)
    if scores.shape != (len(ids), seen) or not np.isfinite(scores).all():
        raise ValueError(f"invalid cached logit matrix: {path}")
    return scores


def save_prediction_cache(path: Path, signature: str, ids, labels, scores) -> None:
    import numpy as np

    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, signature=np.asarray(signature), sample_ids=ids,
                        labels=labels, logits=scores)
    os.replace(temporary, path)


def infer_task(checkpoint: dict, data_manager, ids, labels, task: int, seen: int,
               device, batch_size: int, num_workers: int, scratch: Path):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from oracle_audit.runtime import load_frozen_learner, release_learner

    dataset = data_manager.get_dataset(np.arange(seen, dtype=np.int64), source="test", mode="test")
    if len(dataset) != len(ids):
        raise ValueError(f"T{task}: dataset size and canonical ID count differ")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    learner = load_frozen_learner(checkpoint, device, scratch)
    all_scores = []
    observed_labels = []
    expected_index = 0
    try:
        if learner._network.convnet.flag != task:
            raise AssertionError(f"T{task}: loaded network flag does not match task-end manifest")
        with torch.inference_mode():
            for indices, inputs, targets in loader:
                idx = indices.cpu().numpy()
                want = np.arange(expected_index, expected_index + len(idx))
                if not np.array_equal(idx, want):
                    raise AssertionError(f"T{task}: test loader ordering changed")
                if not np.array_equal(targets.numpy(), labels[idx]):
                    raise AssertionError(f"T{task}: test labels differ from canonical mapping")
                logits = learner._network(inputs.to(device, non_blocking=True))["logits"]
                if logits.ndim != 2 or logits.shape[1] != seen:
                    raise AssertionError(f"T{task}: candidate head must have exactly {seen} seen classes")
                all_scores.append(logits.detach().cpu().float().numpy())
                observed_labels.append(targets.numpy().astype(np.int64, copy=False))
                expected_index += len(idx)
        scores = np.concatenate(all_scores, axis=0)
        target_array = np.concatenate(observed_labels)
        if not np.array_equal(target_array, labels) or scores.shape != (len(ids), seen):
            raise AssertionError(f"T{task}: inference sample alignment failed")
        if not np.isfinite(scores).all():
            raise ValueError(f"T{task}: nonfinite logits")
        return scores
    finally:
        release_learner(learner)
        del learner
        import gc
        gc.collect()
        torch.cuda.empty_cache()


def evaluate_protocol(protocol: str, options, manifest: dict) -> None:
    import numpy as np
    import torch
    from oracle_audit.config import BASE_ROOT, DEFAULT_DATA_ROOTS, git_commit
    from oracle_audit.runtime import build_data_manager, torch_load
    from representation_relation.pipeline import _dataset_identity
    from metrics import aggregate, trajectories, write_csv

    data_root = DEFAULT_DATA_ROOTS[protocol]
    manager, expected_config = build_data_manager(protocol, data_root, SEED)
    partition = [int(manager.get_task_size(task)) for task in range(manager.nb_tasks)]
    if len(partition) != 10 or sum(partition) != manager.get_total_classnum():
        raise ValueError(f"{protocol}: not an exhaustive ten-task data manager")
    if partition != [int(x) for x in expected_config["task_partition"]]:
        raise ValueError(f"{protocol}: data manager task partition differs from protocol config")
    run_dir, checkpoints, source_config = source_preflight(protocol, expected_config)
    with (run_dir / "task_summary.csv").open(newline="", encoding="utf-8") as handle:
        historical_rows = list(csv.DictReader(handle))
    historical_by_task = {int(row["task_id"]): row for row in historical_rows
                          if row.get("eval_stage") == "task_final"}
    if set(historical_by_task) != set(range(1, 11)):
        raise ValueError(f"{protocol}: historical task summary lacks one or more T1..T10 rows")

    original_order = json.loads((run_dir / "class_order.json").read_text(encoding="utf-8"))
    if [int(value) for value in original_order["class_order_original_ids"]] != [
        int(value) for value in manager._class_order
    ]:
        raise ValueError(f"{protocol}: class order differs from original training")
    original_protocol = json.loads((run_dir / "protocol_manifest.json").read_text(encoding="utf-8"))
    if [int(value) for value in original_protocol["task_partition"]] != partition:
        raise ValueError(f"{protocol}: protocol manifest partition differs from current data manager")
    if [int(value) for value in original_protocol["class_order_original_ids"]] != [
        int(value) for value in manager._class_order
    ]:
        raise ValueError(f"{protocol}: protocol manifest class order differs from current data manager")
    original_test = original_protocol["splits"]["test"]
    current_test = manager.protocol_manifest["splits"]["test"]
    for key in ("file_count", "root", "sha256", "hash_mode"):
        if original_test[key] != current_test[key]:
            raise ValueError(f"{protocol}: official test split {key} changed since training")

    all_ids, all_labels = _dataset_identity(manager, np.arange(sum(partition)), "test")
    if len(all_ids) != len(set(all_ids.tolist())) or len(all_ids) == 0:
        raise ValueError(f"{protocol}: canonical test IDs missing or duplicated")
    if np.any(np.bincount(all_labels, minlength=sum(partition)) == 0):
        raise ValueError(f"{protocol}: some class has no official test samples")
    if not np.array_equal(np.sort(np.unique(all_labels)), np.arange(sum(partition))):
        raise ValueError(f"{protocol}: labels do not cover exactly mapped classes")
    transform = repr(manager._test_trsf) + " | " + repr(manager._common_trsf)
    source_code_paths = [ROOT / "run.py", ROOT / "metrics.py",
                         AUDIT_ROOT / "oracle_audit/runtime.py",
                         AUDIT_ROOT / "representation_relation/pipeline.py",
                         BASE_ROOT / "models/CDS_adapter.py",
                         BASE_ROOT / "data/data.py", BASE_ROOT / "data/data_manager.py"]
    code_hashes = {str(path): digest_file(path) for path in source_code_paths}
    class_order = [int(value) for value in manager._class_order]
    manifest["protocols"][protocol] = {
        "status": "running", "started_utc": utc_now(), "source_run_dir": str(run_dir),
        "source_resolved_config": str(run_dir / "resolved_config.json"),
        "source_snapshot_manifest": str(run_dir / "task_checkpoints/manifest.json"),
        "source_git_commit_recorded_at_training": source_config.get("git_commit"),
        "source_snapshot_hash_at_training": original_protocol.get("source_snapshot_hash"),
        "current_base_git_commit": git_commit(BASE_ROOT),
        "seed": SEED, "task_partition": partition, "original_class_order": class_order,
        "test_transform": transform, "sample_id_method": "representation_relation.pipeline._dataset_identity",
        "sample_id_label_sha256": digest_ids(all_ids, all_labels),
        "official_test_split_sha256": current_test["sha256"],
        "official_test_split_hash_mode": current_test["hash_mode"],
        "final_test_samples": int(len(all_ids)), "source_code_sha256": code_hashes,
        "checkpoints": [],
    }
    atomic_json(ROOT / "run_manifest.json", manifest)
    write_audit(manifest)

    predictions = np.full((len(all_ids), 10), -1, dtype=np.int64)
    correctness = np.full((len(all_ids), 10), -1, dtype=np.int8)
    final_scores = None
    output = ROOT / protocol
    cache_dir = output / ".inference_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")  # process-local ID; physical GPU selected before torch import
    for task, checkpoint_path in enumerate(checkpoints, start=1):
        began = time.monotonic()
        seen = sum(partition[:task])
        expected_mask = all_labels < seen
        task_ids, task_labels = _dataset_identity(manager, np.arange(seen), "test")
        if len(task_ids) != int(expected_mask.sum()):
            raise AssertionError(f"{protocol} T{task}: test sample count is not prefix-consistent")
        master_position = {str(sample_id): index for index, sample_id in enumerate(all_ids)}
        try:
            positions = np.asarray([master_position[str(item)] for item in task_ids], dtype=np.int64)
        except KeyError as error:
            raise AssertionError(f"{protocol} T{task}: task test ID absent from final sample map") from error
        if len(set(positions.tolist())) != len(positions) or not np.array_equal(all_labels[positions], task_labels):
            raise AssertionError(f"{protocol} T{task}: test sample IDs/labels do not join one-to-one")
        if not np.array_equal(np.sort(positions), np.flatnonzero(expected_mask)):
            raise AssertionError(f"{protocol} T{task}: task test set differs from final-set label subset")
        checkpoint_hash = digest_file(checkpoint_path)
        signature = hashlib.sha256(json.dumps({
            "checkpoint_sha256": checkpoint_hash, "id_sha256": digest_ids(task_ids, task_labels),
            "transform": transform, "code_hashes": code_hashes, "candidate_count": seen,
        }, sort_keys=True).encode()).hexdigest()
        cache_path = cache_dir / f"task_{task:02d}_logits.npz"
        scores = load_prediction_cache(cache_path, signature, task_ids, task_labels, seen) if options.resume else None
        checkpoint = torch_load(checkpoint_path)
        try:
            validate_checkpoint(checkpoint, checkpoint_path, protocol, task, partition, source_config)
            if scores is None:
                print(f"[{protocol}] T{task}/10: frozen inference, {len(task_ids)} samples, {seen} candidates", flush=True)
                scores = infer_task(checkpoint, manager, task_ids, task_labels, task, seen, device,
                                    options.batch_size, options.num_workers,
                                    output / ".scratch" / f"task_{task:02d}")
                save_prediction_cache(cache_path, signature, task_ids, task_labels, scores)
            else:
                print(f"[{protocol}] T{task}/10: validated local inference cache", flush=True)
        finally:
            del checkpoint
        predicted = np.argmax(scores, axis=1).astype(np.int64)
        historical = historical_by_task[task]
        if int(historical["num_samples"]) != len(task_ids):
            raise ValueError(f"{protocol} T{task}: official test count differs from original task summary")
        historical_correct_delta = int((predicted == task_labels).sum()) - int(historical["num_correct"])
        if historical_correct_delta:
            print(f"[{protocol}] T{task}: historical pre-after_task correct-count delta "
                  f"{historical_correct_delta:+d}; recorded in audit", flush=True)
        predictions[positions, task - 1] = predicted
        correctness[positions, task - 1] = (predicted == task_labels).astype(np.int8)
        if task == 10:
            final_scores = np.empty_like(scores)
            final_scores[positions] = scores
        manifest["protocols"][protocol]["checkpoints"].append({
            "task": task, "path": str(checkpoint_path), "sha256": checkpoint_hash,
            "state_id": f"task_end:task_{task}", "ca_state": "post_CA" if task > 1 else "task_end_no_CA_at_T1",
            "candidate_classes": list(range(seen)), "candidate_count": seen,
            "sample_count": len(task_ids), "sample_id_label_sha256": digest_ids(task_ids, task_labels),
            "top1_correct": int((predicted == task_labels).sum()),
            "top1_accuracy": float((predicted == task_labels).mean()),
            "historical_top1_correct": int(historical["num_correct"]),
            "historical_correct_delta": historical_correct_delta,
            "elapsed_seconds": round(time.monotonic() - began, 2),
            "prediction_cache": str(cache_path),
        })
        atomic_json(ROOT / "run_manifest.json", manifest)
        write_audit(manifest)
    if final_scores is None:
        raise AssertionError("no final model logits")
    rows, state = trajectories(all_ids, all_labels, partition, predictions, correctness, final_scores)
    for row in rows:
        row["protocol"] = protocol
    aggregates = aggregate(protocol, all_labels, state)
    write_csv(output / "sample_trajectory.csv", rows)
    write_csv(output / "final_error_composition.csv", aggregates["composition"])
    write_csv(output / "loss_recovery.csv", aggregates["loss_recovery"])
    write_csv(output / "typeB_candidate_restriction.csv", aggregates["restriction"])
    manifest["protocols"][protocol]["status"] = "complete"
    manifest["protocols"][protocol]["completed_utc"] = utc_now()
    manifest["protocols"][protocol]["n_final_wrong"] = int(state["final_wrong"].sum())
    atomic_json(ROOT / "run_manifest.json", manifest)
    write_audit(manifest)
    print(f"[{protocol}] complete; results in {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="physical GPU index; use 3 for this run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="reuse only this diagnostic's hash-validated inference cache")
    options = parser.parse_args()
    if options.gpu < 0 or options.batch_size < 1 or options.num_workers < 0:
        parser.error("GPU ID and batch size must be nonnegative/positive; workers must be nonnegative")
    if "torch" in sys.modules:
        raise RuntimeError("torch imported before physical GPU isolation")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.gpu)
    sys.path.insert(0, str(AUDIT_ROOT))
    from oracle_audit.config import bootstrap_paths
    bootstrap_paths()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"physical GPU {options.gpu} is unavailable to this process")
    from oracle_audit.config import DEFAULT_DATA_ROOTS, protocol_config
    for protocol in PROTOCOLS:
        source_preflight(protocol, protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol]))
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "run_manifest.json"
    if manifest_path.exists() and options.resume:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("seed") != SEED or manifest.get("physical_gpu") != options.gpu:
            raise ValueError("resume manifest seed/GPU mismatch")
    else:
        manifest = {"diagnostic": "Final Error Trajectory", "seed": SEED,
                    "physical_gpu": options.gpu, "created_utc": utc_now(), "protocols": {}}
    try:
        for protocol in PROTOCOLS:
            try:
                evaluate_protocol(protocol, options, manifest)
            except Exception as error:
                manifest.setdefault("errors", []).append({
                    "protocol": protocol, "at_utc": utc_now(),
                    "error_type": type(error).__name__, "message": str(error),
                })
                if protocol in manifest["protocols"]:
                    manifest["protocols"][protocol]["status"] = "failed"
                raise
    finally:
        atomic_json(manifest_path, manifest)
        write_audit(manifest)
    if all(manifest.get("protocols", {}).get(name, {}).get("status") == "complete" for name in PROTOCOLS):
        from report import create_report
        create_report(ROOT, manifest)
        print(f"Final report: {ROOT / 'final_report.md'}", flush=True)
    else:
        print("Both protocols are required for the cross-protocol report; partial results retained.", flush=True)


if __name__ == "__main__":
    main()
