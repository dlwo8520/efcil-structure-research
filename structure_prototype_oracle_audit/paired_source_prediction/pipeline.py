from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from classifier_diagnostic.pipeline import (
    _assemble_bank, _edge_fingerprint, _fixed_gaussian_bank, _stats,
    read_csv, write_csv, write_json,
)
from classifier_procedure.pipeline import _make_epoch_permutations, _train_cosine
from oracle_audit.config import ROOT
from oracle_audit.oracle_head import cosine_scores, load_classifier_initialization
from oracle_audit.runtime import birth_task, torch_load
from ten_task_revalidation.pipeline import _checkpoint_dir, _feature_cache_path


@dataclass
class Options:
    output_root: Path
    device: torch.device
    physical_gpu: str
    protocols: tuple[str, ...] = ("IA10", "CAR10")
    seed: int = 1993
    bootstrap_repeats: int = 2000
    force: bool = False


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _source_paths(protocol: str, seed: int) -> dict[str, Path]:
    task = 10 if protocol.endswith("10") else 5
    base = ROOT / "ten_task_hypothesis_revalidation" / protocol
    return {
        "checkpoint": _checkpoint_dir(protocol, seed) / f"task_{task:02d}.pt",
        "feature_cache": _feature_cache_path(base, task, task),
        "accuracy": base / "g256_r256_final_task.csv" if task == 10 else ROOT / "classifier_procedure_diagnostic" / f"{protocol}_T5" / "g256_r256_accuracy.csv",
        "support_A": ROOT / "coverage_generalization_diagnostic" / protocol / "query_support_A.csv",
        "support_B": ROOT / "coverage_generalization_diagnostic" / protocol / "query_support_B.csv",
        "support_C": ROOT / "coverage_generalization_diagnostic" / protocol / "query_support_C.csv",
        "support_delta": ROOT / "coverage_generalization_diagnostic" / protocol / "paired_support_delta.csv",
    }


def _reference_accuracy(path: Path) -> dict[str, float]:
    result = {}
    for row in read_csv(path):
        condition = row.get("condition")
        if condition in {"R256_FIXED", "G256_FIXED"}:
            result[condition] = float(row["seen_class_accuracy"])
    return result


def _support_join(paths: dict[str, Path], sample_ids: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise RuntimeError("canonical test sample IDs are not unique")
    result: dict[str, np.ndarray] = {}
    for name in ("A", "B", "C"):
        rows = read_csv(paths[f"support_{name}"])
        grouped: dict[str, list[float]] = {}
        classes: dict[str, int] = {}
        for row in rows:
            sid = row["sample_id"]
            grouped.setdefault(sid, []).append(float(row["support_margin"]))
            if sid in classes and classes[sid] != int(row["class_id"]):
                raise RuntimeError(f"support class mismatch: {sid}")
            classes[sid] = int(row["class_id"])
        if set(grouped) != set(map(str, sample_ids)):
            raise RuntimeError(f"support {name} sample ID set mismatch")
        expected_draws = 1 if name == "A" else 3
        if any(len(values) != expected_draws for values in grouped.values()):
            raise RuntimeError(f"support {name}: expected {expected_draws} draws per sample")
        if any(classes[str(sid)] != int(label) for sid, label in zip(sample_ids, labels)):
            raise RuntimeError(f"support {name} true label mismatch")
        result[name] = np.asarray([np.mean(grouped[str(sid)]) for sid in sample_ids], dtype=np.float64)
        result[f"{name}_draw_std"] = np.asarray([np.std(grouped[str(sid)]) for sid in sample_ids], dtype=np.float64)
    paired = read_csv(paths["support_delta"])
    if set(row["sample_id"] for row in paired) != set(map(str, sample_ids)):
        raise RuntimeError("paired support delta sample ID set mismatch")
    return result


def _train_matched_heads(
    protocol: str, checkpoint: dict, checkpoint_path: Path, cache: dict[str, np.ndarray],
    device: torch.device, seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    task = 10 if protocol.endswith("10") else 5
    train_features = np.asarray(cache["train_features"], dtype=np.float32)
    train_labels = np.asarray(cache["train_labels"], dtype=np.int64)
    classes = int(checkpoint["total_classes"])
    class_ids = np.arange(classes)
    partition = list(map(int, checkpoint["task_sizes"]))
    known_before = classes - partition[-1]
    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle_means, oracle_covariances, actual_banks = _stats(train_features, train_labels, class_ids, jitter)
    stored_means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float64)
    stored_covariances = np.stack([value.detach().cpu().numpy() for value in checkpoint["class_covs"][:classes]])
    initial, sidecar, source = load_classifier_initialization(checkpoint, checkpoint_path, class_ids, task)
    if sidecar is None or source != "exact_pre_ca_sidecar":
        raise RuntimeError(f"{protocol}: exact pre-CA sidecar required")
    exposure = int(sidecar["replay_samples_per_class"])
    epochs = int(sidecar["ca_epochs"])
    if exposure != 256:
        raise RuntimeError(f"{protocol}: expected R256/G256 exposure=256, got {exposure}")
    task_size_arg = int(sidecar["task_size_argument"])
    old_counts = {i: exposure for i in range(known_before)}
    new_counts = {i: exposure for i in range(known_before, classes)}
    # The calls/seeds below reproduce the existing matched G256/R256 diagnostic.
    gaussian_old = _fixed_gaussian_bank(
        oracle_means, oracle_covariances, old_counts, task - 1,
        task_size_arg, device, seed + task * 1000 + 101,
    )
    new_common = _fixed_gaussian_bank(
        stored_means, stored_covariances, new_counts, task - 1,
        task_size_arg, device, seed + task * 1000 + 102,
    )
    rng = np.random.RandomState(seed + task * 1000 + 103)
    actual_old = {
        i: actual_banks[i][rng.randint(0, actual_banks[i].shape[0], size=exposure)].copy()
        for i in range(known_before)
    }
    # C uses the exact same Gaussian helper and mean scaling as B; only old
    # class moments change. Structure's dynamic CA is never used as h_C.
    stored_old = _fixed_gaussian_bank(
        stored_means, stored_covariances, old_counts, task - 1,
        task_size_arg, device, seed + task * 1000 + 104,
    )
    banks = {
        "A": _assemble_bank({**actual_old, **new_common}, class_ids),
        "B": _assemble_bank({**gaussian_old, **new_common}, class_ids),
        "C": _assemble_bank({**stored_old, **new_common}, class_ids),
    }
    labels = banks["B"][1]
    if not all(np.array_equal(labels, item[1]) for item in banks.values()):
        raise RuntimeError("A/B/C exposure or label order mismatch")
    permutations = _make_epoch_permutations(labels, epochs, seed + task * 1000 + 200)
    common = dict(
        protocol=protocol,
        initial_weights=initial,
        task_sizes=list(map(int, sidecar["task_sizes"])),
        validation_features=np.empty((0, train_features.shape[1]), dtype=np.float32),
        validation_labels=np.empty(0, dtype=np.int64),
        scale=float(sidecar["scale"]),
        init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]),
        max_epochs=epochs,
        batch_size=exposure,
        device=device,
        permutations=permutations,
        convergence=False,
        min_epochs=epochs,
        patience=epochs,
        relative_loss_threshold=1e-4,
    )
    scores: dict[str, np.ndarray] = {}
    steps = {}
    test_features = np.asarray(cache["test_features"], dtype=np.float32)
    for name in ("A", "B", "C"):
        print(f"[{protocol}] train matched h_{name} ({epochs} epochs, {exposure}/class)", flush=True)
        result = _train_cosine(
            condition=f"MATCHED_{name}", train_features=banks[name][0], train_labels=labels, **common,
        )
        scores[name] = cosine_scores(test_features, result.weights, common["scale"])
        steps[name] = result.optimizer_steps
    expected_steps = epochs * classes
    if len(set(steps.values())) != 1 or next(iter(steps.values())) != expected_steps:
        raise RuntimeError(f"optimizer step mismatch: {steps}, expected {expected_steps}")
    metadata = {
        "exposure_per_class": exposure,
        "epochs": epochs,
        "optimizer_steps": steps,
        "same_pre_ca_initialization": True,
        "initialization_source": source,
        "same_new_bank": True,
        "same_label_order_and_batch_boundaries": True,
        "same_optimizer_lr_schedule": True,
        "old_source_A": "actual feature with replacement (legacy R256)",
        "old_source_B": "oracle Gaussian (legacy G256)",
        "old_source_C": "stored/APR Gaussian, matched new head",
        "old_gaussian_seed_B": seed + task * 1000 + 101,
        "old_gaussian_seed_C": seed + task * 1000 + 104,
        "new_bank_seed": seed + task * 1000 + 102,
        "actual_old_seed": seed + task * 1000 + 103,
        "permutation_seed": seed + task * 1000 + 200,
        "coverage_bank_note": "Coverage A/B/C uses matched unique class counts; head A/B/C uses matched 256/class exposure. Support is explanatory, not the training source itself.",
    }
    return scores, metadata


def _prediction_columns(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.argmax(scores, axis=1)
    rivals = scores.copy()
    rivals[np.arange(labels.size), labels] = -np.inf
    margins = scores[np.arange(labels.size), labels] - np.max(rivals, axis=1)
    return predictions, predictions == labels, margins


def run_protocol(protocol: str, options: Options) -> dict:
    started = time.time()
    output = options.output_root / protocol
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "protocol_manifest.json"
    if manifest_path.is_file() and not options.force:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "complete" and (output / "paired_source_prediction_table.csv").is_file():
            print(f"[{protocol}] complete, reuse", flush=True)
            return prior
    paths = _source_paths(protocol, options.seed)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs:\n" + "\n".join(missing))
    before = _edge_fingerprint(paths["checkpoint"])
    checkpoint = torch_load(paths["checkpoint"])
    with np.load(paths["feature_cache"], allow_pickle=False) as file:
        cache = {key: np.asarray(file[key]) for key in file.files}
    ids = np.asarray(cache["test_sample_ids"]).astype(str)
    labels = np.asarray(cache["test_labels"], dtype=np.int64)
    support = _support_join(paths, ids, labels)
    reference = _reference_accuracy(paths["accuracy"])
    if not {"R256_FIXED", "G256_FIXED"}.issubset(reference):
        raise RuntimeError(f"{protocol}: legacy matched accuracy rows missing")
    scores_path = output / "matched_scores.npz"
    metadata_path = output / "matched_head_manifest.json"
    if scores_path.is_file() and metadata_path.is_file() and not options.force:
        with np.load(scores_path, allow_pickle=False) as file:
            if not np.array_equal(file["sample_ids"].astype(str), ids) or not np.array_equal(file["labels"], labels):
                raise RuntimeError(f"{protocol}: cached score identity mismatch")
            scores = {name: np.asarray(file[f"scores_{name}"], dtype=np.float32) for name in "ABC"}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"[{protocol}] reuse matched score cache", flush=True)
    else:
        scores, metadata = _train_matched_heads(
            protocol, checkpoint, paths["checkpoint"], cache, options.device, options.seed,
        )
    observed = {}
    predictions, correctness, margins = {}, {}, {}
    for name in "ABC":
        predictions[name], correctness[name], margins[name] = _prediction_columns(scores[name], labels)
        observed[name] = float(correctness[name].mean() * 100.0)
    # A/B reproduction is the gate before any paired interpretation or cache write.
    tolerance = 1e-6
    for name, legacy in (("A", "R256_FIXED"), ("B", "G256_FIXED")):
        if abs(observed[name] - reference[legacy]) > tolerance:
            raise RuntimeError(
                f"{protocol} h_{name} failed legacy accuracy regression: "
                f"observed={observed[name]:.10f}, previous={reference[legacy]:.10f}"
            )
    if not scores_path.is_file() or options.force:
        _atomic_npz(scores_path, sample_ids=ids, labels=labels,
                    scores_A=scores["A"], scores_B=scores["B"], scores_C=scores["C"])
        write_json(metadata_path, metadata)
    partition = list(map(int, checkpoint["task_sizes"]))
    old_count = int(checkpoint["total_classes"]) - partition[-1]
    reference_counts = {int(row["class_id"]): int(row["unique_actual_train_count"])
                        for row in read_csv(ROOT / "coverage_generalization_diagnostic" / protocol / "bank_stats.csv")}
    rows = []
    for index, sid in enumerate(ids):
        cls = int(labels[index])
        row = {
            "protocol": protocol, "sample_id": sid, "true_label": cls, "class_id": cls,
            "old_or_new": "old" if cls < old_count else "new",
            "class_age": len(partition) - birth_task(cls, partition),
            "reference_count": reference_counts[cls],
        }
        for name in "ABC":
            row.update({f"pred_{name}": int(predictions[name][index]),
                        f"correct_{name}": bool(correctness[name][index]),
                        f"classifier_margin_{name}": float(margins[name][index]),
                        f"support_{name}": float(support[name][index]),
                        f"support_{name}_draw_std": float(support[f"{name}_draw_std"][index])})
        row.update({
            "delta_BA_support": row["support_B"] - row["support_A"],
            "delta_CB_support": row["support_C"] - row["support_B"],
            "signflip_AB": row["support_A"] > 0 and row["support_B"] < 0,
            "signflip_BC": row["support_B"] > 0 and row["support_C"] < 0,
        })
        rows.append(row)
    write_csv(output / "paired_source_prediction_table.csv", rows)
    after = _edge_fingerprint(paths["checkpoint"])
    if before != after:
        raise RuntimeError(f"{protocol}: source checkpoint fingerprint changed")
    result = {
        "status": "complete", "protocol": protocol, "seed": options.seed,
        "sample_count": len(rows), "accuracy_percent": observed,
        "legacy_accuracy_percent": reference,
        "legacy_regression_passed": True,
        "checkpoint_fingerprint_before": before,
        "checkpoint_fingerprint_after": after,
        "matched_head": metadata,
        "support_draw_policy": "B/C per-sample mean over three coverage draws",
        "elapsed_seconds": time.time() - started,
    }
    write_json(manifest_path, result)
    return result


def run(options: Options) -> dict:
    started = time.time()
    options.output_root.mkdir(parents=True, exist_ok=True)
    audit_lines = ["# Paired Source Prediction Input Audit", "", f"- Seed: {options.seed}",
                   f"- GPU: physical {options.physical_gpu}, process-local {options.device}",
                   "- Existing R256/G256 sample-level logits and trained weights: absent; only aggregate accuracies are available.",
                   "- h_A/h_B are reproduced from the exact prior phase-3 code path and checked against its aggregate accuracy.",
                   "- h_C uses the same fixed exposure, new bank, pre-CA head and optimization controls.",
                   "- Coverage support B/C is averaged over its three independent draws per canonical sample ID.", ""]
    for protocol in options.protocols:
        paths = _source_paths(protocol, options.seed)
        audit_lines.extend([f"## {protocol}", ""])
        for name, path in paths.items():
            audit_lines.append(f"- {name}: {'OK' if path.is_file() else 'MISSING'} — `{path}`")
        audit_lines.append("")
    (options.output_root / "audit.md").write_text("\n".join(audit_lines), encoding="utf-8")
    results = {}
    for protocol in options.protocols:
        print(f"[{protocol}] paired source diagnostic start", flush=True)
        results[protocol] = run_protocol(protocol, options)
        write_json(options.output_root / "run_manifest.json", {
            "status": "running", "completed_protocols": list(results), "protocols": list(options.protocols),
        })
    from .report import generate_report
    report = generate_report(options.output_root, options.protocols, options.bootstrap_repeats, options.seed)
    result = {"status": "complete", "protocols": list(options.protocols), "seed": options.seed,
              "physical_gpu": options.physical_gpu, "protocol_results": results,
              "report": str(report), "elapsed_seconds": time.time() - started}
    write_json(options.output_root / "run_manifest.json", result)
    return result
