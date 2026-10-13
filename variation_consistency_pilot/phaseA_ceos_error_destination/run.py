"""Recover the earlier C1 head deterministically, then compare paired New logits."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT.parent / "structure_prototype_oracle_audit"
FECAM = ROOT.parent / "fecam_boundary_followup"
sys.path[:0] = [str(AUDIT), str(FECAM)]

from ceos.run import ceos_expand  # noqa: E402
from classifier_diagnostic.pipeline import _assemble_bank, _fixed_gaussian_bank  # noqa: E402
from classifier_procedure.pipeline import _make_epoch_permutations, _train_cosine  # noqa: E402
from oracle_audit.oracle_head import cosine_scores, load_classifier_initialization  # noqa: E402
from ten_task_revalidation.pipeline import _checkpoint_dir  # noqa: E402
from oracle_audit.runtime import torch_load  # noqa: E402


def _distribution(values: np.ndarray) -> dict:
    if not len(values):
        return {key: None for key in ("mean", "median", "q10", "q25", "q75", "q90")}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "q10": float(np.quantile(values, .10)), "q25": float(np.quantile(values, .25)),
            "q75": float(np.quantile(values, .75)), "q90": float(np.quantile(values, .90))}


def _margins(scores: np.ndarray, labels: np.ndarray, known: int) -> tuple[np.ndarray, np.ndarray]:
    own = scores[np.arange(len(labels)), labels]
    best_old = scores[:, :known].max(axis=1)
    new = scores[:, known:].copy()
    new[np.arange(len(labels)), labels - known] = -np.inf
    best_other_new = new.max(axis=1)
    return own - best_old, own - best_other_new


def _type(g: float, w: float) -> str:
    if g < 0 and w > 0:
        return "type1_old_competition"
    if g > 0 and w < 0:
        return "type2_new_internal"
    if g < 0 and w < 0:
        return "type3_both"
    return "tie_or_ambiguous"


def _recover_c1(protocol: str, cache: dict, checkpoint: dict, checkpoint_path: Path,
                known: int, device: torch.device, seed: int) -> tuple[np.ndarray, dict]:
    classes = int(checkpoint["total_classes"])
    ids = np.arange(classes, dtype=np.int64)
    initial, sidecar, source = load_classifier_initialization(checkpoint, checkpoint_path, ids, 10)
    if sidecar is None or source != "exact_pre_ca_sidecar":
        raise RuntimeError("exact pre-CA sidecar required")
    exposure = int(sidecar["replay_samples_per_class"])
    epochs = int(sidecar["ca_epochs"])
    task_size = int(sidecar["task_size_argument"])
    means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float64)
    covs = np.stack([value.detach().cpu().numpy() for value in checkpoint["class_covs"][:classes]])
    old = _fixed_gaussian_bank(means, covs, {c: exposure for c in range(known)},
                               9, task_size, device, seed + 10104)
    new = _fixed_gaussian_bank(means, covs, {c: exposure for c in range(known, classes)},
                               9, task_size, device, seed + 10102)
    features, labels = _assemble_bank({**old, **new}, ids)
    old_count = exposure * known
    features[:old_count] = ceos_expand(
        features[:old_count], labels[:old_count],
        cache["train_features"][cache["train_labels"] >= known],
        cache["train_labels"][cache["train_labels"] >= known],
        device, seed + 10105,
    )
    permutations = _make_epoch_permutations(labels, epochs, seed + 10200)
    result = _train_cosine(
        condition="C1_CEOS_ONLY_RECOVERY", protocol=protocol, initial_weights=initial,
        task_sizes=list(map(int, sidecar["task_sizes"])),
        train_features=features, train_labels=labels,
        validation_features=np.empty((0, features.shape[1]), dtype=np.float32),
        validation_labels=np.empty(0, dtype=np.int64),
        scale=float(sidecar["scale"]), init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]), max_epochs=epochs,
        batch_size=exposure, device=device, permutations=permutations,
        convergence=False, min_epochs=epochs, patience=epochs,
        relative_loss_threshold=1e-4,
    )
    if result.optimizer_steps != epochs * classes:
        raise RuntimeError("C1 recovery step mismatch")
    scores = cosine_scores(cache["test_features"], result.weights, float(sidecar["scale"]))
    return scores, {"optimizer_steps": result.optimizer_steps, "train_loss": result.train_loss,
                    "exposure_per_class": exposure, "epochs": epochs}


def run(protocol: str, device: torch.device, seed: int = 1993) -> dict:
    checkpoint_path = _checkpoint_dir(protocol, seed) / "task_10.pt"
    checkpoint = torch_load(checkpoint_path)
    # task_end snapshots are saved after after_task(), so known_classes==total.
    known = int(checkpoint["total_classes"]) - int(checkpoint["task_sizes"][-1])
    cache_path = AUDIT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / "model_10_on_task_10.npz"
    with np.load(cache_path, allow_pickle=False) as source:
        cache = {key: source[key].copy() for key in ("train_features", "train_labels", "test_features", "test_labels", "test_sample_ids")}
    scores_c1, recovery = _recover_c1(protocol, cache, checkpoint, checkpoint_path, known, device, seed)
    previous = json.loads((FECAM / "ceos" / protocol / "metrics.json").read_text())
    from oracle_audit.oracle_head import evaluate_scores
    measured = evaluate_scores(scores_c1, cache["test_labels"], np.arange(int(checkpoint["total_classes"])), known)
    max_metric_difference = max(abs(float(measured[key]) - float(previous["C1_ceos_only"][key]))
                                for key in ("seen_class_accuracy", "old_accuracy", "new_accuracy"))
    if max_metric_difference > 1e-9 or abs(recovery["train_loss"] - previous["train_loss_c1"]) > 1e-9:
        raise RuntimeError(f"recovered C1 differs from previous result: metric={max_metric_difference}")
    matched = AUDIT / "paired_source_prediction_diagnostic" / protocol / "matched_scores.npz"
    with np.load(matched, allow_pickle=False) as source:
        if not np.array_equal(source["labels"], cache["test_labels"]):
            raise RuntimeError("C0/New sample label mismatch")
        scores_c0 = source["scores_C"].copy()
        sample_ids = source["sample_ids"].copy()
    if not np.array_equal(sample_ids, cache["test_sample_ids"]):
        raise RuntimeError("C0/New sample identity mismatch")
    new_mask = cache["test_labels"] >= known
    labels = cache["test_labels"][new_mask]
    ids = sample_ids[new_mask]
    a, b = scores_c0[new_mask], scores_c1[new_mask]
    g0, w0 = _margins(a, labels, known)
    g1, w1 = _margins(b, labels, known)
    correct0 = a.argmax(axis=1) == labels
    correct1 = b.argmax(axis=1) == labels
    loss = correct0 & ~correct1
    recovery_mask = ~correct0 & correct1
    rows = []
    for i in range(len(labels)):
        category = _type(float(g1[i]), float(w1[i])) if loss[i] else (
            _type(float(g0[i]), float(w0[i])) if recovery_mask[i] else "")
        rows.append({"sample_id": str(ids[i]), "label": int(labels[i]),
                     "gaussian_prediction": int(a[i].argmax()), "ceos_prediction": int(b[i].argmax()),
                     "transition": "loss" if loss[i] else "recovery" if recovery_mask[i] else "unchanged",
                     "type": category, "g_gaussian": float(g0[i]), "w_gaussian": float(w0[i]),
                     "g_ceos": float(g1[i]), "w_ceos": float(w1[i]),
                     "delta_g": float(g1[i] - g0[i]), "delta_w": float(w1[i] - w0[i])})
    out = ROOT / "phaseA_ceos_error_destination"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{protocol}_new_error_breakdown.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    type_names = ("type1_old_competition", "type2_new_internal", "type3_both", "tie_or_ambiguous")
    counts = {"loss": {name: int(sum(bool(loss[i]) and rows[i]["type"] == name for i in range(len(rows)))) for name in type_names},
              "recovery": {name: int(sum(bool(recovery_mask[i]) and rows[i]["type"] == name for i in range(len(rows)))) for name in type_names}}
    summary = {"protocol": protocol, "N_new_test": len(labels),
               "gaussian_correct_ceos_wrong_count": int(loss.sum()),
               "gaussian_correct_ceos_wrong_rate": float(loss.mean()),
               "gaussian_wrong_ceos_correct_count": int(recovery_mask.sum()),
               "gaussian_wrong_ceos_correct_rate": float(recovery_mask.mean()),
               "type_counts": counts,
               "type_rates_within_transition": {kind: {name: count / max(int(mask.sum()), 1) for name, count in counts[kind].items()}
                                                for kind, mask in (("loss", loss), ("recovery", recovery_mask))},
               "delta_g": {kind: _distribution((g1 - g0)[mask]) for kind, mask in
                           (("all_new", np.ones(len(labels), bool)), ("loss", loss), ("recovery", recovery_mask))},
               "delta_w": {kind: _distribution((w1 - w0)[mask]) for kind, mask in
                           (("all_new", np.ones(len(labels), bool)), ("loss", loss), ("recovery", recovery_mask))},
               "c1_recovery": recovery, "max_prior_metric_difference": max_metric_difference}
    (out / f"{protocol}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(out / f"{protocol}_paired_new_logits.npz", sample_ids=ids, labels=labels,
                        scores_gaussian=a, scores_ceos=b)
    return summary
