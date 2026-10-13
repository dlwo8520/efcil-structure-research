from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common import AUDIT, ROOT, fixed_scores, inputs, metrics, write_json
from classifier_diagnostic.pipeline import _assemble_bank, _fixed_gaussian_bank
from classifier_procedure.pipeline import _make_epoch_permutations, _train_cosine
from oracle_audit.oracle_head import cosine_scores, load_classifier_initialization


def ceos_expand(
    old_samples: np.ndarray,
    labels: np.ndarray,
    enemy_features: np.ndarray,
    enemy_labels: np.ndarray,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """CEOS.py expansive_oversampling with k=1, lambda in [0.7,1), L2 normalization.

    The official method uses enemies from the current minibatch. This offline
    matched-CA adaptation uses all actual current-task training features.
    """
    old = torch.as_tensor(old_samples, dtype=torch.float32, device=device)
    old_y = torch.as_tensor(labels, dtype=torch.long, device=device)
    enemy = torch.as_tensor(enemy_features, dtype=torch.float32, device=device)
    enemy_y = torch.as_tensor(enemy_labels, dtype=torch.long, device=device)
    if len(enemy) == 0:
        raise ValueError("no current actual features")
    result = np.empty_like(old_samples, dtype=np.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for start in range(0, len(old), 512):
        batch = old[start : start + 512]
        batch_y = old_y[start : start + 512]
        distance = torch.cdist(batch, enemy, p=2)
        distance[batch_y[:, None] == enemy_y[None, :]] = torch.inf
        if not bool(torch.isfinite(distance).any(dim=1).all()):
            raise RuntimeError("CEOS: old class has no enemy")
        nearest = enemy[distance.argmin(dim=1)]
        lam = 0.7 + 0.3 * torch.rand((len(batch), 1), generator=generator, device=device)
        mixed = F.normalize(lam * batch + (1 - lam) * nearest, p=2, dim=1)
        result[start : start + len(batch)] = mixed.cpu().numpy()
    return result


def run(protocol: str, device: torch.device, seed: int = 1993) -> dict:
    checkpoint_path, checkpoint, cache, known = inputs(protocol, seed)
    out = ROOT / "ceos" / protocol
    out.mkdir(parents=True, exist_ok=True)
    classes = int(checkpoint["total_classes"])
    ids = np.arange(classes, dtype=np.int64)
    initial, sidecar, source = load_classifier_initialization(checkpoint, checkpoint_path, ids, 10)
    if sidecar is None or source != "exact_pre_ca_sidecar":
        raise RuntimeError("exact pre-CA sidecar required")
    exposure = int(sidecar["replay_samples_per_class"])
    epochs = int(sidecar["ca_epochs"])
    old_counts = {c: exposure for c in range(known)}
    new_counts = {c: exposure for c in range(known, classes)}
    stored_means = np.asarray(checkpoint["class_means"][:classes], dtype=np.float64)
    stored_covs = np.stack([x.detach().cpu().numpy() for x in checkpoint["class_covs"][:classes]])
    old_bank = _fixed_gaussian_bank(stored_means, stored_covs, old_counts, 9,
                                     int(sidecar["task_size_argument"]), device, seed + 10104)
    new_bank = _fixed_gaussian_bank(stored_means, stored_covs, new_counts, 9,
                                     int(sidecar["task_size_argument"]), device, seed + 10102)
    gaussian, labels = _assemble_bank({**old_bank, **new_bank}, ids)
    old_count = exposure * known
    expanded = ceos_expand(
        gaussian[:old_count], labels[:old_count],
        cache["train_features"][cache["train_labels"] >= known],
        cache["train_labels"][cache["train_labels"] >= known],
        device, seed + 10105,
    )
    ceos = gaussian.copy()
    ceos[:old_count] = expanded
    permutations = _make_epoch_permutations(labels, epochs, seed + 10200)
    common = dict(
        protocol=protocol, initial_weights=initial,
        task_sizes=list(map(int, sidecar["task_sizes"])),
        validation_features=np.empty((0, gaussian.shape[1]), dtype=np.float32),
        validation_labels=np.empty(0, dtype=np.int64),
        scale=float(sidecar["scale"]), init_lr=float(sidecar["init_lr"]),
        weight_decay=float(sidecar["weight_decay"]), max_epochs=epochs,
        batch_size=exposure, device=device, permutations=permutations,
        convergence=False, min_epochs=epochs, patience=epochs,
        relative_loss_threshold=1e-4,
    )
    baseline = _train_cosine(condition="C0_REPRODUCED", train_features=gaussian,
                             train_labels=labels, **common)
    learned = _train_cosine(condition="C1_CEOS_ONLY", train_features=ceos,
                            train_labels=labels, **common)
    if learned.optimizer_steps != epochs * classes or baseline.optimizer_steps != learned.optimizer_steps:
        raise RuntimeError("CEOS matched step mismatch")
    matched_path = AUDIT / "paired_source_prediction_diagnostic" / protocol / "matched_scores.npz"
    with np.load(matched_path, allow_pickle=False) as previous:
        if not np.array_equal(previous["labels"], cache["test_labels"]):
            raise RuntimeError("C0 labels differ from canonical test cache")
        c0_scores = previous["scores_C"].copy()
    c0_reproduced = cosine_scores(cache["test_features"], baseline.weights, float(sidecar["scale"]))
    max_difference = float(np.max(np.abs(c0_reproduced - c0_scores)))
    if max_difference > 1e-4:
        raise RuntimeError(f"C0 reproduction differs from saved matched baseline: max score diff={max_difference}")
    c1_scores = cosine_scores(cache["test_features"], learned.weights, float(sidecar["scale"]))
    results = {
        "C0_matched_stored_gaussian": metrics(c0_scores, cache["test_labels"], classes, known),
        "C1_ceos_only": metrics(c1_scores, cache["test_labels"], classes, known),
        "control": {"exposure_per_class": exposure, "epochs": epochs,
                    "optimizer_steps": learned.optimizer_steps,
                    "pre_ca_initialization": source, "same_new_bank": True,
                    "same_label_order_batch_boundaries": True,
                    "acb": False, "focal_loss": False, "class_weights": False,
                    "enemy_pool": "all actual T10 current-task training features",
                    "official_difference": "official CEOS uses current minibatch enemies",
                    "old_source": "stored Gaussian samples plus CEOS expansion",
                    "c0_source": str(matched_path),
                    "c0_reproduction_max_score_difference": max_difference},
        "train_loss_c0": float(baseline.train_loss),
        "train_loss_c1": float(learned.train_loss),
    }
    for name, features in (("c0_synthetic_margin", gaussian[:old_count]),
                           ("ceos_synthetic_margin", expanded)):
        synthetic_scores = fixed_scores(features, checkpoint)
        own = synthetic_scores[np.arange(old_count), labels[:old_count]].copy()
        synthetic_scores[np.arange(old_count), labels[:old_count]] = -np.inf
        margin = own - synthetic_scores.max(axis=1)
        results[name] = {"q05": float(np.quantile(margin, .05)),
                         "q10": float(np.quantile(margin, .1)),
                         "negative_fraction": float(np.mean(margin < 0))}
    write_json(out / "metrics.json", results)
    print(f"[{protocol} CEOS] C0={results['C0_matched_stored_gaussian']['seen_class_accuracy']:.2f} C1={results['C1_ceos_only']['seen_class_accuracy']:.2f}", flush=True)
    return results
