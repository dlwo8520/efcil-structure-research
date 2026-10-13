from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from classifier_diagnostic.pipeline import _edge_fingerprint, read_csv, write_csv, write_json
from classifier_procedure.pipeline import _make_epoch_permutations, _train_cosine
from oracle_audit.config import DEFAULT_DATA_ROOTS, ROOT, bootstrap_paths, protocol_config
from oracle_audit.oracle_head import checkpoint_classifier_weights, cosine_scores
from oracle_audit.runtime import birth_task, build_data_manager, release_learner, torch_load
from representation_relation.pipeline import _dataset_identity, _validation_indices

from .metrics import SupportResult, class_macro, cosine_support, distribution_summary, macro_accuracy, normalize


PRIMARY_PROTOCOLS = ("IA10", "CAR10")
OPTIONAL_PROTOCOLS = ("IA5", "CAR5")
ALL_PROTOCOLS = PRIMARY_PROTOCOLS + OPTIONAL_PROTOCOLS


@dataclass
class CoverageOptions:
    output_root: Path
    device: torch.device
    physical_gpu: str
    protocols: tuple[str, ...] = PRIMARY_PROTOCOLS
    seed: int = 1993
    gaussian_draws: int = 3
    support_k: int = 5
    bank_cap: int | None = None
    batch_size: int = 64
    support_chunk_size: int = 256
    num_workers: int = 4
    probe_epochs: int = 30
    probe_batch_size: int = 256
    probe_lrs: tuple[float, ...] = (0.01, 0.03, 0.1)
    force: bool = False


def _checkpoint_path(protocol: str, seed: int) -> Path:
    task = int(protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])["expected_tasks"])
    return ROOT / "training_outputs" / protocol / "Structure_AUDIT" / f"seed_{seed}" / "task_checkpoints" / f"task_{task:02d}.pt"


def _final_cache_path(protocol: str, seed: int) -> Path:
    del seed
    task = int(protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])["expected_tasks"])
    return ROOT / "ten_task_hypothesis_revalidation" / protocol / ".cache" / f"model_{task:02d}_on_task_{task:02d}.npz"


def _pretrained_cache_path(output_root: Path, protocol: str) -> Path:
    return output_root / protocol / ".cache" / "pretrained_features.npz"


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _reference_bin(count: int) -> str:
    if count <= 4:
        return "01-04"
    if count <= 9:
        return "05-09"
    if count <= 19:
        return "10-19"
    if count <= 49:
        return "20-49"
    return "50+"


def _pretrained_weight_sources() -> list[dict[str, Any]]:
    candidates = [
        Path.home() / ".cache/torch/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
        Path.home() / ".cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in21k/blobs/2d31bb5597c720912f8ea5e4f1de73909a6911798d78ae5d1b9d63a955fed049",
    ]
    rows = []
    for path in candidates:
        if path.is_file():
            rows.append({"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)})
    return rows


def write_audit(options: CoverageOptions) -> dict[str, Any]:
    records: dict[str, Any] = {}
    missing: list[str] = []
    pretrained_sources = _pretrained_weight_sources()
    lines = [
        "# Coverage + Generalization Diagnostic Audit",
        "",
        f"- Seed: `{options.seed}`",
        f"- Requested physical GPU: `{options.physical_gpu}`; process-local device: `{options.device}`",
        f"- Protocol order: `{list(options.protocols)}`",
        f"- Gaussian draws: `{options.gaussian_draws}`; support k: `{options.support_k}`",
        f"- Bank cap: `{options.bank_cap if options.bank_cap is not None else 'none'}`",
        "- Policy: frozen forward and offline diagnostic only; no Structure checkpoint is written.",
        "",
        "## Pretrained initialization source",
        "",
        "`pretrained_vit_b16_224_in21k_adapter` initializes the ViT backbone from the cached timm ImageNet-21K weights; newly created adapters have zero output projections.",
        "",
    ]
    if pretrained_sources:
        lines.extend(["| Cached source | Bytes | SHA-256 |", "|---|---:|---|"])
        for row in pretrained_sources:
            lines.append(f"| `{row['path']}` | {row['size']} | `{row['sha256']}` |")
    else:
        missing.append("local pretrained ViT-B/16 ImageNet-21K weights")
        lines.append("MISSING: no local pretrained ViT-B/16 ImageNet-21K weight file was found.")

    for protocol in options.protocols:
        config = protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])
        partition = list(map(int, config["task_partition"]))
        task = len(partition)
        classes = sum(partition)
        known_before = classes - partition[-1]
        checkpoint_path = _checkpoint_path(protocol, options.seed)
        cache_path = _final_cache_path(protocol, options.seed)
        if not checkpoint_path.is_file():
            missing.append(str(checkpoint_path))
            continue
        if not cache_path.is_file():
            missing.append(str(cache_path))
            continue
        checkpoint = torch_load(checkpoint_path)
        weights = checkpoint_classifier_weights(checkpoint)
        data_manager, _ = build_data_manager(protocol, DEFAULT_DATA_ROOTS[protocol], options.seed)
        class_ids = np.arange(classes, dtype=np.int64)
        train_ids, train_labels = _dataset_identity(data_manager, class_ids, "train")
        test_ids, test_labels = _dataset_identity(data_manager, class_ids, "test")
        train_counts = np.bincount(train_labels, minlength=classes)
        test_counts = np.bincount(test_labels, minlength=classes)
        cache = np.load(cache_path, allow_pickle=False)
        cache_valid = (
            np.array_equal(cache["train_labels"], train_labels)
            and np.array_equal(cache["test_labels"], test_labels)
            and np.array_equal(cache["train_sample_ids"], train_ids)
            and np.array_equal(cache["test_sample_ids"], test_ids)
        )
        record = {
            "task": task,
            "partition": partition,
            "class_count": classes,
            "old_class_count": known_before,
            "new_class_count": partition[-1],
            "checkpoint": _edge_fingerprint(checkpoint_path),
            "stage": checkpoint.get("stage"),
            "classifier_rows": int(weights.shape[0]),
            "stored_mean_count": len(checkpoint.get("class_means", [])),
            "stored_covariance_count": len(checkpoint.get("class_covs", [])),
            "final_feature_cache": str(cache_path),
            "final_feature_cache_identity_valid": bool(cache_valid),
            "train_samples": int(train_labels.size),
            "test_samples": int(test_labels.size),
            "train_count_min": int(train_counts.min()),
            "train_count_median": float(np.median(train_counts)),
            "train_count_max": int(train_counts.max()),
            "test_count_min": int(test_counts.min()),
            "test_count_median": float(np.median(test_counts)),
            "test_count_max": int(test_counts.max()),
            "canonical_sample_ids": True,
            "feature_dimension": int(cache["train_features"].shape[1]),
        }
        records[protocol] = record
        lines.extend([
            "",
            f"## {protocol}",
            "",
            f"- Final task/classes: `T{task}` / `{classes}` (`{known_before}` old + `{partition[-1]}` new)",
            f"- Final checkpoint: `{checkpoint_path}`",
            f"- Checkpoint fingerprint: size `{record['checkpoint']['size']}`, edge SHA-256 `{record['checkpoint']['edge_sha256']}`",
            f"- Head rows / stored means / stored covariances: `{weights.shape[0]}` / `{record['stored_mean_count']}` / `{record['stored_covariance_count']}`",
            f"- Actual train samples per class min/median/max: `{train_counts.min()}/{np.median(train_counts):.1f}/{train_counts.max()}`",
            f"- Test samples per class min/median/max: `{test_counts.min()}/{np.median(test_counts):.1f}/{test_counts.max()}`",
            f"- Canonical train/test IDs agree with reused final cache: `{cache_valid}`",
        ])
        if not cache_valid:
            missing.append(f"{protocol}: final cache identity mismatch")
        del checkpoint, cache, data_manager

    lines.extend([
        "",
        "## Metric-definition checks",
        "",
        "- Classification margin is `true-class cosine score - strongest rival cosine score`; its negative fraction is treated as error-equivalent, not independent representation evidence.",
        "- Support uses L2-normalized query/reference features. Gaussian samples are generated in raw feature space and normalized only before support computation.",
        "- Support k is class-adaptive: `min(k, available class references)`; no duplicate or replacement references are introduced.",
        "- The shared support implementation excludes every reference with the same canonical sample ID when reference/query sets overlap.",
        "- Existing pair normalized entropy uses the nonzero directional-pair probability vector and divides Shannon entropy by `log(number of nonzero pairs)`.",
        "- Existing stored-vs-oracle gamma analysis reports ranking correlation and target-specific Recall@K/captured mass; geometric agreement is not treated as predictive success.",
        "",
        "## Missing inputs",
        "",
        *( ["MISSING INPUTS: none"] if not missing else [f"- `{item}`" for item in missing] ),
    ])
    options.output_root.mkdir(parents=True, exist_ok=True)
    audit_path = options.output_root / "coverage_generalization_audit.md"
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    value = {"records": records, "pretrained_sources": pretrained_sources, "missing": missing, "audit": str(audit_path)}
    write_json(options.output_root / "input_audit.json", value)
    return value


def _extract_pretrained_features(
    protocol: str,
    checkpoint: dict[str, Any],
    data_manager: Any,
    class_ids: np.ndarray,
    options: CoverageOptions,
) -> Path:
    cache_path = _pretrained_cache_path(options.output_root, protocol)
    expected_train_ids, expected_train_labels = _dataset_identity(data_manager, class_ids, "train")
    expected_test_ids, expected_test_labels = _dataset_identity(data_manager, class_ids, "test")
    if cache_path.is_file() and not options.force:
        cached = np.load(cache_path, allow_pickle=False)
        required = {"train_features", "train_labels", "train_sample_ids", "test_features", "test_labels", "test_sample_ids"}
        if required.issubset(cached.files) and np.array_equal(cached["train_sample_ids"], expected_train_ids) and np.array_equal(cached["test_sample_ids"], expected_test_ids):
            return cache_path

    bootstrap_paths()
    from models.CDS_adapter import Learner as StructureLearner

    saved_numpy = np.random.get_state()
    saved_cpu = torch.get_rng_state()
    saved_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    learner = None
    try:
        np.random.seed(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)
        config = dict(checkpoint["config"])
        config.update({
            "device": [options.device],
            "run_dir": str(options.output_root / protocol / ".scratch" / "pretrained"),
            "num_workers": 0,
        })
        learner = StructureLearner(config)
        convnet = learner._network.convnet.to(options.device)
        convnet.eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in convnet.parameters()):
            raise AssertionError("pretrained extractor must be frozen")
        # At initialization both adapter up-projections are exactly zero.  flag=1
        # is the sfeature route used when the network flag is zero.
        nonzero_adapter_outputs = [
            name for name, parameter in convnet.named_parameters()
            if "up_proj" in name and torch.count_nonzero(parameter.detach()).item() != 0
        ]
        if nonzero_adapter_outputs:
            raise RuntimeError(f"fresh pretrained adapter has nonzero output projection: {nonzero_adapter_outputs[:3]}")

        def extract(source: str, expected_ids: np.ndarray, expected_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            dataset = data_manager.get_dataset(class_ids, source=source, mode="test")
            loader = DataLoader(
                dataset,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                pin_memory=options.device.type == "cuda",
            )
            features, labels = [], []
            with torch.no_grad():
                for batch_index, (_, inputs, targets) in enumerate(loader, start=1):
                    inputs = inputs.to(options.device, non_blocking=True)
                    output = convnet.forward_features(inputs, flag=1)["features"]
                    output = convnet.head(output)
                    features.append(output.detach().cpu().float().numpy())
                    labels.append(targets.numpy().astype(np.int64, copy=False))
                    if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
                        print(
                            f"[{protocol}] pretrained {source}: batch {batch_index}/{len(loader)}",
                            flush=True,
                        )
            feature_array = np.concatenate(features)
            label_array = np.concatenate(labels)
            if not np.array_equal(label_array, expected_labels):
                raise RuntimeError(f"{protocol} pretrained {source} label order mismatch")
            if feature_array.shape[0] != expected_ids.size:
                raise RuntimeError(f"{protocol} pretrained {source} sample count mismatch")
            return feature_array, label_array

        train_features, train_labels = extract("train", expected_train_ids, expected_train_labels)
        test_features, test_labels = extract("test", expected_test_ids, expected_test_labels)
        _atomic_npz(
            cache_path,
            train_features=train_features,
            train_labels=train_labels,
            train_sample_ids=expected_train_ids,
            test_features=test_features,
            test_labels=test_labels,
            test_sample_ids=expected_test_ids,
            class_ids=class_ids,
            representation=np.asarray("pretrained_vit_b16_224_in21k_adapter_zero_output"),
        )
        return cache_path
    finally:
        if learner is not None:
            release_learner(learner)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_cpu)
        if saved_cuda is not None:
            torch.cuda.set_rng_state_all(saved_cuda)


def _select_actual_banks(
    features: np.ndarray,
    labels: np.ndarray,
    class_ids: np.ndarray,
    cap: int | None,
    seed: int,
) -> dict[int, np.ndarray]:
    banks: dict[int, np.ndarray] = {}
    for class_id in class_ids.tolist():
        values = np.asarray(features[labels == class_id], dtype=np.float32)
        if not values.size:
            raise RuntimeError(f"class {class_id} has no actual reference")
        count = values.shape[0] if cap is None else min(values.shape[0], cap)
        if count < values.shape[0]:
            indices = np.sort(np.random.RandomState(seed + class_id * 997).choice(values.shape[0], count, replace=False))
            values = values[indices]
        banks[class_id] = values.copy()
    return banks


def _assemble_banks(banks: dict[int, np.ndarray], class_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for class_id in class_ids.tolist():
        values = np.asarray(banks[class_id], dtype=np.float32)
        features.append(values)
        labels.append(np.full(values.shape[0], class_id, dtype=np.int64))
    return np.concatenate(features), np.concatenate(labels)


def _sample_empirical_gaussian_draws(
    actual_banks: dict[int, np.ndarray],
    class_ids: Iterable[int],
    draws: int,
    jitter: float,
    seed: int,
) -> list[dict[int, np.ndarray]]:
    result: list[dict[int, np.ndarray]] = [dict() for _ in range(draws)]
    for class_id in class_ids:
        values = np.asarray(actual_banks[int(class_id)], dtype=np.float64)
        mean = values.mean(axis=0)
        centered = values - mean
        denominator = np.sqrt(max(values.shape[0] - 1, 1))
        rng = np.random.RandomState(seed + int(class_id) * 1009)
        for draw in range(draws):
            count = values.shape[0]
            if values.shape[0] >= 2:
                low_rank = rng.standard_normal((count, values.shape[0])) @ centered / denominator
            else:
                low_rank = np.zeros((count, values.shape[1]), dtype=np.float64)
            noise = np.sqrt(jitter) * rng.standard_normal((count, values.shape[1]))
            result[draw][int(class_id)] = (mean + low_rank + noise).astype(np.float32)
    return result


def _sample_stored_gaussian_draws(
    means: np.ndarray,
    covariances: np.ndarray,
    counts: dict[int, int],
    draws: int,
    device: torch.device,
    seed: int,
) -> tuple[list[dict[int, np.ndarray]], list[dict[str, Any]]]:
    result: list[dict[int, np.ndarray]] = [dict() for _ in range(draws)]
    repairs: list[dict[str, Any]] = []
    ordered_classes = sorted(counts)
    for class_index, class_id in enumerate(ordered_classes, start=1):
        mean = torch.as_tensor(means[class_id], dtype=torch.float32, device=device)
        covariance = torch.as_tensor(covariances[class_id], dtype=torch.float32, device=device)
        covariance = 0.5 * (covariance + covariance.T)
        added_jitter = 0.0
        identity = torch.eye(covariance.shape[0], dtype=covariance.dtype, device=device)
        for attempt in range(7):
            factor, info = torch.linalg.cholesky_ex(covariance + added_jitter * identity)
            if int(info.item()) == 0:
                break
            added_jitter = 10.0 ** (-7 + attempt)
        else:
            raise RuntimeError(f"stored covariance is not positive definite for class {class_id}")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + class_id * 1013)
        epsilon = torch.randn(
            (draws, counts[class_id], covariance.shape[0]),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        samples = mean[None, None, :] + epsilon @ factor.T
        for draw in range(draws):
            result[draw][class_id] = samples[draw].cpu().numpy()
        repairs.append({"class_id": class_id, "additional_numerical_jitter": added_jitter})
        del covariance, factor, epsilon, samples
        if class_index == 1 or class_index % 20 == 0 or class_index == len(ordered_classes):
            print(
                f"[stored Gaussian] factorized/sampled class {class_index}/{len(ordered_classes)}",
                flush=True,
            )
    return result, repairs


def _classifier_observations(
    features: np.ndarray,
    labels: np.ndarray,
    checkpoint: dict[str, Any],
) -> dict[str, np.ndarray]:
    class_count = int(checkpoint["total_classes"])
    weights = checkpoint_classifier_weights(checkpoint).numpy()[:class_count]
    scale = float(checkpoint["config"].get("scale", 1.0))
    scores = cosine_scores(features, weights, scale)
    rows = np.arange(labels.size)
    rivals = scores.copy()
    rivals[rows, labels] = -np.inf
    rival_class = np.argmax(rivals, axis=1)
    margin = scores[rows, labels] - rivals[rows, rival_class]
    predictions = np.argmax(scores, axis=1)
    return {
        "prediction": predictions,
        "correct": predictions == labels,
        "margin": margin,
        "rival_class": rival_class,
    }


def _support_rows(
    *,
    protocol: str,
    source: str,
    draw: int,
    result: SupportResult,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    partition: list[int],
    reference_counts: dict[int, int],
    classifier: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    task = len(partition)
    known_before = sum(partition[:-1])
    rows: list[dict[str, Any]] = []
    for index, label_value in enumerate(labels.tolist()):
        label = int(label_value)
        count = int(reference_counts[label])
        rows.append({
            "protocol": protocol,
            "task": task,
            "source": source,
            "draw": draw,
            "query_type": "old" if label < known_before else "new",
            "sample_id": str(sample_ids[index]),
            "class_id": label,
            "class_birth_task": birth_task(label, partition),
            "class_age": task - birth_task(label, partition),
            "actual_reference_count": count,
            "reference_count_bin": _reference_bin(count),
            "own_support": float(result.own_support[index]),
            "strongest_rival_support": float(result.rival_support[index]),
            "support_margin": float(result.support_margin[index]),
            "nearest_own_cosine": float(result.nearest_own_cosine[index]),
            "nearest_rival_cosine": float(result.nearest_rival_cosine[index]),
            "support_rival_class": int(result.rival_class[index]),
            "knn_purity": float(result.knn_purity[index]),
            "classifier_correct": bool(classifier["correct"][index]),
            "classifier_prediction": int(classifier["prediction"][index]),
            "classifier_rival_class": int(classifier["rival_class"][index]),
            "classifier_margin": float(classifier["margin"][index]),
            "feature_normalization": "L2 before cosine support",
            "gaussian_sampling_space": "raw then L2 normalized" if source != "A_ACTUAL_CURRENT" else "not applicable",
        })
    return rows


def _aggregate_support_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_specs = (
        ("overall", ()),
        ("query_type", ("query_type",)),
        ("classifier_outcome", ("classifier_correct",)),
        ("class_age", ("class_age",)),
        ("reference_count_bin", ("reference_count_bin",)),
    )
    for source in sorted({str(row["source"]) for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        for draw in sorted({int(row["draw"]) for row in source_rows}):
            draw_rows = [row for row in source_rows if int(row["draw"]) == draw]
            for grouping, keys in group_specs:
                grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
                for row in draw_rows:
                    grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
                for values, members in grouped.items():
                    margins = np.asarray([float(row["support_margin"]) for row in members])
                    own = np.asarray([float(row["own_support"]) for row in members])
                    rival = np.asarray([float(row["strongest_rival_support"]) for row in members])
                    labels = np.asarray([int(row["class_id"]) for row in members])
                    item: dict[str, Any] = {
                        "protocol": members[0]["protocol"],
                        "source": source,
                        "draw": draw,
                        "grouping": grouping,
                        **dict(zip(keys, values)),
                        **distribution_summary(margins, "support_margin_"),
                        **distribution_summary(own, "own_support_"),
                        **distribution_summary(rival, "rival_support_"),
                        "support_margin_class_macro": class_macro(margins, labels),
                    }
                    output.append(item)
    return output


def _run_coverage_source(
    protocol: str,
    checkpoint: dict[str, Any],
    final_values: Any,
    partition: list[int],
    options: CoverageOptions,
) -> tuple[SupportResult, list[dict[str, Any]], dict[str, Any]]:
    output = options.output_root / protocol
    output.mkdir(parents=True, exist_ok=True)
    train_features = np.asarray(final_values["train_features"], dtype=np.float32)
    train_labels = np.asarray(final_values["train_labels"], dtype=np.int64)
    train_ids = np.asarray(final_values["train_sample_ids"])
    test_features = np.asarray(final_values["test_features"], dtype=np.float32)
    test_labels = np.asarray(final_values["test_labels"], dtype=np.int64)
    test_ids = np.asarray(final_values["test_sample_ids"])
    class_ids = np.arange(sum(partition), dtype=np.int64)
    known_before = sum(partition[:-1])
    old_class_ids = class_ids[:known_before]
    actual_banks = _select_actual_banks(
        train_features, train_labels, class_ids, options.bank_cap, options.seed + len(partition) * 1000
    )
    counts = {class_id: int(values.shape[0]) for class_id, values in actual_banks.items()}
    classifier = _classifier_observations(test_features, test_labels, checkpoint)
    actual_features, actual_labels = _assemble_banks(
        {class_id: normalize(values) for class_id, values in actual_banks.items()}, class_ids
    )
    actual_ids = []
    for class_id in class_ids.tolist():
        class_mask = np.flatnonzero(train_labels == class_id)
        selected_values = actual_banks[class_id]
        if selected_values.shape[0] == class_mask.size:
            actual_ids.extend(train_ids[class_mask].tolist())
        else:
            # Test queries never overlap train references.  IDs remain unique
            # provenance labels when a deterministic cap is requested.
            actual_ids.extend([f"capped:{protocol}:{class_id}:{index}" for index in range(selected_values.shape[0])])
    actual_result = cosine_support(
        actual_features,
        actual_labels,
        test_features,
        test_labels,
        class_ids,
        options.support_k,
        options.device,
        reference_ids=actual_ids,
        query_ids=test_ids,
        chunk_size=options.support_chunk_size,
    )
    rows_a = _support_rows(
        protocol=protocol,
        source="A_ACTUAL_CURRENT",
        draw=0,
        result=actual_result,
        labels=test_labels,
        sample_ids=test_ids,
        partition=partition,
        reference_counts=counts,
        classifier=classifier,
    )
    write_csv(output / "query_support_A.csv", rows_a)

    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle_draws = _sample_empirical_gaussian_draws(
        actual_banks, old_class_ids.tolist(), options.gaussian_draws, jitter,
        options.seed + len(partition) * 10000 + 101,
    )
    stored_means = np.asarray(checkpoint["class_means"][:known_before], dtype=np.float32)
    stored_covariances = np.asarray(checkpoint["class_covs"][:known_before].cpu(), dtype=np.float32)
    old_counts = {class_id: counts[class_id] for class_id in old_class_ids.tolist()}
    stored_draws, repairs = _sample_stored_gaussian_draws(
        stored_means,
        stored_covariances,
        old_counts,
        options.gaussian_draws,
        options.device,
        options.seed + len(partition) * 10000 + 202,
    )
    rows_b: list[dict[str, Any]] = []
    rows_c: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    new_actual = {class_id: actual_banks[class_id] for class_id in class_ids[known_before:].tolist()}
    for draw in range(options.gaussian_draws):
        print(
            f"[{protocol}] Gaussian support draw {draw + 1}/{options.gaussian_draws}",
            flush=True,
        )
        bank_b = {**oracle_draws[draw], **new_actual}
        bank_c = {**stored_draws[draw], **new_actual}
        features_b, labels_b = _assemble_banks({key: normalize(value) for key, value in bank_b.items()}, class_ids)
        features_c, labels_c = _assemble_banks({key: normalize(value) for key, value in bank_c.items()}, class_ids)
        if not np.array_equal(labels_b, labels_c) or not np.array_equal(labels_b, actual_labels):
            raise AssertionError("A/B/C matched bank labels differ")
        result_b = cosine_support(
            features_b, labels_b, test_features, test_labels, class_ids,
            options.support_k, options.device, chunk_size=options.support_chunk_size,
        )
        result_c = cosine_support(
            features_c, labels_c, test_features, test_labels, class_ids,
            options.support_k, options.device, chunk_size=options.support_chunk_size,
        )
        draw_rows_b = _support_rows(
            protocol=protocol, source="B_ORACLE_GAUSSIAN", draw=draw,
            result=result_b, labels=test_labels, sample_ids=test_ids,
            partition=partition, reference_counts=counts, classifier=classifier,
        )
        draw_rows_c = _support_rows(
            protocol=protocol, source="C_STRUCTURE_GAUSSIAN", draw=draw,
            result=result_c, labels=test_labels, sample_ids=test_ids,
            partition=partition, reference_counts=counts, classifier=classifier,
        )
        rows_b.extend(draw_rows_b)
        rows_c.extend(draw_rows_c)
        for index, label in enumerate(test_labels.tolist()):
            paired.append({
                "protocol": protocol,
                "task": len(partition),
                "draw": draw,
                "sample_id": str(test_ids[index]),
                "class_id": int(label),
                "query_type": "old" if label < known_before else "new",
                "class_age": len(partition) - birth_task(int(label), partition),
                "actual_reference_count": counts[int(label)],
                "reference_count_bin": _reference_bin(counts[int(label)]),
                "classifier_correct": bool(classifier["correct"][index]),
                "margin_A": float(actual_result.support_margin[index]),
                "margin_B": float(result_b.support_margin[index]),
                "margin_C": float(result_c.support_margin[index]),
                "delta_B_minus_A": float(result_b.support_margin[index] - actual_result.support_margin[index]),
                "delta_C_minus_B": float(result_c.support_margin[index] - result_b.support_margin[index]),
                "own_delta_B_minus_A": float(result_b.own_support[index] - actual_result.own_support[index]),
                "own_delta_C_minus_B": float(result_c.own_support[index] - result_b.own_support[index]),
            })
        del features_b, features_c, result_b, result_c
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output / "query_support_B.csv", rows_b)
    write_csv(output / "query_support_C.csv", rows_c)
    write_csv(output / "paired_support_delta.csv", paired)
    all_rows = rows_a + rows_b + rows_c
    summary_rows = _aggregate_support_rows(all_rows)
    write_csv(output / "support_summary.csv", summary_rows)
    bank_rows = []
    repair_lookup = {int(row["class_id"]): row for row in repairs}
    for class_id in class_ids.tolist():
        values = actual_banks[class_id]
        bank_rows.append({
            "protocol": protocol,
            "class_id": class_id,
            "class_birth_task": birth_task(class_id, partition),
            "class_age": len(partition) - birth_task(class_id, partition),
            "old_or_new": "old" if class_id < known_before else "new",
            "unique_actual_train_count": int(np.sum(train_labels == class_id)),
            "matched_bank_size": counts[class_id],
            "cap": options.bank_cap,
            "replacement_sampling_A": False,
            "gaussian_draws_B_C": options.gaussian_draws if class_id < known_before else 0,
            "oracle_covariance_jitter": jitter if class_id < known_before else None,
            "stored_additional_numerical_jitter": repair_lookup.get(class_id, {}).get("additional_numerical_jitter"),
        })
    write_csv(output / "bank_stats.csv", bank_rows)
    manifest = {
        "status": "complete",
        "matched_bank_size_per_class": True,
        "actual_replacement_sampling": False,
        "gaussian_draws": options.gaussian_draws,
        "oracle_seed_base": options.seed + len(partition) * 10000 + 101,
        "stored_seed_base": options.seed + len(partition) * 10000 + 202,
        "old_new_reported_separately": True,
        "query_count": int(test_labels.size),
        "support_k": options.support_k,
        "normalization": "raw Gaussian sampling followed by feature-wise L2 normalization",
        "test_used_for_selection": False,
    }
    write_json(output / "coverage_manifest.json", manifest)
    return actual_result, rows_a, manifest


def _probe_initial_weights(features: np.ndarray, labels: np.ndarray, class_ids: np.ndarray) -> torch.Tensor:
    weights = []
    for class_id in class_ids.tolist():
        values = features[labels == class_id]
        if not values.size:
            raise RuntimeError(f"probe fit split is empty for class {class_id}")
        weights.append(values.mean(axis=0))
    return torch.as_tensor(normalize(np.stack(weights)), dtype=torch.float32)


def _score_probe(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    scale: float,
    class_ids: np.ndarray,
    known_before: int,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    scores = cosine_scores(features, weights, scale)
    predictions = np.argmax(scores, axis=1)
    rows = np.arange(labels.size)
    rival = scores.copy()
    rival[rows, labels] = -np.inf
    margins = scores[rows, labels] - np.max(rival, axis=1)
    old = labels < known_before
    new = ~old
    return {
        "sample_count": int(labels.size),
        "accuracy": float(np.mean(predictions == labels) * 100.0),
        "class_macro_accuracy": macro_accuracy(predictions, labels),
        "old_accuracy": float(np.mean(predictions[old] == labels[old]) * 100.0) if old.any() else float("nan"),
        "new_accuracy": float(np.mean(predictions[new] == labels[new]) * 100.0) if new.any() else float("nan"),
        "margin_mean": float(margins.mean()),
        "margin_median": float(np.median(margins)),
        "margin_q10": float(np.quantile(margins, 0.10)),
        "margin_q05": float(np.quantile(margins, 0.05)),
        "wrong_margin_magnitude": float(np.mean(-margins[predictions != labels])) if np.any(predictions != labels) else 0.0,
    }, predictions, margins


def _train_identical_probe(
    *,
    protocol: str,
    representation: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    class_ids: np.ndarray,
    partition: list[int],
    options: CoverageOptions,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    normalized_train = normalize(train_features)
    normalized_test = normalize(test_features)
    fit_features = normalized_train[train_indices]
    fit_labels = train_labels[train_indices]
    validation_features = normalized_train[validation_indices]
    validation_labels = train_labels[validation_indices]
    initial_weights = _probe_initial_weights(fit_features, fit_labels, class_ids)
    permutations = _make_epoch_permutations(
        fit_labels, options.probe_epochs, options.seed + len(partition) * 3000 + 77
    )
    scale = 20.0
    candidates: list[tuple[float, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for learning_rate in options.probe_lrs:
        print(
            f"[{protocol}] {representation} cosine probe LR={learning_rate:g}",
            flush=True,
        )
        result = _train_cosine(
            condition=f"{representation}_LR_{learning_rate:g}",
            protocol=protocol,
            initial_weights=initial_weights,
            task_sizes=[int(class_ids.size)],
            train_features=fit_features,
            train_labels=fit_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            scale=scale,
            init_lr=float(learning_rate),
            weight_decay=5e-4,
            max_epochs=options.probe_epochs,
            batch_size=options.probe_batch_size,
            device=options.device,
            permutations=permutations,
            convergence=True,
            min_epochs=min(10, options.probe_epochs),
            patience=min(6, max(2, options.probe_epochs // 3)),
            relative_loss_threshold=1e-4,
        )
        candidates.append((learning_rate, result))
        for row in result.trace:
            trace_rows.append({**row, "representation": representation, "learning_rate_candidate": learning_rate})
    selected_lr, selected = max(
        candidates,
        key=lambda item: (
            item[1].validation_accuracy,
            -item[1].validation_loss,
            -item[0],
        ),
    )
    known_before = sum(partition[:-1])
    split_values: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    for split, features, labels in (
        ("train", fit_features, fit_labels),
        ("validation", validation_features, validation_labels),
        ("test", normalized_test, test_labels),
    ):
        split_values[split] = _score_probe(
            features, labels, selected.weights, scale, class_ids, known_before
        )
    candidate_rows: list[dict[str, Any]] = []
    for learning_rate, result in candidates:
        is_selected = learning_rate == selected_lr
        candidate_rows.append({
            "row_type": "candidate",
            "protocol": protocol,
            "representation": representation,
            "learning_rate": learning_rate,
            "selected": is_selected,
            "selected_epoch": result.selected_epoch,
            "stop_reason": result.stop_reason,
            "fit_accuracy": result.train_accuracy,
            "validation_accuracy": result.validation_accuracy,
            "validation_loss": result.validation_loss,
            "test_accuracy": split_values["test"][0]["accuracy"] if is_selected else None,
            "selection_rule": "max validation accuracy, then min validation loss, then smaller LR",
            "test_used_for_selection": False,
            "class_weighting": "none (identical for both representations)",
        })
    train_metrics = split_values["train"][0]
    validation_metrics = split_values["validation"][0]
    test_metrics, test_predictions, test_margins = split_values["test"]
    selected_row = {
        "row_type": "selected",
        "protocol": protocol,
        "representation": representation,
        "selected_learning_rate": selected_lr,
        "selected_epoch": selected.selected_epoch,
        "probe_type": "normalized cosine head",
        "optimizer": "SGD(momentum=0.9), cosine schedule",
        "weight_decay": 5e-4,
        "scale": scale,
        "class_weighting": "none (identical)",
        "train_accuracy": train_metrics["accuracy"],
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "old_accuracy": test_metrics["old_accuracy"],
        "new_accuracy": test_metrics["new_accuracy"],
        "class_macro_accuracy": test_metrics["class_macro_accuracy"],
        "train_test_accuracy_gap": train_metrics["accuracy"] - test_metrics["accuracy"],
        "validation_test_accuracy_gap": validation_metrics["accuracy"] - test_metrics["accuracy"],
        "train_margin_mean": train_metrics["margin_mean"],
        "validation_margin_mean": validation_metrics["margin_mean"],
        "test_margin_mean": test_metrics["margin_mean"],
        "test_margin_median": test_metrics["margin_median"],
        "test_margin_q10": test_metrics["margin_q10"],
        "test_margin_q05": test_metrics["margin_q05"],
        "wrong_test_margin_magnitude": test_metrics["wrong_margin_magnitude"],
        "same_split_indices": True,
        "same_permutations": True,
        "test_used_for_selection": False,
    }
    return selected_row, candidate_rows, trace_rows, test_predictions, test_margins


def _representation_support_rows(
    *,
    protocol: str,
    representation: str,
    result: SupportResult,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    predictions: np.ndarray,
    probe_margins: np.ndarray,
    partition: list[int],
    reference_counts: dict[int, int],
) -> list[dict[str, Any]]:
    task = len(partition)
    known_before = sum(partition[:-1])
    rows = []
    for index, label_value in enumerate(labels.tolist()):
        label = int(label_value)
        rows.append({
            "protocol": protocol,
            "task": task,
            "representation": representation,
            "sample_id": str(sample_ids[index]),
            "class_id": label,
            "query_type": "old" if label < known_before else "new",
            "class_birth_task": birth_task(label, partition),
            "class_age": task - birth_task(label, partition),
            "actual_reference_count": reference_counts[label],
            "reference_count_bin": _reference_bin(reference_counts[label]),
            "own_support": float(result.own_support[index]),
            "strongest_rival_support": float(result.rival_support[index]),
            "support_margin": float(result.support_margin[index]),
            "nearest_own_cosine": float(result.nearest_own_cosine[index]),
            "nearest_rival_cosine": float(result.nearest_rival_cosine[index]),
            "knn_purity": float(result.knn_purity[index]),
            "probe_correct": bool(predictions[index] == label),
            "probe_prediction": int(predictions[index]),
            "probe_margin": float(probe_margins[index]),
        })
    return rows


def _run_representation_comparison(
    protocol: str,
    checkpoint: dict[str, Any],
    final_values: Any,
    pretrained_values: Any,
    final_support: SupportResult,
    partition: list[int],
    options: CoverageOptions,
) -> dict[str, Any]:
    output = options.output_root / protocol
    for key in ("train_labels", "test_labels", "train_sample_ids", "test_sample_ids"):
        if not np.array_equal(final_values[key], pretrained_values[key]):
            raise RuntimeError(f"{protocol}: pretrained/final {key} mismatch")
    train_labels = np.asarray(final_values["train_labels"], dtype=np.int64)
    test_labels = np.asarray(final_values["test_labels"], dtype=np.int64)
    train_ids = np.asarray(final_values["train_sample_ids"])
    test_ids = np.asarray(final_values["test_sample_ids"])
    final_train = np.asarray(final_values["train_features"], dtype=np.float32)
    final_test = np.asarray(final_values["test_features"], dtype=np.float32)
    pretrained_train = np.asarray(pretrained_values["train_features"], dtype=np.float32)
    pretrained_test = np.asarray(pretrained_values["test_features"], dtype=np.float32)
    class_ids = np.arange(sum(partition), dtype=np.int64)
    train_indices, validation_indices, split_rows, no_validation = _validation_indices(
        train_labels, class_ids, options.seed + len(partition) * 1700
    )
    for row in split_rows:
        row.update({"protocol": protocol, "seed": options.seed + len(partition) * 1700})
    write_csv(output / "probe_split_manifest.csv", split_rows)

    probe_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    margins: dict[str, np.ndarray] = {}
    for representation, train_features, test_features in (
        ("PRETRAINED", pretrained_train, pretrained_test),
        ("STRUCTURE_FINAL", final_train, final_test),
    ):
        selected, candidates, traces, test_predictions, test_margins = _train_identical_probe(
            protocol=protocol,
            representation=representation,
            train_features=train_features,
            train_labels=train_labels,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_features=test_features,
            test_labels=test_labels,
            class_ids=class_ids,
            partition=partition,
            options=options,
        )
        probe_rows.append(selected)
        candidate_rows.extend(candidates)
        trace_rows.extend(traces)
        predictions[representation] = test_predictions
        margins[representation] = test_margins

    pretrained_banks = _select_actual_banks(
        pretrained_train, train_labels, class_ids, options.bank_cap,
        options.seed + len(partition) * 1000,
    )
    pretrained_bank_features, pretrained_bank_labels = _assemble_banks(
        {class_id: normalize(values) for class_id, values in pretrained_banks.items()}, class_ids
    )
    pretrained_result = cosine_support(
        pretrained_bank_features,
        pretrained_bank_labels,
        pretrained_test,
        test_labels,
        class_ids,
        options.support_k,
        options.device,
        chunk_size=options.support_chunk_size,
    )
    counts = {class_id: int(np.sum(train_labels == class_id)) if options.bank_cap is None else min(int(np.sum(train_labels == class_id)), options.bank_cap) for class_id in class_ids.tolist()}
    pretrained_rows = _representation_support_rows(
        protocol=protocol, representation="PRETRAINED", result=pretrained_result,
        labels=test_labels, sample_ids=test_ids, predictions=predictions["PRETRAINED"],
        probe_margins=margins["PRETRAINED"], partition=partition, reference_counts=counts,
    )
    final_rows = _representation_support_rows(
        protocol=protocol, representation="STRUCTURE_FINAL", result=final_support,
        labels=test_labels, sample_ids=test_ids, predictions=predictions["STRUCTURE_FINAL"],
        probe_margins=margins["STRUCTURE_FINAL"], partition=partition, reference_counts=counts,
    )
    write_csv(output / "pretrained_support.csv", pretrained_rows)
    write_csv(output / "final_support.csv", final_rows)
    paired_rows = []
    for index, label in enumerate(test_labels.tolist()):
        paired_rows.append({
            "protocol": protocol,
            "sample_id": str(test_ids[index]),
            "class_id": int(label),
            "query_type": "old" if label < sum(partition[:-1]) else "new",
            "pretrained_support_margin": float(pretrained_result.support_margin[index]),
            "final_support_margin": float(final_support.support_margin[index]),
            "delta_final_minus_pretrained": float(final_support.support_margin[index] - pretrained_result.support_margin[index]),
            "pretrained_probe_correct": bool(predictions["PRETRAINED"][index] == label),
            "final_probe_correct": bool(predictions["STRUCTURE_FINAL"][index] == label),
            "pretrained_probe_margin": float(margins["PRETRAINED"][index]),
            "final_probe_margin": float(margins["STRUCTURE_FINAL"][index]),
        })
    write_csv(output / "pretrained_vs_final_paired.csv", paired_rows)

    for row in probe_rows:
        representation_rows = pretrained_rows if row["representation"] == "PRETRAINED" else final_rows
        support_values = np.asarray([float(item["support_margin"]) for item in representation_rows])
        support_labels = np.asarray([int(item["class_id"]) for item in representation_rows])
        correct_mask = np.asarray([bool(item["probe_correct"]) for item in representation_rows])
        row.update({
            "support_margin_micro": float(support_values.mean()),
            "support_margin_class_macro": class_macro(support_values, support_labels),
            "correct_query_support": float(support_values[correct_mask].mean()) if correct_mask.any() else float("nan"),
            "wrong_query_support": float(support_values[~correct_mask].mean()) if (~correct_mask).any() else float("nan"),
        })
    write_csv(output / "probe_results.csv", probe_rows + candidate_rows)
    write_csv(output / "probe_training_trace.csv", trace_rows)
    manifest = {
        "status": "complete",
        "representations": ["PRETRAINED", "STRUCTURE_FINAL"],
        "same_samples": True,
        "same_train_validation_split": True,
        "same_probe_type": True,
        "same_lr_grid": list(options.probe_lrs),
        "same_training_budget": options.probe_epochs,
        "same_epoch_permutations": True,
        "validation_classes_without_holdout": no_validation,
        "test_used_for_selection": False,
        "pretrained_initialization": "ViT-B/16 ImageNet-21K with zero-output fresh adapters",
    }
    write_json(output / "representation_manifest.json", manifest)
    return manifest


def _protocol_complete(output: Path) -> bool:
    path = output / "protocol_manifest.json"
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = (
        "bank_stats.csv",
        "query_support_A.csv",
        "query_support_B.csv",
        "query_support_C.csv",
        "paired_support_delta.csv",
        "pretrained_support.csv",
        "final_support.csv",
        "probe_results.csv",
    )
    return value.get("status") == "complete" and all((output / filename).is_file() for filename in required)


def _stage_complete(output: Path, manifest_name: str, required: tuple[str, ...]) -> bool:
    path = output / manifest_name
    if not path.is_file() or not all((output / filename).is_file() for filename in required):
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def _support_result_from_rows(rows: list[dict[str, str]]) -> SupportResult:
    return SupportResult(
        own_support=np.asarray([float(row["own_support"]) for row in rows], dtype=np.float32),
        rival_support=np.asarray([float(row["strongest_rival_support"]) for row in rows], dtype=np.float32),
        support_margin=np.asarray([float(row["support_margin"]) for row in rows], dtype=np.float32),
        nearest_own_cosine=np.asarray([float(row["nearest_own_cosine"]) for row in rows], dtype=np.float32),
        nearest_rival_cosine=np.asarray([float(row["nearest_rival_cosine"]) for row in rows], dtype=np.float32),
        rival_class=np.asarray([int(row["support_rival_class"]) for row in rows], dtype=np.int64),
        knn_purity=np.asarray([float(row["knn_purity"]) for row in rows], dtype=np.float32),
    )


def _run_protocol(protocol: str, options: CoverageOptions) -> dict[str, Any]:
    output = options.output_root / protocol
    if _protocol_complete(output) and not options.force:
        value = json.loads((output / "protocol_manifest.json").read_text(encoding="utf-8"))
        print(f"[{protocol}] complete outputs found; skip", flush=True)
        return value
    started = time.time()
    config = protocol_config(protocol, DEFAULT_DATA_ROOTS[protocol])
    partition = list(map(int, config["task_partition"]))
    class_ids = np.arange(sum(partition), dtype=np.int64)
    checkpoint_path = _checkpoint_path(protocol, options.seed)
    final_cache_path = _final_cache_path(protocol, options.seed)
    before = _edge_fingerprint(checkpoint_path)
    checkpoint = torch_load(checkpoint_path)
    data_manager, _ = build_data_manager(protocol, DEFAULT_DATA_ROOTS[protocol], options.seed)
    print(f"[{protocol}] pretrained frozen feature extraction/cache validation", flush=True)
    pretrained_cache_path = _extract_pretrained_features(
        protocol, checkpoint, data_manager, class_ids, options
    )
    final_values = np.load(final_cache_path, allow_pickle=False)
    pretrained_values = np.load(pretrained_cache_path, allow_pickle=False)
    coverage_required = (
        "bank_stats.csv", "query_support_A.csv", "query_support_B.csv",
        "query_support_C.csv", "paired_support_delta.csv", "support_summary.csv",
    )
    if _stage_complete(output, "coverage_manifest.json", coverage_required) and not options.force:
        rows_a = read_csv(output / "query_support_A.csv")
        if [row["sample_id"] for row in rows_a] != final_values["test_sample_ids"].astype(str).tolist():
            raise RuntimeError(f"{protocol}: resumed A-support sample IDs differ from final cache")
        final_support = _support_result_from_rows(rows_a)
        coverage_manifest = json.loads((output / "coverage_manifest.json").read_text(encoding="utf-8"))
        print(f"[{protocol}] coverage stage complete; reuse", flush=True)
    else:
        print(f"[{protocol}] A/B/C matched-size coverage diagnostic", flush=True)
        final_support, _, coverage_manifest = _run_coverage_source(
            protocol, checkpoint, final_values, partition, options
        )
    representation_required = (
        "pretrained_support.csv", "final_support.csv", "pretrained_vs_final_paired.csv",
        "probe_results.csv", "probe_training_trace.csv", "probe_split_manifest.csv",
    )
    if _stage_complete(output, "representation_manifest.json", representation_required) and not options.force:
        representation_manifest = json.loads((output / "representation_manifest.json").read_text(encoding="utf-8"))
        print(f"[{protocol}] representation stage complete; reuse", flush=True)
    else:
        print(f"[{protocol}] pretrained-vs-final support and identical cosine probes", flush=True)
        representation_manifest = _run_representation_comparison(
            protocol, checkpoint, final_values, pretrained_values, final_support,
            partition, options,
        )
    after = _edge_fingerprint(checkpoint_path)
    if before != after:
        raise RuntimeError(f"{protocol}: checkpoint fingerprint changed")
    manifest = {
        "status": "complete",
        "protocol": protocol,
        "task": len(partition),
        "checkpoint_fingerprint_before": before,
        "checkpoint_fingerprint_after": after,
        "checkpoint_unchanged": True,
        "final_feature_cache": str(final_cache_path),
        "pretrained_feature_cache": str(pretrained_cache_path),
        "coverage": coverage_manifest,
        "representation": representation_manifest,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output / "protocol_manifest.json", manifest)
    del checkpoint, final_values, pretrained_values, final_support, data_manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def run_diagnostic(options: CoverageOptions) -> dict[str, Any]:
    started = time.time()
    options.output_root.mkdir(parents=True, exist_ok=True)
    selected = tuple(protocol.upper() for protocol in options.protocols)
    invalid = [protocol for protocol in selected if protocol not in ALL_PROTOCOLS]
    if invalid:
        raise ValueError(f"unknown protocols: {invalid}; choose from {ALL_PROTOCOLS}")
    audit = write_audit(options)
    if audit["missing"]:
        raise FileNotFoundError("MISSING INPUTS:\n" + "\n".join(audit["missing"]))
    running = {
        "status": "running",
        "seed": options.seed,
        "physical_gpu": options.physical_gpu,
        "process_device": str(options.device),
        "protocols": list(selected),
        "gaussian_draws": options.gaussian_draws,
        "support_k": options.support_k,
        "bank_cap": options.bank_cap,
        "probe_lrs": list(options.probe_lrs),
        "probe_epochs": options.probe_epochs,
        "started_unix": started,
    }
    write_json(options.output_root / "run_manifest.json", running)
    protocol_results: dict[str, Any] = {}
    for protocol in selected:
        print(f"[{protocol}] sequential diagnostic start", flush=True)
        protocol_results[protocol] = _run_protocol(protocol, options)
        running["completed_protocols"] = list(protocol_results)
        running["protocol_results"] = protocol_results
        write_json(options.output_root / "run_manifest.json", running)
        print(f"[{protocol}] complete", flush=True)
    from .report import generate_report

    reports = generate_report(options.output_root, selected, options.gaussian_draws)
    manifest = {
        **running,
        "status": "complete",
        "protocol_results": protocol_results,
        "reports": reports,
        "candidate_count_stress_test": "not run (nice-to-have; must-have diagnostics prioritized)",
        "new_method_implemented": False,
        "structure_training_rerun": False,
        "test_used_for_hyperparameter_selection": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(options.output_root / "run_manifest.json", manifest)
    return manifest
