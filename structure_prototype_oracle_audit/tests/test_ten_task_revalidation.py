from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from ten_task_revalidation.metrics import (
    aggregate,
    directional_rows,
    prototype_rows_from_snapshots,
)
from ten_task_revalidation.pipeline import RevalidationOptions, _generalization_analysis


def _snapshot(task: int, oracle: np.ndarray, stored: np.ndarray) -> dict[str, np.ndarray]:
    classes = oracle.shape[0]
    return {
        "class_ids": np.arange(classes, dtype=np.int64),
        "oracle_means": oracle.astype(np.float32),
        "structure_means": stored.astype(np.float32),
        "birth_tasks": np.asarray([1] + [task] * (classes - 1), dtype=np.int64),
    }


def test_tracking_temporal_and_interclass_are_distinct_metrics():
    first = _snapshot(
        1,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    second = _snapshot(
        2,
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
        np.asarray([[0.8, 0.2], [0.2, 0.8]]),
    )
    tracking, temporal, interclass = prototype_rows_from_snapshots("SYN", [first, second])
    old_t2 = next(row for row in tracking if row["task"] == 2 and row["class_id"] == 0)
    drift_t2 = next(row for row in temporal if row["task"] == 2 and row["class_id"] == 0)
    rival_t2 = next(
        row for row in interclass
        if row["task"] == 2 and row["class_id"] == 0 and row["definition"] == "oracle"
    )
    assert np.isclose(old_t2["tracking_similarity"], 1.0)
    assert drift_t2["oracle_temporal_similarity"] < 1.0
    assert 0.0 < rival_t2["nearest_rival_similarity"] < 1.0


def test_class_age_aggregation_preserves_protocol_and_old_flag():
    rows = [
        {"protocol": "IA10", "class_age": 2, "is_old": True, "x": 1.0},
        {"protocol": "IA10", "class_age": 2, "is_old": True, "x": 3.0},
    ]
    result = aggregate(rows, ("protocol", "class_age", "is_old"), ("x",))
    assert len(result) == 1
    assert result[0]["x_mean"] == 2.0
    assert result[0]["class_count"] == 2


def test_error_concentration_and_direction_are_counted():
    labels = np.asarray([0, 0, 0, 1, 1, 2])
    predictions = np.asarray([0, 1, 1, 1, 0, 2])
    rows, concentration = directional_rows("SYN", 2, labels, predictions, known_before=2)
    assert sum(row["error_count"] for row in rows) == 3
    assert concentration["total_errors"] == 3
    assert np.isclose(concentration["top10_error_share"], 1.0)
    assert {row["pair_type"] for row in rows} == {"old-old"}


def test_generalization_geometry_uses_train_only_reference_and_cpu_knn():
    rng = np.random.RandomState(11)
    train0 = np.asarray([1.0, 0.0]) + 0.05 * rng.normal(size=(12, 2))
    train1 = np.asarray([0.0, 1.0]) + 0.05 * rng.normal(size=(12, 2))
    test0 = np.asarray([1.0, 0.0]) + 0.05 * rng.normal(size=(4, 2))
    test1 = np.asarray([0.0, 1.0]) + 0.05 * rng.normal(size=(4, 2))
    train = np.vstack([train0, train1]).astype(np.float32)
    test = np.vstack([test0, test1]).astype(np.float32)
    train_labels = np.repeat([0, 1], 12).astype(np.int64)
    test_labels = np.repeat([0, 1], 4).astype(np.int64)
    checkpoint = {
        "network_state_dict": {
            "fc.heads.0.weight": torch.tensor([[1.0, 0.0]]),
            "fc.heads.1.weight": torch.tensor([[0.0, 1.0]]),
        }
    }
    with TemporaryDirectory(prefix="ten_task_geometry_test_") as directory:
        cache = Path(directory) / "features.npz"
        np.savez_compressed(
            cache,
            train_features=train,
            train_labels=train_labels,
            train_sample_ids=np.asarray([f"train:{i}" for i in range(train.shape[0])]),
            test_features=test,
            test_labels=test_labels,
            test_sample_ids=np.asarray([f"test:{i}" for i in range(test.shape[0])]),
        )
        options = RevalidationOptions(
            output_root=Path(directory), device=torch.device("cpu"), physical_gpu="synthetic"
        )
        margins, purities, geometry = _generalization_analysis(
            "SYN", 2, cache, checkpoint, [1, 1], options
        )
    assert {row["split"] for row in margins} == {"train", "validation", "test"}
    assert {row["split"] for row in purities} == {"validation", "test"}
    assert geometry
    assert all(row["own_class_neighborhood_purity_mean"] >= 0.8 for row in geometry)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} ten-task revalidation tests")
