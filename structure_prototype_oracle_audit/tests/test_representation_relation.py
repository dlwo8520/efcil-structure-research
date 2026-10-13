from __future__ import annotations

import numpy as np
import torch

from representation_relation.knn import _vote, cosine_knn, purity
from representation_relation.pipeline import (
    _directional_confusions,
    _focused_pairs,
    _linear_equivalence,
    _pair_margin_rows,
    _validation_indices,
)
from representation_relation.linear import train_linear_lbfgs


def test_cosine_to_linear_conversion_is_exact_within_float_tolerance():
    rng = np.random.RandomState(3)
    features = rng.normal(size=(31, 9)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    weights = rng.normal(size=(7, 9)).astype(np.float32)
    audit, converted, cosine_logits = _linear_equivalence(features, weights, 20.0)
    assert audit["status"] == "PASS"
    assert audit["prediction_mismatch_count"] == 0
    np.testing.assert_allclose(features @ converted.T, cosine_logits, atol=1e-5, rtol=0)


def test_original_sample_self_match_and_replacement_duplicate_are_excluded():
    reference_features = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32
    )
    reference_labels = np.asarray([0, 0, 1, 2], dtype=np.int64)
    result = cosine_knn(
        reference_features,
        reference_labels,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        2,
        torch.device("cpu"),
        reference_ids=["same", "same", "other-1", "other-2"],
        query_ids=["same"],
    )
    assert set(result.neighbor_indices[0].tolist()) == {2, 3}
    assert 0 not in result.neighbor_indices[0]
    assert 1 not in result.neighbor_indices[0]


def test_vote_is_deterministic_for_count_and_similarity_ties():
    labels = np.asarray([[2, 1, 2, 1], [3, 4, 3, 4]])
    similarities = np.asarray([[0.9, 0.8, 0.7, 0.6], [0.8, 0.9, 0.7, 0.6]])
    # Row 0 chooses 2 by similarity sum; row 1 is an exact sum tie and chooses low ID 3.
    np.testing.assert_array_equal(_vote(labels, similarities), np.asarray([2, 3]))


def test_validation_split_is_original_index_level_and_disjoint():
    labels = np.repeat(np.arange(3), [10, 7, 4]).astype(np.int64)
    train, validation, rows, no_validation = _validation_indices(
        labels, np.arange(3), seed=1993
    )
    assert not set(train).intersection(validation)
    assert sorted(np.concatenate([train, validation]).tolist()) == list(range(labels.size))
    assert [row["validation_samples"] for row in rows] == [2, 1, 0]
    assert no_validation == [2]


def test_directional_pairs_and_margin_are_not_symmetrized():
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    predictions = np.asarray([1, 1, 0, 0, 1, 1])
    rows = _directional_confusions(
        predictions, labels, labels, partition=[1, 1], known_before=1
    )
    mapping = {(row["true_class"], row["predicted_class"]): row for row in rows}
    assert mapping[(0, 1)]["confusion_count"] == 2
    assert mapping[(1, 0)]["confusion_count"] == 1
    assert mapping[(0, 1)]["pair_type"] == "old-new"
    assert mapping[(1, 0)]["pair_type"] == "new-old"
    focused = _focused_pairs(rows, global_k=2, conditional_k=2)
    scores = np.asarray(
        [[2.0, 1.0], [0.0, 1.0], [3.0, 2.0], [1.0, 0.0], [0.0, 2.0], [0.0, 3.0]]
    )
    margin_rows = _pair_margin_rows(
        focused,
        {"train": scores, "validation": scores, "test": scores},
        {"train": labels, "validation": labels, "test": labels},
    )
    row = next(
        item
        for item in margin_rows
        if item["true_class"] == 0
        and item["competitor_class"] == 1
        and item["split"] == "test"
    )
    assert row["negative_fraction"] == 1.0 / 3.0
    assert row["num_samples"] == 3


def test_purity_uses_query_true_label():
    neighbors = np.asarray([[1, 1, 2], [0, 1, 0]])
    labels = np.asarray([1, 1])
    np.testing.assert_allclose(purity(neighbors, labels), [2 / 3, 1 / 3])


def test_linear_sanity_solver_fits_separable_normalized_features_on_cpu():
    rng = np.random.RandomState(11)
    centers = np.eye(3, dtype=np.float32)
    features = np.concatenate(
        [centers[class_id] + 0.03 * rng.normal(size=(20, 3)) for class_id in range(3)]
    ).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    labels = np.repeat(np.arange(3), 20).astype(np.int64)
    result = train_linear_lbfgs(
        initial_weights=centers * 20.0,
        train_features=features[:48],
        train_labels=labels[:48],
        validation_features=features[48:54],
        validation_labels=labels[48:54],
        test_features=features[54:],
        test_labels=labels[54:],
        l2=5e-5,
        device=torch.device("cpu"),
        max_iter=30,
    )
    assert result.train_micro == 100.0
    assert np.isfinite(result.objective)
    assert result.closure_count > 0


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} representation-relation synthetic tests")
