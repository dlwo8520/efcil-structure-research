from __future__ import annotations

import numpy as np
import torch

from coverage_generalization.metrics import class_macro, cosine_support, normalize
from coverage_generalization.pipeline import (
    _assemble_banks,
    _sample_empirical_gaussian_draws,
    _sample_stored_gaussian_draws,
    _select_actual_banks,
)


def test_cosine_support_uses_class_conditional_top_k_and_self_exclusion() -> None:
    reference = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    ids = np.asarray(["a", "b", "c", "d"])
    result = cosine_support(
        reference,
        labels,
        reference[[0]],
        labels[[0]],
        np.asarray([0, 1]),
        k=2,
        device=torch.device("cpu"),
        reference_ids=ids,
        query_ids=ids[[0]],
        chunk_size=1,
    )
    expected_own = float(normalize(reference[[0]]) @ normalize(reference[[1]]).T)
    assert np.isclose(result.own_support[0], expected_own)
    assert result.support_margin[0] > 0
    assert result.knn_purity[0] == 0.5


def test_matched_actual_bank_has_no_replacement_and_respects_cap() -> None:
    features = np.arange(30, dtype=np.float32).reshape(10, 3)
    labels = np.asarray([0] * 4 + [1] * 6)
    banks = _select_actual_banks(features, labels, np.asarray([0, 1]), cap=3, seed=19)
    assert banks[0].shape[0] == 3
    assert banks[1].shape[0] == 3
    assert len({tuple(row) for row in banks[0]}) == 3
    assembled, assembled_labels = _assemble_banks(banks, np.asarray([0, 1]))
    assert assembled.shape == (6, 3)
    assert np.array_equal(np.bincount(assembled_labels), [3, 3])


def test_empirical_gaussian_draws_are_reproducible_and_matched_size() -> None:
    banks = {
        0: np.asarray([[1.0, 0.0], [0.8, 0.2], [1.2, -0.2]], dtype=np.float32),
        1: np.asarray([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
    }
    first = _sample_empirical_gaussian_draws(banks, [0, 1], 3, 1e-3, 11)
    second = _sample_empirical_gaussian_draws(banks, [0, 1], 3, 1e-3, 11)
    assert len(first) == 3
    for draw in range(3):
        for class_id in (0, 1):
            assert first[draw][class_id].shape == banks[class_id].shape
            assert np.array_equal(first[draw][class_id], second[draw][class_id])


def test_class_macro_is_not_sample_weighted() -> None:
    values = np.asarray([1.0, 1.0, 1.0, 0.0])
    labels = np.asarray([0, 0, 0, 1])
    assert class_macro(values, labels) == 0.5


def test_stored_gaussian_draws_are_deterministic_and_matched_size() -> None:
    means = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    covariances = np.asarray([np.eye(2) * 0.1, np.eye(2) * 0.2], dtype=np.float32)
    counts = {0: 3, 1: 2}
    first, repairs = _sample_stored_gaussian_draws(
        means, covariances, counts, 3, torch.device("cpu"), 17
    )
    second, _ = _sample_stored_gaussian_draws(
        means, covariances, counts, 3, torch.device("cpu"), 17
    )
    assert len(first) == 3
    assert all(row["additional_numerical_jitter"] == 0.0 for row in repairs)
    for draw in range(3):
        assert first[draw][0].shape == (3, 2)
        assert first[draw][1].shape == (2, 2)
        assert np.array_equal(first[draw][0], second[draw][0])
