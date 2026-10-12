from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from pair_boundary_risk.pipeline import (
    BoundaryRiskOptions,
    _pair_actual_rows,
    _ranking_analysis,
    _risk_rows,
    _transition_type,
    boundary_sign_equivalence,
)


def test_boundary_sign_equivalence():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(17, 6)).astype(np.float32)
    weights = rng.normal(size=(4, 6)).astype(np.float32)
    weights /= np.linalg.norm(weights, axis=1, keepdims=True)
    labels = rng.randint(0, 4, size=17)
    result = boundary_sign_equivalence(features, weights, labels)
    assert result["status"] == "PASS"
    assert result["pairwise_sign_mismatch_count"] == 0


def test_transition_labels_keep_new_error_and_destination_change_separate():
    assert _transition_type(1, 1, 1) == "correct_to_correct"
    assert _transition_type(1, 1, 2) == "correct_to_wrong_B"
    assert _transition_type(1, 2, 1) == "wrong_B_to_correct"
    assert _transition_type(1, 2, 2) == "wrong_B_to_wrong_B"
    assert _transition_type(1, 2, 3) == "wrong_C_to_wrong_B"


def test_directional_c_and_b_are_distinct():
    labels = np.asarray([0, 0, 0, 1, 1, 2, 2])
    scores_t4 = np.asarray(
        [[3, 1], [2, 1], [2, 1], [1, 3], [1, 3], [-2, -1], [-2, -1]],
        dtype=np.float32,
    )
    # For true class 0, competitor 1 beats class 0 for sample 0 but class 0 still
    # wins multiclass only in this two-class fixture, so add a third competitor.
    scores_t4 = np.pad(scores_t4, ((0, 0), (0, 1)), constant_values=-3)
    scores_t4[5:, 2] = 3
    scores_t5 = scores_t4.copy()
    scores_t5[0] = [1.0, 2.0, 3.0]  # B_01 violation, final prediction 2.
    pred4 = scores_t4.argmax(1)
    pred5 = scores_t5.argmax(1)
    metrics, margins = _pair_actual_rows(
        labels, scores_t4, scores_t5, pred4, pred5,
        labels, [1, 1, 1], known_before=2,
    )
    row = next(r for r in metrics if r["true_class"] == 0 and r["competitor_class"] == 1)
    assert row["B_t5"] == 1 / 3
    assert row["C_t5"] == 0
    margin = next(r for r in margins if r["true_class"] == 0 and r["competitor_class"] == 1)
    assert margin["t5_negative_fraction"] == 1 / 3


def test_oracle_ranking_recovers_synthetic_target():
    rows = []
    for index in range(20):
        risky = index < 5
        rows.append(
            {
                "B_t5": float(risky),
                "delta_B": float(risky),
                "correct_to_wrong_j_rate": float(risky),
                "boundary_violation_count_t5": int(risky),
                "positive_boundary_degradation_count": int(risky),
                "correct_to_wrong_j": int(risky),
                "prototype_similarity": 0.0,
                "gamma_oracle": float(index),
                "gamma_stored": float(index),
            }
        )
    summary, recall, captured, corr = _ranking_analysis(
        rows, (5, 10), seed=1, random_trials=20
    )
    oracle = next(
        row for row in recall
        if row["target"] == "B_POSITIVE_DELTA_B"
        and row["ranking"] == "ORACLE_GAMMA" and row["K"] == 5
    )
    assert oracle["recall_at_k"] == 1.0


def test_stored_and_oracle_risk_share_raw_coordinate():
    features = np.asarray(
        [[2.0, 0.0], [1.8, 0.1], [0.0, 2.0], [0.1, 1.8]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1])
    weights = np.eye(2, dtype=np.float32)
    scores = 20.0 * features / np.linalg.norm(features, axis=1, keepdims=True)
    predictions = scores.argmax(1)
    metrics, margins = _pair_actual_rows(
        labels, scores, scores, predictions, predictions, labels, [1, 1], known_before=1
    )
    options = BoundaryRiskOptions(
        protocol="SYNTHETIC",
        t4_checkpoint=Path("t4"),
        t5_checkpoint=Path("t5"),
        t4_feature_cache=Path("f4"),
        t5_feature_cache=Path("f5"),
        output_dir=Path("out"),
        device=torch.device("cpu"),
    )
    rows = _risk_rows(
        options,
        features,
        labels,
        weights,
        np.asarray([[1.9, 0.05], [0.05, 1.9]]),
        [torch.eye(2) * 0.1, torch.eye(2) * 0.1],
        1e-3,
        metrics,
        margins,
    )
    assert len(rows) == 2
    assert all(np.isfinite(row["gamma_oracle"]) for row in rows)
    assert all(np.isfinite(row["gamma_stored"]) for row in rows)
    assert rows[0]["prototype_similarity"] == rows[1]["prototype_similarity"]


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS all {len(tests)} pair-boundary tests")
