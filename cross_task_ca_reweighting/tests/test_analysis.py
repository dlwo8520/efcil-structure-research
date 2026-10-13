"""Tiny CPU-only paired-count invariant (not a pilot experiment)."""
import tempfile
from pathlib import Path

import numpy as np

from analysis import make_tables


def _scores(predicted: list[int], classes: int) -> np.ndarray:
    values = np.full((len(predicted), classes), -1.0, dtype=np.float32)
    values[np.arange(len(predicted)), predicted] = 1.0
    return values


def test_paired_repair_harm_and_error_partition():
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    ids = np.asarray([f"test:{i}" for i in range(len(labels))])
    baseline = _scores([2, 0, 1, 1, 2, 2, 3, 3], 4)
    control = baseline.copy()
    intervention = _scores([0, 0, 1, 0, 2, 2, 3, 3], 4)
    b1 = np.asarray([True, False, False, False, False, False, False, False])
    with tempfile.TemporaryDirectory() as temporary:
        result = make_tables("TOY", ids, labels, control, intervention, baseline,
                             b1, [2, 2], Path(temporary))
    assert result["overall"]["repair_n"] == 1
    assert result["overall"]["harm_n"] == 1
    assert result["overall"]["net_gain_n"] == 0
    assert result["b1"]["repair_n"] == 1
    for name in ("control", "intervention"):
        branch = result["branch"][name]
        assert branch["same_task_error_n"] + branch["cross_task_error_n"] == 8 - branch["correct_n"]
