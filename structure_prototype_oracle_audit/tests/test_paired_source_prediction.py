from __future__ import annotations

import numpy as np

from paired_source_prediction.report import _comparison


def test_repair_harm_regression_and_subgroups() -> None:
    rows = []
    for index, (a, b, c) in enumerate(((1, 0, 0), (0, 1, 0), (1, 1, 1), (0, 0, 1), (1, 0, 1))):
        rows.append({
            "sample_id": f"s{index}", "true_label": index % 2, "class_id": index % 2,
            "old_or_new": "old" if index < 3 else "new",
            "correct_A": bool(a), "correct_B": bool(b), "correct_C": bool(c),
            "pred_A": 0 if a else 2, "pred_B": 0 if b else 2, "pred_C": 0 if c else 2,
            "support_A": .2 + index * .1, "support_B": -.1 + index * .1,
            "support_C": -.2 + index * .1,
            "delta_BA_support": -.3, "delta_CB_support": -.1,
            "signflip_AB": index == 0, "signflip_BC": index == 1,
        })
    ab = _comparison(rows, "A", "B", 100, 13)
    bc = _comparison(rows, "B", "C", 100, 14)
    assert ab["main"][0]["regression_check"] == "PASS"
    assert ab["main"][0]["repair_count"] == 2
    assert ab["main"][0]["harm_count"] == 1
    assert np.isclose(ab["main"][0]["net_gain_pp"], 20.)
    assert bc["main"][0]["regression_check"] == "PASS"
    assert ab["signflip"][1]["n"] == 1
