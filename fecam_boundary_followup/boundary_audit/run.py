from __future__ import annotations

import numpy as np
import torch

from common import ROOT, fixed_scores, inputs, write_csv, write_json
from coverage_generalization.pipeline import (
    _sample_empirical_gaussian_draws,
    _sample_stored_gaussian_draws,
    _select_actual_banks,
)


def _margin(features: np.ndarray, class_id: int, checkpoint: dict) -> np.ndarray:
    scores = fixed_scores(features, checkpoint)
    own = scores[:, class_id].copy()
    scores[:, class_id] = -np.inf
    return own - scores.max(axis=1)


def _row(protocol: str, source: str, draw: int, class_id: int, values: np.ndarray) -> dict:
    return {
        "protocol": protocol, "source": source, "draw": draw, "class_id": class_id,
        "count": len(values), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
        "q10": float(np.quantile(values, 0.10)), "q05": float(np.quantile(values, 0.05)),
        "negative_fraction": float(np.mean(values < 0)),
    }


def run(protocol: str, device: torch.device, seed: int = 1993, draws: int = 3) -> dict:
    _, checkpoint, cache, known = inputs(protocol, seed)
    out = ROOT / "boundary_audit" / protocol
    out.mkdir(parents=True, exist_ok=True)
    ids = np.arange(known, dtype=np.int64)
    actual = _select_actual_banks(cache["train_features"], cache["train_labels"], ids, None, seed)
    jitter = float(checkpoint["config"].get("stats_cov_jitter", 1e-3))
    oracle_draws = _sample_empirical_gaussian_draws(actual, ids, draws, jitter, seed + 10101)
    stored_means = np.asarray(checkpoint["class_means"][:known], dtype=np.float64)
    stored_covs = np.stack([value.detach().cpu().numpy() for value in checkpoint["class_covs"][:known]])
    counts = {int(c): len(actual[int(c)]) for c in ids}
    stored_draws, repairs = _sample_stored_gaussian_draws(stored_means, stored_covs, counts, draws, device, seed + 10202)
    rows = []
    for class_id in ids.tolist():
        rows.append(_row(protocol, "BA_actual", -1, class_id, _margin(actual[class_id], class_id, checkpoint)))
        for draw in range(draws):
            rows.append(_row(protocol, "BB_oracle_gaussian", draw, class_id, _margin(oracle_draws[draw][class_id], class_id, checkpoint)))
            rows.append(_row(protocol, "BC_stored_gaussian", draw, class_id, _margin(stored_draws[draw][class_id], class_id, checkpoint)))
    write_csv(out / "per_class_margins.csv", rows)
    summary = {}
    for source in ("BB_oracle_gaussian", "BC_stored_gaussian"):
        by_class = []
        for class_id in ids.tolist():
            actual_row = next(row for row in rows if row["source"] == "BA_actual" and row["class_id"] == class_id)
            simulated = [row for row in rows if row["source"] == source and row["class_id"] == class_id]
            by_class.append({key: float(np.mean([row[key] for row in simulated]) - actual_row[key]) for key in ("q05", "q10", "negative_fraction")})
        summary[source] = {
            "class_macro_q05_gap": float(np.mean([row["q05"] for row in by_class])),
            "class_macro_q10_gap": float(np.mean([row["q10"] for row in by_class])),
            "class_macro_negative_fraction_gap": float(np.mean([row["negative_fraction"] for row in by_class])),
            "positive_q05_class_fraction": float(np.mean([row["q05"] > 0 for row in by_class])),
            "positive_q10_class_fraction": float(np.mean([row["q10"] > 0 for row in by_class])),
            "draws": draws,
        }
    # Conservative preregistered gate: both Gaussian sources must be easier on
    # both lower-tail quantiles and a majority of old classes.
    confirmed = all(
        summary[source]["class_macro_q05_gap"] > 0
        and summary[source]["class_macro_q10_gap"] > 0
        and summary[source]["positive_q05_class_fraction"] > 0.5
        and summary[source]["positive_q10_class_fraction"] > 0.5
        for source in summary
    )
    result = {"protocol": protocol, "known_classes": known, "draws": draws,
              "matched_class_counts": True, "boundary_gap_confirmed": confirmed,
              "gate": "both Gaussian sources: macro q05/q10 > actual and >50% classes positive",
              "sources": summary, "stored_covariance_repairs": repairs}
    write_json(out / "summary.json", result)
    print(f"[{protocol} boundary] confirmed={confirmed} {summary}", flush=True)
    return result
