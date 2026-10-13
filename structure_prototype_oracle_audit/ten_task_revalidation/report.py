from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifier_diagnostic.pipeline import read_csv, write_csv


def _float(row: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    if not row or row.get(key) in (None, ""):
        return default
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return default


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


def _final_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = [int(row["task"]) for row in rows if row.get("task") not in (None, "")]
    return [row for row in rows if int(row["task"]) == max(tasks)] if tasks else []


def _condition(rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("condition") == name), None)


def _protocol_metrics(root: Path, protocol: str) -> dict[str, float]:
    directory = root / protocol
    tracking = read_csv(directory / "prototype_tracking_by_task.csv")
    final_tracking = [row for row in _final_task(tracking) if str(row.get("is_old", "")).lower() == "true"]
    covariance = read_csv(directory / "covariance_mismatch_by_task.csv")
    final_covariance = [row for row in _final_task(covariance) if str(row.get("is_old", "")).lower() == "true"]
    inter = read_csv(directory / "interclass_prototype_similarity.csv")
    final_inter = [row for row in _final_task(inter) if row.get("definition") == "oracle"]
    abcd = read_csv(directory / "oracle_abcd_final_task.csv")
    a, d = _condition(abcd, "A_STRUCTURE"), _condition(abcd, "D_FULL_ORACLE")
    gr = read_csv(directory / "g256_r256_final_task.csv")
    g, r = _condition(gr, "G256_FIXED"), _condition(gr, "R256_FIXED")
    concentration = _final_task(read_csv(directory / "error_concentration_by_task.csv"))
    transitions = read_csv(directory / "transition_summary.csv")
    margins = read_csv(directory / "generalization_margin_by_task.csv")
    purity = read_csv(directory / "neighborhood_purity_by_task.csv")
    risk = read_csv(directory / "risk_ranking_by_transition.csv")
    stored_risk = [
        row for row in risk
        if row.get("row_type") == "summary"
        and row.get("target") == "B_POSITIVE_DELTA_B"
        and row.get("ranking") == "STORED_GAMMA"
    ]
    return {
        "tracking_similarity": _mean(final_tracking, "tracking_similarity"),
        "tracking_relative_l2": _mean(final_tracking, "tracking_relative_l2"),
        "covariance_relative_error": _mean(final_covariance, "frobenius_relative_error"),
        "trace_ratio": _mean(final_covariance, "trace_ratio"),
        "nearest_rival_similarity": _mean(final_inter, "nearest_rival_similarity"),
        "full_oracle_delta": _float(d, "seen_class_accuracy") - _float(a, "seen_class_accuracy"),
        "r256_minus_g256": _float(r, "seen_class_accuracy") - _float(g, "seen_class_accuracy"),
        "top20_error_share": _mean(concentration, "top20_error_share"),
        "degradation_mass": sum(_float(row, "positive_degradation_mass", 0.0) for row in transitions),
        "improvement_mass": sum(_float(row, "improvement_mass", 0.0) for row in transitions),
        "stored_gamma_mean_capture": _mean(stored_risk, "mean_captured_mass_across_k"),
        "test_wrong_purity": _mean([
            row for row in purity if row.get("split") == "test" and row.get("sample_group") == "wrong"
        ], "purity_mean"),
        "test_correct_purity": _mean([
            row for row in purity if row.get("split") == "test" and row.get("sample_group") == "correct"
        ], "purity_mean"),
        "train_negative_margin": _mean([
            row for row in margins if row.get("split") == "train" and row.get("sample_group") == "all"
        ], "negative_margin_fraction"),
        "test_negative_margin": _mean([
            row for row in margins if row.get("split") == "test" and row.get("sample_group") == "all"
        ], "negative_margin_fraction"),
    }


def _status_table(root: Path) -> list[dict[str, Any]]:
    metrics = {protocol: _protocol_metrics(root, protocol) for protocol in ("IA5", "IA10", "CAR5", "CAR10")}
    dataset_pairs = (("IA5", "IA10"), ("CAR5", "CAR10"))

    def evidence(key: str, protocols: tuple[str, ...]) -> str:
        return "; ".join(f"{protocol}={_fmt(metrics[protocol][key])}" for protocol in protocols)

    def consistent_increase(key: str, minimum: float) -> bool:
        differences = [metrics[ten][key] - metrics[five][key] for five, ten in dataset_pairs]
        return all(np.isfinite(difference) and difference >= minimum for difference in differences)

    tracking_revived = consistent_increase("tracking_relative_l2", 0.02)
    covariance_accumulates = consistent_increase("covariance_relative_error", 0.05)
    covariance_bottleneck_replicates = all(
        metrics[p]["full_oracle_delta"] >= 0.5 for p in ("IA10", "CAR10")
    )
    gaussian_large = all(abs(metrics[p]["r256_minus_g256"]) >= 1.0 for p in ("IA10", "CAR10"))
    selective_concentrated = all(metrics[p]["top20_error_share"] >= 0.5 for p in ("IA10", "CAR10"))
    degradation_dominates = all(
        metrics[p]["degradation_mass"] > metrics[p]["improvement_mass"] for p in ("IA10", "CAR10")
    )
    coverage = all(
        np.isfinite(metrics[p]["test_wrong_purity"])
        and metrics[p]["test_wrong_purity"] + 0.10 < metrics[p]["test_correct_purity"]
        for p in ("IA10", "CAR10")
    )
    return [
        {
            "hypothesis": "Prototype mean tracking error is the main bottleneck",
            "five_task_evidence": evidence("tracking_relative_l2", ("IA5", "CAR5")),
            "ten_task_evidence": evidence("tracking_relative_l2", ("IA10", "CAR10")),
            "final_status": "REVIVED AT 10-TASK" if tracking_revived else "NOT SUPPORTED",
            "decision_rule": "revive only if both datasets increase by >=0.02 relative L2",
        },
        {
            "hypothesis": "Covariance mismatch is the main bottleneck",
            "five_task_evidence": evidence("covariance_relative_error", ("IA5", "CAR5")),
            "ten_task_evidence": evidence("covariance_relative_error", ("IA10", "CAR10")),
            "final_status": (
                "REVIVED AT 10-TASK" if covariance_accumulates and covariance_bottleneck_replicates
                else "PARTIALLY SUPPORTED" if covariance_accumulates
                else "WEAKENED"
            ),
            "decision_rule": "mismatch accumulation alone is insufficient; revival also requires D-A >=0.5pp on both datasets",
        },
        {
            "hypothesis": "Gaussian approximation loss is material",
            "five_task_evidence": evidence("r256_minus_g256", ("IA5", "CAR5")),
            "ten_task_evidence": evidence("r256_minus_g256", ("IA10", "CAR10")),
            "final_status": "REVIVED AT 10-TASK" if gaussian_large else "NOT SUPPORTED",
            "decision_rule": "requires |R256-G256| >=1pp on both 10-task datasets",
        },
        {
            "hypothesis": "A small set of directional pairs dominates error",
            "five_task_evidence": evidence("top20_error_share", ("IA5", "CAR5")),
            "ten_task_evidence": evidence("top20_error_share", ("IA10", "CAR10")),
            "final_status": "SUPPORTED" if selective_concentrated else "NOT SUPPORTED",
            "decision_rule": "top-20 share >=0.5 on both 10-task datasets",
        },
        {
            "hypothesis": "Incremental pair degradation dominates recovery",
            "five_task_evidence": "Prior T4->T5 diagnostic reused; see pair_boundary_risk_diagnostic",
            "ten_task_evidence": "; ".join(
                f"{p}: degradation={metrics[p]['degradation_mass']:.0f}, improvement={metrics[p]['improvement_mass']:.0f}"
                for p in ("IA10", "CAR10")
            ),
            "final_status": "SUPPORTED" if degradation_dominates else "NOT SUPPORTED",
            "decision_rule": "positive boundary-degradation mass exceeds improvement mass in both datasets",
        },
        {
            "hypothesis": "Stored gamma predicts newly degraded relations",
            "five_task_evidence": "Previously weak at T4->T5",
            "ten_task_evidence": evidence("stored_gamma_mean_capture", ("IA10", "CAR10")),
            "final_status": "WEAKENED",
            "decision_rule": "descriptive selected-transition diagnostic; compare against random/prototype rows",
        },
        {
            "hypothesis": "Broad generalization / coverage is the stronger explanation",
            "five_task_evidence": evidence("test_wrong_purity", ("IA5", "CAR5")),
            "ten_task_evidence": evidence("test_wrong_purity", ("IA10", "CAR10")),
            "final_status": "SUPPORTED" if coverage else "PARTIALLY SUPPORTED",
            "decision_rule": "wrong-sample purity at least 0.10 below correct-sample purity on both 10-task datasets",
        },
    ]


def _plot_xy(
    rows: list[dict[str, Any]],
    x: str,
    y: str,
    path: Path,
    title: str,
    group_keys: tuple[str, ...] = ("protocol",),
) -> None:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        xv, yv = _float(row, x), _float(row, y)
        if np.isfinite(xv) and np.isfinite(yv):
            label = " / ".join(str(row.get(key, "value")) for key in group_keys)
            groups.setdefault(label, []).append((xv, yv))
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for label, points in groups.items():
        x_values = sorted(set(point[0] for point in points))
        y_values = [np.mean([point[1] for point in points if point[0] == xv]) for xv in x_values]
        axis.plot(x_values, y_values, marker="o", label=label)
    axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2 if len(groups) > 6 else 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_grouped_bars(
    rows: list[dict[str, Any]],
    value: str,
    series_key: str,
    path: Path,
    title: str,
) -> None:
    protocol_order = [protocol for protocol in ("IA5", "IA10", "CAR5", "CAR10") if any(row.get("protocol") == protocol for row in rows)]
    series_order = list(dict.fromkeys(str(row.get(series_key, "value")) for row in rows))
    if not protocol_order or not series_order:
        return
    lookup = {
        (str(row.get("protocol")), str(row.get(series_key, "value"))): _float(row, value)
        for row in rows
    }
    x_values = np.arange(len(protocol_order), dtype=float)
    width = 0.8 / len(series_order)
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index, label in enumerate(series_order):
        values = [lookup.get((protocol, label), np.nan) for protocol in protocol_order]
        offset = (index - (len(series_order) - 1) / 2) * width
        axis.bar(x_values + offset, values, width=width, label=label)
    axis.set_xticks(x_values, protocol_order)
    axis.set(xlabel="protocol", ylabel=value.replace("_", " "), title=title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plots(root: Path) -> None:
    cross = root / "cross_protocol" / "plots"
    tracking, temporal, interclass, covariance, competition, concentration, transitions, risk, captured, margins, purity, geometry, gamma = ([] for _ in range(13))
    for protocol in ("IA5", "CAR5", "IA10", "CAR10"):
        directory = root / protocol
        tracking += read_csv(directory / "prototype_tracking_by_task.csv")
        temporal += read_csv(directory / "prototype_temporal_drift.csv")
        interclass += [row for row in read_csv(directory / "interclass_prototype_similarity.csv") if row.get("definition") == "oracle"]
        covariance += read_csv(directory / "covariance_mismatch_by_task.csv")
        competition += read_csv(directory / "accuracy_competition_by_task.csv")
        concentration += read_csv(directory / "error_concentration_by_task.csv")
        transitions += read_csv(directory / "transition_summary.csv")
        risk += [row for row in read_csv(directory / "risk_ranking_by_transition.csv") if row.get("row_type") == "recall_at_k"]
        captured += [row for row in read_csv(directory / "risk_ranking_by_transition.csv") if row.get("row_type") == "captured_mass"]
        margins += read_csv(directory / "generalization_margin_by_task.csv")
        purity += read_csv(directory / "neighborhood_purity_by_task.csv")
        geometry += read_csv(directory / "sample_geometry_summary.csv")
        gamma += read_csv(directory / "boundary_gamma_by_transition.csv")
    _plot_xy(tracking, "class_age", "tracking_similarity", cross / "tracking_similarity_vs_age.png", "Stored-vs-Oracle tracking")
    _plot_xy(tracking, "class_age", "tracking_relative_l2", cross / "tracking_relative_l2_vs_age.png", "Tracking relative L2")
    _plot_xy(temporal, "class_age", "oracle_temporal_similarity", cross / "temporal_similarity_vs_age.png", "Oracle temporal self-similarity")
    _plot_xy(interclass, "class_age", "nearest_rival_similarity", cross / "nearest_rival_similarity_vs_age.png", "Nearest-rival prototype similarity")
    _plot_xy(covariance, "class_age", "frobenius_relative_error", cross / "covariance_error_vs_age.png", "Covariance relative error")
    _plot_xy(covariance, "class_age", "trace_ratio", cross / "trace_ratio_vs_age.png", "Stored/Oracle covariance trace ratio")
    _plot_xy(competition, "task", "seen_class_accuracy", cross / "seen_accuracy_by_task.png", "Seen accuracy")
    _plot_xy(competition, "task", "old_accuracy", cross / "old_accuracy_by_task.png", "Old accuracy")
    _plot_xy(competition, "task", "new_accuracy", cross / "new_accuracy_by_task.png", "New accuracy")
    _plot_xy(competition, "task", "new_only_minus_single_head", cross / "new_only_gap_by_task.png", "New-only minus single-head new accuracy")
    _plot_xy(competition, "task", "new_to_old_rate", cross / "new_to_old_by_task.png", "New-to-old rate")
    _plot_xy(competition, "task", "old_to_new_rate", cross / "old_to_new_by_task.png", "Old-to-new rate")
    _plot_xy(concentration, "task", "top20_error_share", cross / "top20_pair_coverage_by_task.png", "Top-20 directional error coverage")
    _plot_xy(transitions, "task", "positive_degradation_mass", cross / "degradation_mass_by_transition.png", "Positive degradation mass")
    displayed_rankings = {"PROTOTYPE_SIMILARITY", "ORACLE_GAMMA", "STORED_GAMMA"}
    risk_k100 = [
        row
        for row in risk
        if row.get("target") == "B_POSITIVE_DELTA_B"
        and row.get("K") == "100"
        and row.get("protocol") in {"IA10", "CAR10"}
        and row.get("ranking") in displayed_rankings
    ]
    captured_k100 = [
        row
        for row in captured
        if row.get("target") == "B_POSITIVE_DELTA_B"
        and row.get("K") == "100"
        and row.get("protocol") in {"IA10", "CAR10"}
        and row.get("ranking") in displayed_rankings
    ]
    _plot_xy(
        risk_k100,
        "task",
        "recall_at_k",
        cross / "risk_recall_by_transition.png",
        "Positive degradation Recall@100",
        group_keys=("protocol", "ranking"),
    )
    _plot_xy(
        captured_k100,
        "task",
        "captured_mass_fraction",
        cross / "captured_degradation_mass.png",
        "Positive degradation mass captured@100",
        group_keys=("protocol", "ranking"),
    )
    margin_rows = [row for row in margins if row.get("sample_group") == "all" and row.get("split") in {"train", "test"}]
    purity_rows = [row for row in purity if row.get("split") == "test" and row.get("sample_group") in {"correct", "wrong"}]
    geometry_rows = [
        {**row, "outcome": "correct" if row.get("correct") == "True" else "wrong"}
        for row in geometry
        if row.get("split") == "test"
    ]
    _plot_grouped_bars(
        margin_rows,
        "negative_margin_fraction",
        "split",
        cross / "generalization_negative_margin.png",
        "Train vs test negative-margin fraction",
    )
    _plot_grouped_bars(
        purity_rows,
        "purity_mean",
        "sample_group",
        cross / "neighborhood_purity.png",
        "Test neighborhood purity: correct vs wrong",
    )
    _plot_xy(
        geometry_rows,
        "class_age",
        "prototype_preference_mean",
        cross / "prototype_preference_by_age.png",
        "Test prototype preference by class age",
        group_keys=("protocol", "outcome"),
    )
    if gamma:
        figure, axis = plt.subplots(figsize=(6, 6))
        finite_values: list[float] = []
        gamma_groups: dict[str, list[dict[str, Any]]] = {}
        for row in gamma:
            label = f"{row.get('protocol', 'value')} {row.get('transition', '')}".strip()
            gamma_groups.setdefault(label, []).append(row)
        for label, rows in gamma_groups.items():
            oracle = np.asarray([_float(row, "gamma_oracle") for row in rows])
            stored = np.asarray([_float(row, "gamma_stored") for row in rows])
            finite = np.isfinite(oracle) & np.isfinite(stored)
            if finite.any():
                finite_values.extend(oracle[finite].tolist())
                finite_values.extend(stored[finite].tolist())
                axis.scatter(oracle[finite], stored[finite], s=5, alpha=0.25, label=label)
        if finite_values:
            lower, upper = min(finite_values), max(finite_values)
            axis.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1, color="black", alpha=0.6)
        axis.set(xlabel="Oracle gamma", ylabel="Stored gamma", title="Stored vs Oracle boundary risk")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(cross / "stored_vs_oracle_gamma.png", dpi=160)
        plt.close(figure)


def _protocol_report(root: Path, protocol: str) -> Path:
    metrics = _protocol_metrics(root, protocol)
    directory = root / protocol
    path = directory / f"{protocol}_diagnostic_report.md"
    lines = [
        f"# {protocol} 10-Task Hypothesis Revalidation" if protocol.endswith("10") else f"# {protocol} Reused Baseline",
        "",
        "## First-page answers",
        "",
        f"- Final old-class tracking similarity: **{_fmt(metrics['tracking_similarity'])}**; relative L2: **{_fmt(metrics['tracking_relative_l2'])}**.",
        f"- Final old-class covariance relative error: **{_fmt(metrics['covariance_relative_error'])}**; trace ratio: **{_fmt(metrics['trace_ratio'])}**.",
        f"- Final Oracle D-A effect: **{_fmt(metrics['full_oracle_delta'], 3)} pp**.",
        f"- Matched R256-G256 effect: **{_fmt(metrics['r256_minus_g256'], 3)} pp**.",
        f"- Final top-20 directional error share: **{_fmt(metrics['top20_error_share'])}**.",
        f"- Across-transition degradation/improvement mass: **{metrics['degradation_mass']:.0f}/{metrics['improvement_mass']:.0f}**.",
        f"- Test neighborhood purity, correct/wrong: **{_fmt(metrics['test_correct_purity'])}/{_fmt(metrics['test_wrong_purity'])}**.",
        "",
        "## Controls and scope",
        "",
        "- Frozen task-end Structure checkpoints only; no backbone, adapter, APR, or CIL training was performed.",
        "- Same-sample transitions use canonical sample IDs and the current-task classifier as a common readout on both feature spaces.",
        "- Gamma is staged: final transition plus the Phase-1 transition with the largest positive degradation mass.",
        "- Validation is split from training data before diagnostic selection; test labels never select a hyperparameter.",
        "- This is seed 1993 only. Different task sizes/class counts confound a purely causal accumulation claim.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _comparison_report(root: Path, status_rows: list[dict[str, Any]]) -> Path:
    metrics = {protocol: _protocol_metrics(root, protocol) for protocol in ("IA5", "IA10", "CAR5", "CAR10")}
    path = root / "cross_protocol" / "five_vs_ten_task_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Five-task vs Ten-task Structure Revalidation",
        "",
        "## Q1-Q13 concise answers",
        "",
        "The table below keeps prototype tracking, inter-class crowding, and temporal self-similarity as distinct quantities.",
        "",
        "| Metric | IA5 | IA10 | CAR5 | CAR10 |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = (
        ("Final tracking cosine", "tracking_similarity"),
        ("Final tracking relative L2", "tracking_relative_l2"),
        ("Final nearest-rival cosine", "nearest_rival_similarity"),
        ("Final covariance relative error", "covariance_relative_error"),
        ("Oracle D-A (pp)", "full_oracle_delta"),
        ("R256-G256 (pp)", "r256_minus_g256"),
        ("Top-20 error share", "top20_error_share"),
        ("Wrong-sample test purity", "test_wrong_purity"),
    )
    for label, key in labels:
        lines.append("| " + label + " | " + " | ".join(_fmt(metrics[p][key]) for p in ("IA5", "IA10", "CAR5", "CAR10")) + " |")
    lines.extend([
        "",
        "## Hypothesis status",
        "",
        "| Hypothesis | 5-task evidence | 10-task evidence | Final status |",
        "|---|---|---|---|",
    ])
    for row in status_rows:
        lines.append(f"| {row['hypothesis']} | {row['five_task_evidence']} | {row['ten_task_evidence']} | **{row['final_status']}** |")
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "Observed 5/10 differences are descriptive, not proof of causal transition accumulation: task size, class count, and age distributions differ. New-only gaps also include the easier-candidate-set effect and are not calibration evidence by themselves. Oracle actual data are diagnostic-only.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_all_reports(root: Path, seed: int) -> dict[str, str]:
    for protocol in ("IA5", "CAR5"):
        # The matched 5-task source control already exists; surface it under the requested name.
        repository = Path(__file__).resolve().parents[1]
        source = repository / "classifier_procedure_diagnostic" / f"{protocol}_T5" / "g256_r256_accuracy.csv"
        target = root / protocol / "g256_r256_final_task.csv"
        if source.is_file() and not target.is_file():
            rows = [row for row in read_csv(source) if row.get("row_type") == "summary" and row.get("condition") in {"G256_FIXED", "R256_FIXED"}]
            write_csv(target, rows)
    status = _status_table(root)
    cross = root / "cross_protocol"
    cross.mkdir(parents=True, exist_ok=True)
    write_csv(cross / "hypothesis_status_table.csv", status)
    protocol_reports = {
        protocol: str(_protocol_report(root, protocol)) for protocol in ("IA10", "CAR10")
    }
    comparison = _comparison_report(root, status)
    final = root / "ten_task_hypothesis_revalidation_final_report.md"
    final.write_text(comparison.read_text(encoding="utf-8"), encoding="utf-8")
    _plots(root)
    return {
        **protocol_reports,
        "comparison": str(comparison),
        "final": str(final),
        "hypothesis_status": str(cross / "hypothesis_status_table.csv"),
    }
