from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifier_diagnostic.pipeline import read_csv, write_csv

from .metrics import class_macro


SOURCE_ORDER = ("A_ACTUAL_CURRENT", "B_ORACLE_GAUSSIAN", "C_STRUCTURE_GAUSSIAN")
SOURCE_LABEL = {
    "A_ACTUAL_CURRENT": "A Actual-current",
    "B_ORACLE_GAUSSIAN": "B Oracle Gaussian",
    "C_STRUCTURE_GAUSSIAN": "C Structure Gaussian",
}


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _bool(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([_float(row, key) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if values.size else float("nan")


def _bootstrap_ci(values: np.ndarray, seed: int = 1993, trials: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    means = np.empty(trials, dtype=np.float64)
    for start in range(0, trials, 100):
        stop = min(start + 100, trials)
        indices = rng.randint(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _selected_probe(rows: list[dict[str, Any]], representation: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("row_type") == "selected" and row.get("representation") == representation]
    if len(matches) != 1:
        raise RuntimeError(f"expected one selected {representation} probe row, got {len(matches)}")
    return matches[0]


def _protocol_metrics(root: Path, protocol: str) -> dict[str, Any]:
    directory = root / protocol
    a = read_csv(directory / "query_support_A.csv")
    b = read_csv(directory / "query_support_B.csv")
    c = read_csv(directory / "query_support_C.csv")
    paired = read_csv(directory / "paired_support_delta.csv")
    representation = read_csv(directory / "pretrained_vs_final_paired.csv")
    probes = read_csv(directory / "probe_results.csv")
    pre_probe = _selected_probe(probes, "PRETRAINED")
    final_probe = _selected_probe(probes, "STRUCTURE_FINAL")
    wrong_a = [row for row in a if not _bool(row, "classifier_correct")]
    correct_a = [row for row in a if _bool(row, "classifier_correct")]
    old_a = [row for row in a if row.get("query_type") == "old"]
    new_a = [row for row in a if row.get("query_type") == "new"]

    draw_ids = sorted({int(row["draw"]) for row in paired})
    ba_by_draw, cb_by_draw, ba_macro_by_draw, cb_macro_by_draw = [], [], [], []
    for draw in draw_ids:
        rows = [row for row in paired if int(row["draw"]) == draw]
        labels = np.asarray([int(row["class_id"]) for row in rows])
        ba = np.asarray([_float(row, "delta_B_minus_A") for row in rows])
        cb = np.asarray([_float(row, "delta_C_minus_B") for row in rows])
        ba_by_draw.append(float(ba.mean()))
        cb_by_draw.append(float(cb.mean()))
        ba_macro_by_draw.append(class_macro(ba, labels))
        cb_macro_by_draw.append(class_macro(cb, labels))
    query_mean_ba = []
    query_mean_cb = []
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        by_sample.setdefault(row["sample_id"], []).append(row)
    for rows in by_sample.values():
        query_mean_ba.append(_mean(rows, "delta_B_minus_A"))
        query_mean_cb.append(_mean(rows, "delta_C_minus_B"))
    ba_ci = _bootstrap_ci(np.asarray(query_mean_ba), seed=1993 + len(draw_ids))
    cb_ci = _bootstrap_ci(np.asarray(query_mean_cb), seed=2993 + len(draw_ids))

    rep_delta = np.asarray([_float(row, "delta_final_minus_pretrained") for row in representation])
    rep_labels = np.asarray([int(row["class_id"]) for row in representation])
    rep_ci = _bootstrap_ci(rep_delta, seed=3993)
    metrics = {
        "protocol": protocol,
        "query_count": len(a),
        "actual_margin": _mean(a, "support_margin"),
        "actual_margin_macro": class_macro(
            np.asarray([_float(row, "support_margin") for row in a]),
            np.asarray([int(row["class_id"]) for row in a]),
        ),
        "actual_correct_margin": _mean(correct_a, "support_margin"),
        "actual_wrong_margin": _mean(wrong_a, "support_margin"),
        "actual_old_margin": _mean(old_a, "support_margin"),
        "actual_new_margin": _mean(new_a, "support_margin"),
        "delta_BA_micro": float(np.mean(ba_by_draw)),
        "delta_BA_draw_std": float(np.std(ba_by_draw)),
        "delta_BA_macro": float(np.mean(ba_macro_by_draw)),
        "delta_BA_ci_low": ba_ci[0],
        "delta_BA_ci_high": ba_ci[1],
        "delta_CB_micro": float(np.mean(cb_by_draw)),
        "delta_CB_draw_std": float(np.std(cb_by_draw)),
        "delta_CB_macro": float(np.mean(cb_macro_by_draw)),
        "delta_CB_ci_low": cb_ci[0],
        "delta_CB_ci_high": cb_ci[1],
        "pretrained_support_margin": _float(pre_probe, "support_margin_micro"),
        "final_support_margin": _float(final_probe, "support_margin_micro"),
        "delta_final_pretrained_support": float(rep_delta.mean()),
        "delta_final_pretrained_support_macro": class_macro(rep_delta, rep_labels),
        "delta_final_pretrained_ci_low": rep_ci[0],
        "delta_final_pretrained_ci_high": rep_ci[1],
        "pretrained_train_accuracy": _float(pre_probe, "train_accuracy"),
        "final_train_accuracy": _float(final_probe, "train_accuracy"),
        "pretrained_validation_accuracy": _float(pre_probe, "validation_accuracy"),
        "final_validation_accuracy": _float(final_probe, "validation_accuracy"),
        "pretrained_test_accuracy": _float(pre_probe, "test_accuracy"),
        "final_test_accuracy": _float(final_probe, "test_accuracy"),
        "pretrained_macro_accuracy": _float(pre_probe, "class_macro_accuracy"),
        "final_macro_accuracy": _float(final_probe, "class_macro_accuracy"),
        "pretrained_old_accuracy": _float(pre_probe, "old_accuracy"),
        "final_old_accuracy": _float(final_probe, "old_accuracy"),
        "pretrained_new_accuracy": _float(pre_probe, "new_accuracy"),
        "final_new_accuracy": _float(final_probe, "new_accuracy"),
        "pretrained_train_test_gap": _float(pre_probe, "train_test_accuracy_gap"),
        "final_train_test_gap": _float(final_probe, "train_test_accuracy_gap"),
    }
    metrics["delta_probe_test"] = metrics["final_test_accuracy"] - metrics["pretrained_test_accuracy"]
    metrics["delta_probe_macro"] = metrics["final_macro_accuracy"] - metrics["pretrained_macro_accuracy"]
    metrics["delta_probe_train"] = metrics["final_train_accuracy"] - metrics["pretrained_train_accuracy"]
    return metrics


def _boxplot(rows_by_source: dict[str, list[dict[str, Any]]], metric: str, path: Path, title: str) -> None:
    values, labels = [], []
    for source in SOURCE_ORDER:
        array = np.asarray([_float(row, metric) for row in rows_by_source[source]])
        if source != "A_ACTUAL_CURRENT":
            # Multiple Gaussian draws repeat each query.  The distribution is
            # intentionally pooled while draw variability is reported in CSV.
            pass
        values.append(array[np.isfinite(array)])
        labels.append(SOURCE_LABEL[source].split()[0])
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(values, labels=labels, showfliers=False)
    axis.set(title=title, ylabel=metric.replace("_", " "))
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _hist(values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(values[np.isfinite(values)], bins=50, alpha=0.8)
    axis.axvline(0.0, linestyle="--", color="black", linewidth=1)
    axis.set(title=title, xlabel=xlabel, ylabel="queries")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _group_bar(
    rows: list[dict[str, Any]],
    category_key: str,
    series_key: str,
    value_key: str,
    path: Path,
    title: str,
) -> None:
    categories = list(dict.fromkeys(str(row[category_key]) for row in rows))
    series = list(dict.fromkeys(str(row[series_key]) for row in rows))
    lookup: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        lookup.setdefault((str(row[category_key]), str(row[series_key])), []).append(_float(row, value_key))
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(series), 1)
    figure, axis = plt.subplots(figsize=(max(7, len(categories) * 0.7), 4.5))
    for index, label in enumerate(series):
        values = [float(np.nanmean(lookup.get((category, label), [np.nan]))) for category in categories]
        offset = (index - (len(series) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=label)
    axis.set_xticks(x, categories, rotation=30 if len(categories) > 8 else 0)
    axis.set(title=title, ylabel=value_key.replace("_", " "))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _protocol_plots(root: Path, protocol: str) -> None:
    directory = root / protocol
    plot_dir = directory / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows_by_source = {
        "A_ACTUAL_CURRENT": read_csv(directory / "query_support_A.csv"),
        "B_ORACLE_GAUSSIAN": read_csv(directory / "query_support_B.csv"),
        "C_STRUCTURE_GAUSSIAN": read_csv(directory / "query_support_C.csv"),
    }
    for metric, filename, title in (
        ("support_margin", "abc_support_margin_distribution.png", "A/B/C support-margin distribution"),
        ("own_support", "abc_own_support_distribution.png", "A/B/C own-support distribution"),
        ("strongest_rival_support", "abc_rival_support_distribution.png", "A/B/C strongest-rival distribution"),
    ):
        _boxplot(rows_by_source, metric, plot_dir / filename, f"{protocol}: {title}")
    paired = read_csv(directory / "paired_support_delta.csv")
    # Average repeated Gaussian draws per query before drawing paired deltas.
    sample_groups: dict[str, list[dict[str, Any]]] = {}
    for row in paired:
        sample_groups.setdefault(row["sample_id"], []).append(row)
    ba = np.asarray([_mean(rows, "delta_B_minus_A") for rows in sample_groups.values()])
    cb = np.asarray([_mean(rows, "delta_C_minus_B") for rows in sample_groups.values()])
    _hist(ba, plot_dir / "paired_B_minus_A.png", f"{protocol}: paired B - A", "support-margin delta")
    _hist(cb, plot_dir / "paired_C_minus_B.png", f"{protocol}: paired C - B", "support-margin delta")

    combined = sum(rows_by_source.values(), [])
    for row in combined:
        row["source_short"] = row["source"].split("_")[0]
        row["outcome"] = "correct" if _bool(row, "classifier_correct") else "wrong"
    _group_bar(
        combined, "source_short", "outcome", "support_margin",
        plot_dir / "correct_wrong_support.png", f"{protocol}: classifier correct vs wrong support",
    )
    _group_bar(
        combined, "reference_count_bin", "source_short", "support_margin",
        plot_dir / "reference_count_support.png", f"{protocol}: support by reference-count bin",
    )
    old_rows = [row for row in combined if row.get("query_type") == "old"]
    _group_bar(
        old_rows, "class_age", "source_short", "support_margin",
        plot_dir / "class_age_support.png", f"{protocol}: old-query support by class age",
    )

    rep = read_csv(directory / "pretrained_vs_final_paired.csv")
    pre = np.asarray([_float(row, "pretrained_support_margin") for row in rep])
    final = np.asarray([_float(row, "final_support_margin") for row in rep])
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(pre, final, s=6, alpha=0.25)
    lower, upper = float(min(pre.min(), final.min())), float(max(pre.max(), final.max()))
    axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
    axis.set(title=f"{protocol}: pretrained vs final support", xlabel="pretrained margin", ylabel="final margin")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(plot_dir / "pretrained_vs_final_support_scatter.png", dpi=160)
    plt.close(figure)
    _hist(
        final - pre,
        plot_dir / "final_minus_pretrained_support.png",
        f"{protocol}: final - pretrained support",
        "support-margin delta",
    )
    rep_rows = read_csv(directory / "pretrained_support.csv") + read_csv(directory / "final_support.csv")
    for row in rep_rows:
        row["representation_short"] = "P0" if row.get("representation") == "PRETRAINED" else "P1"
        row["outcome"] = "correct" if _bool(row, "probe_correct") else "wrong"
    _group_bar(
        rep_rows, "representation_short", "outcome", "support_margin",
        plot_dir / "representation_correct_wrong_support.png",
        f"{protocol}: probe correct vs wrong support",
    )


def _cross_plot(root: Path, protocols: tuple[str, ...], metrics: dict[str, dict[str, Any]]) -> None:
    plot_dir = root / "cross_protocol" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(protocols), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(x - width / 2, [metrics[p]["pretrained_test_accuracy"] for p in protocols], width, label="Pretrained")
    axis.bar(x + width / 2, [metrics[p]["final_test_accuracy"] for p in protocols], width, label="Structure final")
    axis.set_xticks(x, protocols)
    axis.set(title="Cosine-probe test accuracy", ylabel="accuracy (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "pretrained_vs_final_probe_accuracy.png", dpi=160)
    plt.close(figure)


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def _status_rows(metrics: dict[str, dict[str, Any]], protocols: tuple[str, ...]) -> tuple[list[dict[str, str]], bool, str]:
    all_actual_low = all(
        metrics[p]["actual_wrong_margin"] < metrics[p]["actual_correct_margin"] - 0.05
        for p in protocols
    )
    gaussian_loss = all(
        metrics[p]["delta_BA_micro"] < -0.01
        and metrics[p]["delta_BA_macro"] < -0.01
        and metrics[p]["delta_BA_ci_high"] < 0.0
        for p in protocols
    )
    stored_loss = all(
        metrics[p]["delta_CB_micro"] < -0.01
        and metrics[p]["delta_CB_macro"] < -0.01
        and metrics[p]["delta_CB_ci_high"] < 0.0
        for p in protocols
    )
    representation_degradation = all(
        metrics[p]["delta_probe_train"] > 0.0
        and metrics[p]["delta_probe_test"] < -0.5
        and metrics[p]["delta_final_pretrained_support"] < -0.01
        and metrics[p]["delta_final_pretrained_ci_high"] < 0.0
        for p in protocols
    )
    direction_mismatch = any(
        np.sign(metrics[p]["delta_probe_test"]) != np.sign(metrics[p]["delta_final_pretrained_support"])
        and abs(metrics[p]["delta_probe_test"]) >= 0.5
        and abs(metrics[p]["delta_final_pretrained_support"]) >= 0.01
        for p in protocols
    )
    rows = [
        {
            "candidate_explanation": "Actual training coverage limitation",
            "evidence": "; ".join(f"{p}: correct/wrong A={_fmt(metrics[p]['actual_correct_margin'])}/{_fmt(metrics[p]['actual_wrong_margin'])}" for p in protocols),
            "counter_evidence": "Low actual-bank support does not identify whether the representation or finite samples caused it.",
            "status": "SUPPORTED" if all_actual_low else "INCONCLUSIVE",
        },
        {
            "candidate_explanation": "Representation generalization degradation",
            "evidence": "; ".join(f"{p}: train/test/support delta={metrics[p]['delta_probe_train']:+.2f}pp/{metrics[p]['delta_probe_test']:+.2f}pp/{metrics[p]['delta_final_pretrained_support']:+.4f}" for p in protocols),
            "counter_evidence": "Single seed; pretrained and final features may improve different aspects of geometry.",
            "status": "SUPPORTED" if representation_degradation else "NOT SUPPORTED",
        },
        {
            "candidate_explanation": "Gaussianization information loss",
            "evidence": "; ".join(f"{p}: B-A micro/macro={metrics[p]['delta_BA_micro']:+.4f}/{metrics[p]['delta_BA_macro']:+.4f}" for p in protocols),
            "counter_evidence": "A geometry effect alone is not downstream error recovery.",
            "status": "SUPPORTED" if gaussian_loss else "NOT SUPPORTED",
        },
        {
            "candidate_explanation": "Stored moment approximation loss",
            "evidence": "; ".join(f"{p}: C-B micro/macro={metrics[p]['delta_CB_micro']:+.4f}/{metrics[p]['delta_CB_macro']:+.4f}" for p in protocols),
            "counter_evidence": "Must be reconciled with the small/inconsistent Oracle classifier-adaptation gains.",
            "status": "SUPPORTED" if stored_loss else "NOT SUPPORTED",
        },
        {
            "candidate_explanation": "Single-head competition",
            "evidence": "A/B/C support and representation-specific cosine probes expose geometry/decision disagreement.",
            "counter_evidence": "This diagnostic does not isolate calibration or candidate-count causally.",
            "status": "INCONCLUSIVE",
        },
        {
            "candidate_explanation": "Local cosine metric limitation",
            "evidence": "Support-margin and probe-accuracy directions disagree in at least one protocol." if direction_mismatch else "Support-margin and probe-accuracy directions do not show a material sign conflict.",
            "counter_evidence": "Agreement in direction still does not make support a sufficient quality metric.",
            "status": "PARTIALLY SUPPORTED" if direction_mismatch else "INCONCLUSIVE",
        },
    ]
    abc_equal = all(abs(metrics[p]["delta_BA_micro"]) < 0.01 and abs(metrics[p]["delta_CB_micro"]) < 0.01 for p in protocols)
    final_not_worse = all(metrics[p]["delta_probe_test"] >= -0.5 and metrics[p]["delta_final_pretrained_support"] >= -0.01 for p in protocols)
    if abc_equal and representation_degradation:
        return rows, True, "Rule A: A≈B≈C이고 final representation의 train fit은 높지만 test/support는 일관되게 낮다."
    if abc_equal and final_not_worse:
        return rows, False, "Rule B: A≈B≈C이고 Structure final이 pretrained보다 명확히 나쁘지 않다."
    if gaussian_loss:
        return rows, False, "Rule C의 geometry 조건은 충족하지만 downstream error recovery 연결은 아직 검증되지 않았다."
    if stored_loss:
        return rows, False, "Rule D의 geometry 조건은 충족하지만 matched downstream 회복 근거가 아직 없다."
    return rows, False, "Rule E: protocol 공통의 뚜렷한 source/representation 차이가 확립되지 않았다."


def generate_report(root: Path, protocols: tuple[str, ...], gaussian_draws: int) -> dict[str, str]:
    root = Path(root)
    metrics = {protocol: _protocol_metrics(root, protocol) for protocol in protocols}
    for protocol in protocols:
        _protocol_plots(root, protocol)
    _cross_plot(root, protocols, metrics)
    cross = root / "cross_protocol"
    cross.mkdir(parents=True, exist_ok=True)
    summary_rows = [metrics[protocol] for protocol in protocols]
    write_csv(cross / "summary_table.csv", summary_rows)
    decision_rows, method_ready, decision_reason = _status_rows(metrics, protocols)
    write_csv(cross / "decision_table.csv", decision_rows)

    report = root / "coverage_generalization_final_report.md"
    lines = [
        "# Coverage Source + Pretrained-vs-Final Generalization Diagnostic",
        "",
        "## 1. 실행 범위와 통제",
        "",
        f"- Protocol: `{list(protocols)}`, seed `1993`, final task only.",
        "- 기존 Structure checkpoint와 final-feature cache를 읽기 전용으로 사용했다. Backbone/CIL training은 재실행하지 않았다.",
        f"- B/C Gaussian은 class별 A의 unique reference 수와 동일하게 맞추고 `{gaussian_draws}`개 독립 draw를 사용했다.",
        "- Gaussian은 raw feature space에서 생성한 뒤 support 계산 직전에 L2 normalize했다.",
        "- New-class bank는 actual-current로 고정했으며 old/new query 결과를 구분했다.",
        "- Pretrained/final cosine probe는 같은 train/validation index, optimizer, LR grid, epoch permutation과 budget을 사용했다. Test label은 선택에 사용하지 않았다.",
        "- Negative classifier-margin fraction은 별도 representation evidence로 재사용하지 않았다.",
        "",
        "## 2. 핵심 결과",
        "",
        "### 2.1 Coverage source",
        "",
        "| Protocol | A margin | A correct | A wrong | B-A micro | B-A macro | C-B micro | C-B macro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in protocols:
        m = metrics[protocol]
        lines.append(
            f"| {protocol} | {_fmt(m['actual_margin'])} | {_fmt(m['actual_correct_margin'])} | {_fmt(m['actual_wrong_margin'])} | "
            f"{m['delta_BA_micro']:+.4f} ± {m['delta_BA_draw_std']:.4f} | {m['delta_BA_macro']:+.4f} | "
            f"{m['delta_CB_micro']:+.4f} ± {m['delta_CB_draw_std']:.4f} | {m['delta_CB_macro']:+.4f} |"
        )
    lines.extend([
        "",
        "`B-A`와 `C-B`의 ± 값은 Gaussian draw 간 표준편차다. Micro와 class-macro를 함께 제시하며, query-level paired delta는 각 protocol CSV에 보존했다.",
        "",
        "### 2.2 Pretrained vs Structure final",
        "",
        "| Protocol | P0 train | P1 train | P0 test | P1 test | Δ test | P0 macro | P1 macro | Δ support |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for protocol in protocols:
        m = metrics[protocol]
        lines.append(
            f"| {protocol} | {m['pretrained_train_accuracy']:.2f} | {m['final_train_accuracy']:.2f} | "
            f"{m['pretrained_test_accuracy']:.2f} | {m['final_test_accuracy']:.2f} | {m['delta_probe_test']:+.2f}pp | "
            f"{m['pretrained_macro_accuracy']:.2f} | {m['final_macro_accuracy']:.2f} | {m['delta_final_pretrained_support']:+.4f} |"
        )
    lines.extend([
        "",
        "![Pretrained와 Structure final cosine-probe test accuracy](cross_protocol/plots/pretrained_vs_final_probe_accuracy.png)",
        "",
        "## 3. 필수 질문 Q1–Q9",
        "",
        "### Q1. Actual-current bank에서도 wrong held-out query support가 낮은가?",
        "",
        "; ".join(
            f"{p} correct/wrong margin은 `{metrics[p]['actual_correct_margin']:.4f}/{metrics[p]['actual_wrong_margin']:.4f}`"
            for p in protocols
        ) + "이다. 이 결과는 실제 current training reference와 cosine metric 기준의 coverage를 말하며 representation overfitting 자체를 확정하지 않는다.",
        "",
        "### Q2. Oracle Gaussian은 actual 대비 support margin을 감소시키는가?",
        "",
        "; ".join(
            f"{p} B-A `{metrics[p]['delta_BA_micro']:+.4f}` (class-macro `{metrics[p]['delta_BA_macro']:+.4f}`, paired 95% CI `[{metrics[p]['delta_BA_ci_low']:+.4f}, {metrics[p]['delta_BA_ci_high']:+.4f}]`)"
            for p in protocols
        ) + ". 감소의 크기·방향이 두 protocol과 class-macro에서 일치하는지를 기준으로 판단해야 한다.",
        "",
        "### Q3. Structure Gaussian은 Oracle Gaussian보다 추가 손실을 만드는가?",
        "",
        "; ".join(
            f"{p} C-B `{metrics[p]['delta_CB_micro']:+.4f}` (class-macro `{metrics[p]['delta_CB_macro']:+.4f}`, paired 95% CI `[{metrics[p]['delta_CB_ci_low']:+.4f}, {metrics[p]['delta_CB_ci_high']:+.4f}]`)"
            for p in protocols
        ) + ". 이 차이는 local support effect이며 기존 Oracle classifier-adaptation 결과와 별도로 downstream 병목을 뜻하지 않는다.",
        "",
        "### Q4. Micro뿐 아니라 class-macro에서도 유지되는가?",
        "",
        "위 표의 micro/macro 부호와 크기를 함께 기준으로 삼았다. 둘 중 하나에서만 나타난 차이는 일부 class 또는 sample 수 불균형의 영향으로 간주한다.",
        "",
        "### Q5. Low-reference-count class 때문만인가?",
        "",
        "각 protocol의 `reference_count_support.png`와 `paired_support_delta.csv`에 `01-04`, `05-09`, `10-19`, `20-49`, `50+` bin을 기록했다. A/B/C bank size는 모든 class에서 동일하고 A에는 replacement가 없다.",
        "",
        "### Q6. Structure final의 train fit, held-out 성능, support는 좋아졌는가?",
        "",
        "; ".join(
            f"{p}: final-pretrained train `{metrics[p]['delta_probe_train']:+.2f}pp`, test `{metrics[p]['delta_probe_test']:+.2f}pp`, support `{metrics[p]['delta_final_pretrained_support']:+.4f}`"
            for p in protocols
        ) + ". 절대 train/test accuracy도 2.2 표에 함께 제시했다.",
        "",
        "### Q7. Continual adaptation이 held-out generalization을 훼손했다는 evidence가 있는가?",
        "",
        "강한 evidence는 두 protocol에서 `final train↑`, `final test↓`, `final support↓`가 동시에 나타나고 paired support CI가 0 아래일 때만 인정한다. 최종 상태는 decision table에 반영했다.",
        "",
        "### Q8. Support metric과 cosine-probe accuracy 변화가 같은 방향인가?",
        "",
        "; ".join(
            f"{p}: probe Δ `{metrics[p]['delta_probe_test']:+.2f}pp`, support Δ `{metrics[p]['delta_final_pretrained_support']:+.4f}`"
            for p in protocols
        ) + ". 부호가 충돌하면 local cosine support를 representation quality의 충분한 proxy로 사용하지 않는다.",
        "",
        "### Q9. 현재 evidence로 method를 만들 준비가 되었는가?",
        "",
        f"**METHOD READY = {'YES' if method_ready else 'NO'}**",
        "",
        decision_reason,
        "",
        "## 4. Hypothesis decision table",
        "",
        "| Candidate explanation | Evidence | Counter-evidence | Status |",
        "|---|---|---|---|",
    ])
    for row in decision_rows:
        lines.append(
            f"| {row['candidate_explanation']} | {row['evidence']} | {row['counter_evidence']} | **{row['status']}** |"
        )
    lines.extend([
        "",
        "## 5. Protocol별 시각화",
        "",
    ])
    for protocol in protocols:
        lines.extend([
            f"### {protocol}",
            "",
            f"![{protocol} A/B/C support margin]({protocol}/plots/abc_support_margin_distribution.png)",
            "",
            f"![{protocol} A/B/C own support]({protocol}/plots/abc_own_support_distribution.png)",
            "",
            f"![{protocol} A/B/C rival support]({protocol}/plots/abc_rival_support_distribution.png)",
            "",
            f"![{protocol} paired B minus A]({protocol}/plots/paired_B_minus_A.png)",
            "",
            f"![{protocol} paired C minus B]({protocol}/plots/paired_C_minus_B.png)",
            "",
            f"![{protocol} correct wrong support]({protocol}/plots/correct_wrong_support.png)",
            "",
            f"![{protocol} reference count support]({protocol}/plots/reference_count_support.png)",
            "",
            f"![{protocol} class age support]({protocol}/plots/class_age_support.png)",
            "",
            f"![{protocol} pretrained final scatter]({protocol}/plots/pretrained_vs_final_support_scatter.png)",
            "",
            f"![{protocol} final minus pretrained support]({protocol}/plots/final_minus_pretrained_support.png)",
            "",
            f"![{protocol} representation correct wrong support]({protocol}/plots/representation_correct_wrong_support.png)",
            "",
        ])
    lines.extend([
        "## 6. 산출물",
        "",
        "- [`coverage_generalization_audit.md`](coverage_generalization_audit.md)",
        "- [`cross_protocol/summary_table.csv`](cross_protocol/summary_table.csv)",
        "- [`cross_protocol/decision_table.csv`](cross_protocol/decision_table.csv)",
        "- [`run_manifest.json`](run_manifest.json)",
        "",
        "각 protocol 폴더에는 요청된 bank 통계, query-level A/B/C, paired delta, pretrained/final support, probe 결과와 plot이 저장되어 있다.",
        "",
        "## 7. 한계",
        "",
        "1. Seed 1993 단일 seed다.",
        "2. Actual old feature는 diagnostic-only이며 replay-free method 성능으로 해석할 수 없다.",
        "3. Historical train split은 backbone이 과거에 본 sample이므로 representation-unseen validation이 아니다.",
        "4. Gaussian support 차이만으로 multimodality나 multi-center method의 필요성을 결론내릴 수 없다.",
        "5. Candidate-count stress test는 must-have 두 실험보다 우선순위가 낮아 이번 기본 실행에서 제외했다.",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "final": str(report),
        "summary": str(cross / "summary_table.csv"),
        "decisions": str(cross / "decision_table.csv"),
    }
