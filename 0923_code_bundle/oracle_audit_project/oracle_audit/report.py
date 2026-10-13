from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from oracle_audit.metrics import pearson, spearman


def _csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _mean(rows: list[dict], key: str) -> float:
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "INCONCLUSIVE"


def _correlation(rows: list[dict], x_key: str, y_key: str, rank: bool = True) -> float:
    x = np.asarray([_number(row, x_key) for row in rows], dtype=float)
    y = np.asarray([_number(row, y_key) for row in rows], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(keep.sum()) < 2:
        return float("nan")
    return spearman(x[keep], y[keep]) if rank else pearson(x[keep], y[keep])


def _correlation_value(rows: list[dict], predictor: str, outcome: str) -> float:
    matched = [
        row for row in rows
        if row.get("task") == "all"
        and row.get("predictor") == predictor
        and row.get("outcome") == outcome
    ]
    return _number(matched[0], "spearman") if matched else float("nan")


def _final_condition_rows(rows: list[dict]) -> tuple[int | None, dict[str, dict]]:
    names = {"A_STRUCTURE", "B_ORACLE_MEAN", "C_ORACLE_COVARIANCE", "D_FULL_ORACLE"}
    diagnostic = [row for row in rows if row.get("condition") in names]
    final_task = max((int(row["task"]) for row in diagnostic), default=None)
    selected = {
        row["condition"]: row
        for row in diagnostic
        if final_task is not None and int(row["task"]) == final_task
    }
    return final_task, selected


def generate_protocol_report(output_dir: Path) -> Path:
    manifest_path = output_dir / "audit_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    prototypes = _csv(output_dir / "prototype_oracle_audit.csv")
    apr = _csv(output_dir / "apr_ablation_audit.csv")
    drift = _csv(output_dir / "apr_oracle_drift_audit.csv")
    covariance = _csv(output_dir / "covariance_oracle_audit.csv")
    mahalanobis = _csv(output_dir / "mahalanobis_oracle_audit.csv")
    classifiers = _csv(output_dir / "prototype_classifier_audit.csv")
    correlations = _csv(output_dir / "class_forgetting_correlation.csv")
    rankings = _csv(output_dir / "mahalanobis_nearest_old_ranking.csv")
    replacements = _csv(output_dir / "oracle_classifier_replacement.csv")
    competition = _csv(output_dir / "new_class_competition_diagnostics.csv")
    margins = _csv(output_dir / "margin_diagnostics.csv")
    gaussian = _csv(output_dir / "gaussian_assumption_diagnostics.csv")

    primary_old = [
        row for row in prototypes
        if row.get("oracle_split") == "train" and row.get("is_old") == "True"
    ]
    cov_old = [
        row for row in covariance
        if row.get("oracle_split") == "train" and row.get("is_old") == "True"
    ]
    diag_mahal = [
        row for row in mahalanobis
        if row.get("oracle_split") == "train"
        and row.get("is_old") == "True"
        and row.get("covariance_mode") == "diagonal"
    ]
    apr_counts = Counter(row.get("APR_result") for row in apr)
    classifier_eval = [row for row in classifiers if row.get("oracle_split") == "test"]
    if not classifier_eval:
        classifier_eval = [row for row in classifiers if row.get("oracle_split") == "train"]
    final_classifier = max(classifier_eval, key=lambda row: int(row["task"])) if classifier_eval else {}

    age_cumulative = _correlation(primary_old, "class_age", "prototype_relative_l2", rank=False)
    age_apr = _correlation(apr, "class_age", "APR_error_reduction")
    proto_forgetting = _correlation_value(correlations, "prototype_relative_l2", "forgetting")
    boundary_forgetting = _correlation_value(correlations, "absolute_boundary_direction_error", "forgetting")
    proto_confusion = _correlation_value(correlations, "prototype_relative_l2", "old_to_new_confusion")
    boundary_confusion = _correlation_value(correlations, "absolute_boundary_direction_error", "old_to_new_confusion")

    final_head_task, heads = _final_condition_rows(replacements)
    head_a = heads.get("A_STRUCTURE", {})
    head_b = heads.get("B_ORACLE_MEAN", {})
    head_c = heads.get("C_ORACLE_COVARIANCE", {})
    head_d = heads.get("D_FULL_ORACLE", {})
    a_accuracy = _number(head_a, "seen_class_accuracy")
    delta_mean = _number(head_b, "seen_class_accuracy") - a_accuracy
    delta_cov = _number(head_c, "seen_class_accuracy") - a_accuracy
    delta_full = _number(head_d, "seen_class_accuracy") - a_accuracy

    final_comp_task = max((int(row["task"]) for row in competition), default=None)
    main_comp = next(
        (
            row for row in competition
            if row.get("condition") == "CHECKPOINT_MAIN"
            and final_comp_task is not None
            and int(row["task"]) == final_comp_task
        ),
        {},
    )
    single_new = _number(main_comp, "single_head_new_acc")
    new_only = _number(main_comp, "new_only_acc")
    competition_gap = _number(main_comp, "new_only_minus_single_head")
    main_margins = [
        row for row in margins
        if row.get("condition") == "CHECKPOINT_MAIN"
        and final_comp_task is not None
        and int(row["task"]) == final_comp_task
    ]
    new_to_old_rate = (
        float(np.mean([row.get("outcome") == "new_to_old" for row in main_margins]))
        if main_margins else float("nan")
    )
    margin_rows_for_correlation = [
        {
            **row,
            "new_to_old_indicator": float(row.get("outcome") == "new_to_old"),
        }
        for row in main_margins
    ]
    margin_confusion_corr = _correlation(
        margin_rows_for_correlation, "new_vs_old_margin", "new_to_old_indicator"
    )
    margin_prototype_corr = _correlation(
        main_margins, "new_vs_old_margin", "strongest_old_prototype_relative_l2"
    )
    margin_boundary_corr = _correlation(
        main_margins, "new_vs_old_margin", "strongest_old_boundary_direction_error"
    )

    mean_error = _mean(primary_old, "prototype_relative_l2")
    covariance_error = _mean(cov_old, "frobenius_relative_error")
    trace_ratio = _mean(cov_old, "trace_ratio")
    trace_rows = [row for row in cov_old if np.isfinite(_number(row, "delta_trace_structure"))]
    structure_trace_delta = _mean(trace_rows, "delta_trace_structure")
    oracle_trace_delta = _mean(trace_rows, "delta_trace_oracle")
    trace_increase_rate = (
        float(np.mean([_number(row, "delta_trace_structure") > 0 for row in trace_rows]))
        if trace_rows else float("nan")
    )
    divergent_trace_rate = (
        float(
            np.mean(
                [
                    _number(row, "delta_trace_structure") > 0
                    and _number(row, "delta_trace_oracle") <= 0
                    for row in trace_rows
                ]
            )
        )
        if trace_rows else float("nan")
    )

    # Explicit diagnostic thresholds keep automatic labels reproducible.
    cases: list[tuple[str, str]] = []
    if mean_error <= 0.20 and abs(delta_mean) < 1.0:
        cases.append(("A", "mean tracking은 주요 병목일 가능성이 낮음"))
    if mean_error >= 0.40 and delta_mean >= 1.0:
        cases.append(("B", "prototype tracking이 실제 classification 병목 후보"))
    if mean_error <= 0.20 and delta_cov >= 1.0:
        cases.append(("C", "covariance estimation/transport가 주요 병목 후보"))
    if abs(delta_full) < 1.0:
        cases.append(("D", "moment 보정보다 Gaussian 가정·CA·representation overlap을 더 의심"))
    if competition_gap >= 5.0:
        cases.append(("E", "old-new score competition/classifier bias가 유력"))
    if np.isfinite(new_only) and new_only < 50.0:
        cases.append(("F", "new-only도 낮아 current representation 손상 가능성"))
    case_text = "; ".join(f"Case {name}: {meaning}" for name, meaning in cases)
    if not case_text:
        case_text = "임계값을 만족하는 단일 Case가 없어 INCONCLUSIVE"

    evidence: list[tuple[float, str]] = []
    if delta_mean >= 1.0:
        evidence.append((delta_mean, "prototype tracking"))
    if delta_cov >= 1.0:
        evidence.append((delta_cov, "covariance estimation"))
    if competition_gap >= 5.0:
        evidence.append((competition_gap, "old-new score competition"))
    if np.isfinite(new_only) and new_only < 50.0 and competition_gap < 5.0:
        evidence.append((50.0 - new_only, "representation learning"))
    if abs(delta_full) < 1.0 and not evidence:
        gaussian_deviation = _mean(gaussian, "mean_absolute_excess_kurtosis")
        evidence.append(
            (1.0, "Gaussian approximation" if gaussian_deviation >= 1.0 else "classifier adaptation")
        )
    bottleneck = max(evidence)[1] if evidence else "inconclusive"
    scientific = bool(manifest.get("scientific_interpretation_allowed", True))
    if not scientific:
        case_text = "SANITY ONLY — 축소/override 실행에서는 Case A–F를 과학적으로 판정하지 않음"
        bottleneck = "inconclusive (sanity override)"

    complete = bool(manifest.get("complete_temporal_coverage"))
    initialization = head_a.get("initialization_source", "not-run")
    legacy_warning = ""
    if initialization == "shared_post_ca_legacy_init":
        legacy_warning = " Legacy checkpoint에는 pre-CA sidecar가 없어 본 CA의 exact replay가 아니며, 동일 post-CA 시작점에서 네 statistics 조건의 상대 효과만 비교한다."
    apr_summary = ", ".join(
        f"`{key}`={value}" for key, value in sorted(apr_counts.items()) if key
    ) or "available row 없음"

    lines = [
        f"# Structure prototype oracle audit — {protocol}",
        "",
        f"> 상태: **{'COMPLETE' if complete else 'PARTIAL'}** · mode: **{'SCIENTIFIC' if scientific else 'SANITY ONLY'}** · audited tasks: `{manifest['audited_tasks']}` · primary oracle: `oracle_train_statistics`",
        "",
        "이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다." + legacy_warning,
        "",
        "## 핵심 수치",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| old-class Structure↔oracle cosine 평균 | {_fmt(_mean(primary_old, 'structure_cosine'))} |",
        f"| old-class prototype relative L2 평균 | {_fmt(mean_error)} |",
        f"| APR error reduction 평균 | {_fmt(_mean(apr, 'APR_error_reduction'))} |",
        f"| one-step drift error 평균 | {_fmt(_mean(apr, 'one_step_drift_error'))} |",
        f"| cumulative prototype error 평균 | {_fmt(_mean(apr, 'cumulative_error'))} |",
        f"| covariance Frobenius relative error 평균 | {_fmt(covariance_error)} |",
        f"| covariance trace ratio 평균 | {_fmt(trace_ratio)} |",
        f"| A→B Oracle Mean gain (pp) | {_fmt(delta_mean)} |",
        f"| A→C Oracle Covariance gain (pp) | {_fmt(delta_cov)} |",
        f"| A→D Full Oracle gain (pp) | {_fmt(delta_full)} |",
        f"| final new-only−single-head new (pp) | {_fmt(competition_gap)} |",
        "",
        f"APR class 집계: {apr_summary}.",
        "",
        "## Final-task oracle classifier replacement",
        "",
        f"진단 task: `{final_head_task if final_head_task is not None else 'not run'}` · initialization: `{initialization}`",
        "",
        "| condition | seen acc | old acc | new acc | old-only | new-only | Δ vs A |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A_STRUCTURE", "B_ORACLE_MEAN", "C_ORACLE_COVARIANCE", "D_FULL_ORACLE"):
        row = heads.get(name, {})
        lines.append(
            f"| {name} | {_fmt(_number(row, 'seen_class_accuracy'))} | "
            f"{_fmt(_number(row, 'old_accuracy'))} | {_fmt(_number(row, 'new_accuracy'))} | "
            f"{_fmt(_number(row, 'old_only_accuracy'))} | {_fmt(_number(row, 'new_only_accuracy'))} | "
            f"{_fmt(_number(row, 'delta_vs_A'))} |"
        )
    lines.extend(
        [
            "",
            "## Case A–F 자동 판정",
            "",
            "임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.",
            "",
            case_text + ".",
            "",
            "## 추가 Q1–Q10 답변",
            "",
            f"1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `{_fmt(_mean(apr, 'APR_error_reduction'))}`, improved 비율은 `{_fmt(apr_counts.get('improved', 0) / max(len(apr), 1))}`다.",
            f"2. **Q2 일부 class는 악화되는가?** worsened는 `{apr_counts.get('worsened', 0)}/{len(apr)}` class-task다.",
            f"3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `{_fmt(age_apr)}`, age↔cumulative relative error Pearson `{_fmt(age_cumulative)}`다.",
            f"4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `{_fmt(delta_mean)}` pp다.",
            f"5. **Q5 Oracle Covariance 회복량:** C−A는 `{_fmt(delta_cov)}` pp다.",
            f"6. **Q6 Full Oracle 회복량:** D−A는 `{_fmt(delta_full)}` pp다.",
            f"7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `{_fmt(proto_forgetting)}`, boundary `{_fmt(boundary_forgetting)}`이고 old→new confusion은 L2 `{_fmt(proto_confusion)}`, boundary `{_fmt(boundary_confusion)}`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.",
            f"8. **Q8 representation 손상인가 competition인가?** single-head new `{_fmt(single_new)}`, new-only `{_fmt(new_only)}`, gap `{_fmt(competition_gap)}` pp, sample-level new→old 비율 `{_fmt(new_to_old_rate)}`다. margin↔new→old Spearman `{_fmt(margin_confusion_corr)}`, margin↔strongest-old prototype error `{_fmt(margin_prototype_corr)}`, margin↔boundary error `{_fmt(margin_boundary_corr)}`다.",
            f"9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `{_fmt(trace_ratio)}`, Structure Δtrace `{_fmt(structure_trace_delta)}`, oracle Δtrace `{_fmt(oracle_trace_delta)}`, Structure 증가 비율 `{_fmt(trace_increase_rate)}`, Structure↑이면서 oracle≤0인 비율 `{_fmt(divergent_trace_rate)}`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.",
            f"10. **Q10 현재 병목:** 자동 근거상 **{bottleneck}**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.",
            "",
            "## 기존 audit 및 Gaussian 보조 결과",
            "",
            f"- APR/true one-step drift cosine `{_fmt(_mean(apr, 'one_step_drift_cosine'))}`; legacy drift CSV cosine `{_fmt(_mean(drift, 'drift_cosine'))}`.",
            f"- diagonal covariance relative L2 `{_fmt(_mean(cov_old, 'diag_relative_l2'))}`, diagonal Mahalanobis D² Pearson `{_fmt(_mean(diag_mahal, 'distance_pearson'))}`.",
            f"- nearest-old ranking 변경률 `{_fmt(_mean(rankings, 'nearest_old_changed_rate'))}`; 기존 oracle nearest-prototype final gain `{_fmt(_number(final_classifier, 'oracle_minus_structure_gap'))}` pp.",
            f"- mean |skew| `{_fmt(_mean(gaussian, 'mean_absolute_skewness'))}`, mean |excess kurtosis| `{_fmt(_mean(gaussian, 'mean_absolute_excess_kurtosis'))}`, D²/dimension `{_fmt(_mean(gaussian, 'd2_mean_over_dimension'))}`.",
            "",
            "## 해석 제한",
            "",
            "- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.",
            "- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.",
            "- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.",
            "- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.",
            "- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.",
        ]
    )
    destination = output_dir / "structure_prototype_oracle_audit.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def generate_combined_report(audit_root: Path) -> Path:
    records = []
    for path in sorted(audit_root.glob("*/seed_*/audit_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        directory = path.parent
        prototypes = [
            row for row in _csv(directory / "prototype_oracle_audit.csv")
            if row.get("oracle_split") == "train" and row.get("is_old") == "True"
        ]
        _, heads = _final_condition_rows(_csv(directory / "oracle_classifier_replacement.csv"))
        a = _number(heads.get("A_STRUCTURE", {}), "seen_class_accuracy")
        competition = _csv(directory / "new_class_competition_diagnostics.csv")
        final_task = max((int(row["task"]) for row in competition), default=None)
        main = next(
            (
                row for row in competition
                if row.get("condition") == "CHECKPOINT_MAIN"
                and final_task is not None and int(row["task"]) == final_task
            ),
            {},
        )
        records.append(
            {
                "protocol": manifest["protocol"],
                "dataset": manifest["dataset"],
                "coverage": "complete" if manifest.get("complete_temporal_coverage") else "partial",
                "tasks": manifest["audited_tasks"],
                "relative_l2": _mean(prototypes, "prototype_relative_l2"),
                "mean_gain": _number(heads.get("B_ORACLE_MEAN", {}), "seen_class_accuracy") - a,
                "cov_gain": _number(heads.get("C_ORACLE_COVARIANCE", {}), "seen_class_accuracy") - a,
                "full_gain": _number(heads.get("D_FULL_ORACLE", {}), "seen_class_accuracy") - a,
                "competition_gap": _number(main, "new_only_minus_single_head"),
            }
        )
    lines = [
        "# Structure prototype oracle audit — combined",
        "",
        "| protocol | dataset | coverage | tasks | relative L2 | Oracle Mean Δpp | Oracle Cov Δpp | Full Oracle Δpp | new-only gap pp |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['protocol']} | {row['dataset']} | {row['coverage']} | `{row['tasks']}` | "
            f"{_fmt(row['relative_l2'])} | {_fmt(row['mean_gain'])} | {_fmt(row['cov_gain'])} | "
            f"{_fmt(row['full_gain'])} | {_fmt(row['competition_gap'])} |"
        )
    complete = {row["protocol"]: row for row in records if row["coverage"] == "complete"}

    def pair(short: str, long: str) -> str:
        if short not in complete or long not in complete:
            return f"{short}/{long} complete pair 없음: INCONCLUSIVE"
        return (
            f"{short} relative L2={_fmt(complete[short]['relative_l2'])}, "
            f"{long}={_fmt(complete[long]['relative_l2'])}"
        )

    lines.extend(
        [
            "",
            "## Cross-protocol 상태",
            "",
            f"- {pair('IA5', 'IA10')}",
            f"- {pair('CAR5', 'CAR10')}",
            "- 위 표의 A–D delta는 diagnostic head 결과이며 EF-CIL main accuracy와 분리해서 해석한다.",
        ]
    )
    destination = audit_root / "structure_prototype_oracle_audit.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
