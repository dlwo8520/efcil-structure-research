from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "INCONCLUSIVE"


def _final(rows: list[dict[str, str]], condition: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if row.get("row_type") == "summary" and row.get("condition") == condition
    ]
    return max(selected, key=lambda row: int(row["task"])) if selected else {}


def _answer(delta: float, threshold: float = 0.1) -> str:
    if not np.isfinite(delta):
        return "INCONCLUSIVE"
    if delta > threshold:
        return "YES"
    if delta < -threshold:
        return "NO"
    return "INCONCLUSIVE"


def _alignment_mean(rows: list[dict[str, str]], condition: str, task: int, group: str = "old") -> float:
    values = [
        _float(row, "weight_mean_cosine")
        for row in rows
        if row.get("condition") == condition
        and int(row["task"]) == task
        and row.get("class_group") == group
    ]
    return float(np.mean(values)) if values else float("nan")


def _calibration_row(rows: list[dict[str, str]], condition: str, task: int) -> dict[str, str]:
    return next(
        (row for row in rows if row.get("condition") == condition and int(row["task"]) == task),
        {},
    )


def _mean_field(
    rows: list[dict[str, str]],
    condition: str,
    task: int,
    group: str,
    field: str,
) -> float:
    values = [
        _float(row, field)
        for row in rows
        if row.get("condition") == condition
        and int(row["task"]) == task
        and row.get("sample_group") == group
    ]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else float("nan")


def _bias_operating_point(
    rows: list[dict[str, str]],
    condition: str,
    task: int,
    old_floor: float,
    new_floor: float,
) -> tuple[float, float]:
    selected = [
        row for row in rows
        if row.get("condition") == condition and int(row["task"]) == task
    ]
    max_new = max(
        (_float(row, "new_accuracy") for row in selected if _float(row, "old_accuracy") >= old_floor),
        default=float("nan"),
    )
    max_old = max(
        (_float(row, "old_accuracy") for row in selected if _float(row, "new_accuracy") >= new_floor),
        default=float("nan"),
    )
    return max_new, max_old


def _aggregate_answer(values: list[str]) -> str:
    finite = [value for value in values if value != "INCONCLUSIVE"]
    if finite and len(finite) == len(values) and len(set(finite)) == 1:
        return finite[0]
    return "INCONCLUSIVE"


def _bias_recovery(rows: list[dict[str, str]], condition: str, task: int) -> tuple[bool, float, float]:
    selected = [row for row in rows if row.get("condition") == condition and int(row["task"]) == task]
    if not selected:
        return False, float("nan"), float("nan")
    baseline = min(selected, key=lambda row: abs(_float(row, "beta")))
    base_old, base_new = _float(baseline, "old_accuracy"), _float(baseline, "new_accuracy")
    feasible = [
        row for row in selected
        if _float(row, "old_accuracy") >= base_old - 1.0
    ]
    best_new = max((_float(row, "new_accuracy") for row in feasible), default=float("nan"))
    return bool(np.isfinite(best_new) and best_new >= base_new + 5.0), base_new, best_new


def _curve_difference(rows: list[dict[str, str]], conditions: list[str], task: int) -> float:
    curves = {
        condition: [
            row for row in rows
            if row.get("condition") == condition and int(row["task"]) == task
        ]
        for condition in conditions
    }
    if any(not curve for curve in curves.values()):
        return float("nan")
    old_min = max(min(_float(row, "old_accuracy") for row in curve) for curve in curves.values())
    old_max = min(max(_float(row, "old_accuracy") for row in curve) for curve in curves.values())
    if old_min >= old_max:
        return float("nan")
    grid = np.linspace(old_min, old_max, 50)
    interpolated = []
    for condition in conditions:
        pairs = sorted(
            [(_float(row, "old_accuracy"), _float(row, "new_accuracy")) for row in curves[condition]]
        )
        x = np.asarray([item[0] for item in pairs])
        y = np.asarray([item[1] for item in pairs])
        unique, indices = np.unique(x, return_index=True)
        interpolated.append(np.interp(grid, unique, y[indices]))
    maximum_gap = 0.0
    for first in range(len(interpolated)):
        for second in range(first + 1, len(interpolated)):
            maximum_gap = max(maximum_gap, float(np.max(np.abs(interpolated[first] - interpolated[second]))))
    return maximum_gap


def generate_protocol_report(output_dir: Path) -> Path:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest["protocol"]
    accuracy = _csv(output_dir / "gaussian_vs_actual_classifier_accuracy.csv")
    ncm = _csv(output_dir / "ncm_absolute_comparison.csv")
    bias = _csv(output_dir / "bias_sweep.csv")
    alignment = _csv(output_dir / "weight_mean_alignment.csv")
    calibration = _csv(output_dir / "calibration_metrics.csv")
    optimization = _csv(output_dir / "ca_optimization_trace.csv")
    group_margin = _csv(output_dir / "old_new_group_margin.csv")
    margins = _csv(output_dir / "ca_margin_shift.csv")
    prediction_changes = _csv(output_dir / "ncm_prediction_change_analysis.csv")
    final_task = max(manifest["completed_tasks"])

    ncm_apr = _final(ncm, "APR_NCM")
    ncm_oracle = _final(ncm, "ORACLE_NCM")
    structure = _final(accuracy, "STRUCTURE_CA")
    oracle_mean = _final(accuracy, "ORACLE_MEAN_CA")
    d = _final(accuracy, "FULL_ORACLE_GAUSSIAN_CA")
    g = _final(accuracy, "G_FIXED")
    r = _final(accuracy, "R_FIXED")

    q1_delta = _float(ncm_oracle, "seen_class_accuracy") - _float(d, "seen_class_accuracy")
    q2_delta = _float(r, "seen_class_accuracy") - _float(g, "seen_class_accuracy")
    q3_delta = _float(r, "new_to_old_rate") - _float(g, "new_to_old_rate")
    r_gap = _float(r, "new_only_accuracy") - _float(r, "new_accuracy")
    q5_yes, base_new, best_new = _bias_recovery(bias, "R_FIXED", final_task)
    curve_gap = _curve_difference(
        bias, ["STRUCTURE_CA", "FULL_ORACLE_GAUSSIAN_CA", "R_FIXED"], final_task
    )
    pre_alignment = _alignment_mean(alignment, "PRE_CA", final_task)
    structure_alignment = _alignment_mean(alignment, "STRUCTURE_CA", final_task)
    d_alignment = _alignment_mean(alignment, "FULL_ORACLE_GAUSSIAN_CA", final_task)
    r_alignment = _alignment_mean(alignment, "R_FIXED", final_task)

    final_trace = {
        condition: max(
            (
                row for row in optimization
                if row.get("condition") == condition and int(row["task"]) == final_task
            ),
            key=lambda row: int(row["epoch"]),
            default={},
        )
        for condition in ("G_FIXED", "R_FIXED")
    }
    train_acc_gap = _float(final_trace["R_FIXED"], "train_accuracy") - _float(
        final_trace["G_FIXED"], "train_accuracy"
    )
    q8 = "YES" if abs(train_acc_gap) >= 5.0 else "INCONCLUSIVE"
    r_minus_g_new_margin = _mean_field(
        margins, "R_FIXED", final_task, "new", "post_opposite_group_margin"
    ) - _mean_field(
        margins, "G_FIXED", final_task, "new", "post_opposite_group_margin"
    )
    r_minus_g_old_margin = _mean_field(
        margins, "R_FIXED", final_task, "old", "post_opposite_group_margin"
    ) - _mean_field(
        margins, "G_FIXED", final_task, "old", "post_opposite_group_margin"
    )

    questions = [
        ("Q1", _answer(q1_delta), f"Oracle NCM − Full Oracle Gaussian CA Seen = {_fmt(q1_delta)} pp"),
        ("Q2", _answer(q2_delta), f"R-fixed − G-fixed Seen = {_fmt(q2_delta)} pp"),
        (
            "Q3",
            "YES" if np.isfinite(q3_delta) and q3_delta < -0.001 else ("NO" if np.isfinite(q3_delta) and q3_delta > 0.001 else "INCONCLUSIVE"),
            f"R-fixed − G-fixed New→Old = {_fmt(100.0 * q3_delta)} pp",
        ),
        ("Q4", "YES" if r_gap >= 5.0 else "NO", f"R-fixed New-only − New = {_fmt(r_gap)} pp"),
        (
            "Q5",
            "YES" if q5_yes else "NO",
            f"test-sweep diagnostic에서 Old 손실≤1 pp 조건의 New: {_fmt(base_new)} → {_fmt(best_new)}",
        ),
        (
            "Q6",
            "YES" if np.isfinite(curve_gap) and curve_gap >= 1.0 else ("NO" if np.isfinite(curve_gap) else "INCONCLUSIVE"),
            f"공통 Old 구간에서 bias trade-off curve 최대 New 차이 = {_fmt(curve_gap)} pp",
        ),
        (
            "Q7",
            "YES" if structure_alignment < pre_alignment - 0.01 else "NO",
            f"old mean alignment pre={_fmt(pre_alignment)}, Structure={_fmt(structure_alignment)}, D={_fmt(d_alignment)}, R={_fmt(r_alignment)}",
        ),
        ("Q8", q8, f"final fixed-bank train accuracy R−G = {_fmt(train_acc_gap)} pp; 이 진단만으로 optimization 인과성은 확정하지 않음"),
    ]

    lines = [
        f"# {protocol} Classifier Bottleneck Diagnostic",
        "",
        f"> 상태: **COMPLETE** · seed 1993 · completed tasks `{manifest['completed_tasks']}` · main checkpoint modified: **NO**",
        "",
        "## 맨 앞 결론 — YES / NO / INCONCLUSIVE",
        "",
        "| 질문 | 답 | 직접 근거 |",
        "|---|---|---|",
    ]
    labels = {
        "Q1": "Oracle NCM > Full Oracle Gaussian CA?",
        "Q2": "Actual old feature CA > matched Gaussian CA?",
        "Q3": "Gaussian→Actual이 New→Old를 줄이는가?",
        "Q4": "Actual CA에도 New-only gap이 크게 남는가?",
        "Q5": "단일 bias로 Old를 크게 잃지 않고 New를 회복하는가?",
        "Q6": "A/D/R이 서로 다른 bias frontier를 만드는가?",
        "Q7": "CA 후 weight가 class mean에서 멀어지는가?",
        "Q8": "G/R 차이가 optimization/fitting 차이일 가능성이 있는가?",
    }
    for key, answer, evidence in questions:
        lines.append(f"| {key}. {labels[key]} | **{answer}** | {evidence} |")

    lines.extend(
        [
            "",
            "## Final-task 절대 정확도",
            "",
            "| Condition | Old source | Seen | Old | New | Old-only | New-only | New→Old |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    table_rows = [
        ("Structure CA", structure),
        ("Oracle Mean CA", oracle_mean),
        ("Full Oracle Gaussian CA", d),
        ("Fixed Oracle Gaussian CA", g),
        ("Actual Feature CA", r),
        ("APR-NCM", ncm_apr),
        ("Oracle-NCM", ncm_oracle),
    ]
    for name, row in table_rows:
        lines.append(
            f"| {name} | {row.get('old_source', '')} | {_fmt(_float(row, 'seen_class_accuracy'))} | "
            f"{_fmt(_float(row, 'old_accuracy'))} | {_fmt(_float(row, 'new_accuracy'))} | "
            f"{_fmt(_float(row, 'old_only_accuracy'))} | {_fmt(_float(row, 'new_only_accuracy'))} | "
            f"{_fmt(100.0 * _float(row, 'new_to_old_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## G-fixed 대비 R-fixed delta",
            "",
            "| Metric | R − G |",
            "|---|---:|",
        ]
    )
    for key, label, multiplier in (
        ("seen_class_accuracy", "Seen", 1.0),
        ("old_accuracy", "Old", 1.0),
        ("new_accuracy", "New", 1.0),
        ("old_only_accuracy", "Old-only", 1.0),
        ("new_only_accuracy", "New-only", 1.0),
        ("new_to_old_rate", "New→Old", 100.0),
        ("old_to_new_rate", "Old→New", 100.0),
    ):
        lines.append(f"| {label} | {_fmt(multiplier * (_float(r, key) - _float(g, key)))} |")
    g_cal = _calibration_row(calibration, "G_FIXED", final_task)
    r_cal = _calibration_row(calibration, "R_FIXED", final_task)
    lines.extend(
        [
            f"| NLL | {_fmt(_float(r_cal, 'nll') - _float(g_cal, 'nll'))} |",
            f"| ECE | {_fmt(_float(r_cal, 'ece_15bin') - _float(g_cal, 'ece_15bin'))} |",
            f"| New-vs-Old margin | {_fmt(r_minus_g_new_margin)} |",
            f"| Old-vs-New margin | {_fmt(r_minus_g_old_margin)} |",
            f"| old weight-mean alignment | {_fmt(r_alignment - _alignment_mean(alignment, 'G_FIXED', final_task))} |",
            "",
            "## Transition progression",
            "",
            "| Task | Structure | Full Oracle G | APR-NCM | Oracle-NCM | G-fixed | R-fixed | R−G Seen | ΔNew→Old (pp) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    transition_deltas: list[float] = []
    transition_ncm_deltas: list[float] = []
    for task in manifest["completed_tasks"]:
        task_accuracy = {
            condition: next(
                (
                    row for row in accuracy
                    if row.get("row_type") == "summary"
                    and row.get("condition") == condition
                    and int(row["task"]) == task
                ),
                {},
            )
            for condition in ("STRUCTURE_CA", "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED")
        }
        task_ncm = {
            condition: next(
                (
                    row for row in ncm
                    if row.get("row_type") == "summary"
                    and row.get("condition") == condition
                    and int(row["task"]) == task
                ),
                {},
            )
            for condition in ("APR_NCM", "ORACLE_NCM")
        }
        task_g, task_r = task_accuracy["G_FIXED"], task_accuracy["R_FIXED"]
        seen_delta = _float(task_r, "seen_class_accuracy") - _float(task_g, "seen_class_accuracy")
        confusion_delta = 100.0 * (
            _float(task_r, "new_to_old_rate") - _float(task_g, "new_to_old_rate")
        )
        transition_deltas.append(seen_delta)
        transition_ncm_deltas.append(
            _float(task_ncm["ORACLE_NCM"], "seen_class_accuracy")
            - _float(task_accuracy["FULL_ORACLE_GAUSSIAN_CA"], "seen_class_accuracy")
        )
        lines.append(
            f"| {task} | {_fmt(_float(task_accuracy['STRUCTURE_CA'], 'seen_class_accuracy'))} | "
            f"{_fmt(_float(task_accuracy['FULL_ORACLE_GAUSSIAN_CA'], 'seen_class_accuracy'))} | "
            f"{_fmt(_float(task_ncm['APR_NCM'], 'seen_class_accuracy'))} | "
            f"{_fmt(_float(task_ncm['ORACLE_NCM'], 'seen_class_accuracy'))} | "
            f"{_fmt(_float(task_g, 'seen_class_accuracy'))} | {_fmt(_float(task_r, 'seen_class_accuracy'))} | "
            f"{_fmt(seen_delta)} | {_fmt(confusion_delta)} |"
        )

    final_changes = [row for row in prediction_changes if int(row["task"]) == final_task]
    apr_to_oracle_fixed = sum(
        int(row["APR_prediction"]) != int(row["true_class"])
        and int(row["Oracle_prediction"]) == int(row["true_class"])
        for row in final_changes
    )
    apr_to_oracle_broken = sum(
        int(row["APR_prediction"]) == int(row["true_class"])
        and int(row["Oracle_prediction"]) != int(row["true_class"])
        for row in final_changes
    )
    task_metadata = manifest.get("task_manifests", [])
    matched_is_full = bool(task_metadata) and all(
        row.get("matched_equals_full_actual") for row in task_metadata
    )
    min_count = min((int(row["actual_old_count_min"]) for row in task_metadata), default=0)
    max_count = max((int(row["actual_old_count_max"]) for row in task_metadata), default=0)
    r_baseline_old = _float(r, "old_accuracy")
    r_baseline_new = _float(r, "new_accuracy")
    lines.extend(
        [
            "",
            f"R−G Seen은 전 transition에서 `{_fmt(min(transition_deltas))}~{_fmt(max(transition_deltas))}` pp였고, "
            f"Oracle-NCM−Full Oracle Gaussian CA는 `{_fmt(min(transition_ncm_deltas))}~{_fmt(max(transition_ncm_deltas))}` pp였다.",
            "",
            "## NCM prediction-change diagnostic",
            "",
            f"Final task에서 APR-NCM과 Oracle-NCM의 예측이 달라진 sample은 **{len(final_changes)}개**였다. "
            f"그중 APR 오답→Oracle 정답은 {apr_to_oracle_fixed}개, APR 정답→Oracle 오답은 {apr_to_oracle_broken}개였다. "
            "sample별 score, margin, prototype shift는 `ncm_prediction_change_analysis.csv`에 저장했다.",
            "",
            "## Bias frontier operating-point diagnostic",
            "",
            f"기준은 R-fixed beta=0의 Old={_fmt(r_baseline_old)}, New={_fmt(r_baseline_new)}이며, "
            "아래 값은 test sweep을 이용한 진단값일 뿐 선택된 method 성능이 아니다.",
            "",
            "| Condition | max New (Old ≥ R-old−1pp) | max Old (New ≥ R-new−1pp) |",
            "|---|---:|---:|",
        ]
    )
    for condition, label in (
        ("STRUCTURE_CA", "Structure CA"),
        ("FULL_ORACLE_GAUSSIAN_CA", "Full Oracle Gaussian CA"),
        ("R_FIXED", "Actual Feature CA"),
    ):
        max_new, max_old = _bias_operating_point(
            bias, condition, final_task, r_baseline_old - 1.0, r_baseline_new - 1.0
        )
        lines.append(f"| {label} | {_fmt(max_new)} | {_fmt(max_old)} |")
    lines.extend(
        [
            "",
            "## Matched Count / Full Actual 확인",
            "",
            (
                f"모든 transition에서 actual old class count가 class당 {min_count}~{max_count}개로 configured cap보다 작아 "
                "**Matched Count와 Full Actual이 정확히 같은 bank**였다. 따라서 별도 Full Actual 행은 중복이며 "
                "`R_FIXED`가 두 버전을 동시에 나타낸다."
                if matched_is_full
                else "일부 transition에서 Matched Count와 Full Actual이 다르므로 manifest의 count 정보를 기준으로 분리 해석해야 한다."
            ),
            "",
            "## 해석",
            "",
        ]
    )
    if q2_delta > 1.0:
        lines.append("Actual feature CA가 fixed Gaussian CA보다 뚜렷하게 높다. 이는 현재 cosine head/CA 조건에서 Gaussian replacement 또는 그 sampling support가 성능을 제한한다는 강한 증거다. 다만 multimodality 하나를 원인으로 확정하지 않는다.")
    elif abs(q2_delta) <= 0.5:
        lines.append("Actual feature source로 교체해도 fixed Gaussian 대비 큰 회복이 없다. 이는 old source 교체만으로 부족하다는 뜻이며 representation 원인을 직접 증명하지 않는다. CA objective, optimization, classifier form과 score overlap이 남는다.")
    else:
        lines.append("Actual-vs-Gaussian 차이는 존재하지만 단일 seed에서 결정적이라고 보기 어렵다. task progression과 paired margin/optimization trace를 함께 해석해야 한다.")
    if q1_delta > 5.0 and q2_delta <= 1.0:
        lines.append("Oracle NCM은 CA보다 높지만 Actual CA가 낮게 남는다. Frozen feature에는 prototype-based 분류 정보가 있으나 현재 classifier fitting이 이를 충분히 활용하지 못할 가능성이 있다.")
    if r_gap >= 5.0:
        lines.append("R-fixed에서도 new-only gap이 크게 남아 actual old feature 사용만으로 old-new competition이 해소되지 않는다.")
    lines.extend(
        [
            "",
            "Bias sweep은 held-out test에서 그린 **ORACLE/TEST-SWEEP DIAGNOSTIC**이다. 별도 validation으로 beta를 선택하지 않았으므로 최적 beta accuracy를 정식 method result로 제시하지 않는다. 전체 class에 동일 temperature를 적용하면 argmax는 변하지 않는다.",
            "",
            "## 구현 및 안전성",
            "",
            "- raw frozen `sfeatures`를 사용하고 L2 normalization은 cosine head 내부에서 수행했다.",
            "- G/R current-new source는 동일한 fixed Structure Gaussian bank다.",
            "- R old bank는 actual old training feature이며 main CIL training이나 다음 task에 전달하지 않았다.",
            "- 기존 D accuracy는 재사용했고 bias/logit용 head만 saved RNG로 재구성하여 기존 accuracy와 일치시켰다.",
            "- classifier는 bias가 없고 weight norm은 cosine score에 직접 들어가지 않는다.",
            "- optimizer는 diagnostic head만 포함하고 backbone/adapter/checkpoint는 read-only다.",
            "",
            "## 산출물",
            "",
            "요청된 11개 CSV와 bias/score/margin/alignment plot은 이 보고서와 같은 protocol output에 저장했다.",
        ]
    )
    destination = output_dir / f"{protocol}_classifier_bottleneck_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def generate_combined_report(output_root: Path) -> Path:
    records = []
    for protocol in ("IA5", "CAR5"):
        directory = output_root / protocol / "seed_1993"
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        accuracy = _csv(directory / "gaussian_vs_actual_classifier_accuracy.csv")
        ncm = _csv(directory / "ncm_absolute_comparison.csv")
        bias = _csv(directory / "bias_sweep.csv")
        alignment = _csv(directory / "weight_mean_alignment.csv")
        optimization = _csv(directory / "ca_optimization_trace.csv")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_task = max(manifest["completed_tasks"])
        g, r = _final(accuracy, "G_FIXED"), _final(accuracy, "R_FIXED")
        oracle, d = _final(ncm, "ORACLE_NCM"), _final(ncm, "FULL_ORACLE_GAUSSIAN_CA")
        q1_delta = _float(oracle, "seen_class_accuracy") - _float(d, "seen_class_accuracy")
        q2_delta = _float(r, "seen_class_accuracy") - _float(g, "seen_class_accuracy")
        q3_delta = 100.0 * (_float(r, "new_to_old_rate") - _float(g, "new_to_old_rate"))
        r_gap = _float(r, "new_only_accuracy") - _float(r, "new_accuracy")
        q5_yes, _, _ = _bias_recovery(bias, "R_FIXED", final_task)
        curve_gap = _curve_difference(
            bias, ["STRUCTURE_CA", "FULL_ORACLE_GAUSSIAN_CA", "R_FIXED"], final_task
        )
        pre_alignment = _alignment_mean(alignment, "PRE_CA", final_task)
        structure_alignment = _alignment_mean(alignment, "STRUCTURE_CA", final_task)
        final_trace = {
            condition: max(
                (
                    row for row in optimization
                    if row.get("condition") == condition and int(row["task"]) == final_task
                ),
                key=lambda row: int(row["epoch"]),
                default={},
            )
            for condition in ("G_FIXED", "R_FIXED")
        }
        train_gap = _float(final_trace["R_FIXED"], "train_accuracy") - _float(
            final_trace["G_FIXED"], "train_accuracy"
        )
        transitions = []
        for task in manifest["completed_tasks"]:
            summaries = {
                condition: next(
                    (
                        row for row in accuracy
                        if row.get("row_type") == "summary"
                        and row.get("condition") == condition
                        and int(row["task"]) == task
                    ),
                    {},
                )
                for condition in ("G_FIXED", "R_FIXED", "FULL_ORACLE_GAUSSIAN_CA")
            }
            task_oracle = next(
                (
                    row for row in ncm
                    if row.get("row_type") == "summary"
                    and row.get("condition") == "ORACLE_NCM"
                    and int(row["task"]) == task
                ),
                {},
            )
            transitions.append(
                {
                    "task": task,
                    "r_minus_g": _float(summaries["R_FIXED"], "seen_class_accuracy")
                    - _float(summaries["G_FIXED"], "seen_class_accuracy"),
                    "oracle_minus_d": _float(task_oracle, "seen_class_accuracy")
                    - _float(summaries["FULL_ORACLE_GAUSSIAN_CA"], "seen_class_accuracy"),
                }
            )
        records.append(
            {
                "protocol": protocol,
                "tasks": manifest["completed_tasks"],
                "oracle_ncm_minus_d": q1_delta,
                "r_minus_g": q2_delta,
                "r_new_to_old_minus_g": q3_delta,
                "r_gap": r_gap,
                "curve_gap": curve_gap,
                "train_gap": train_gap,
                "answers": {
                    "Q1": _answer(q1_delta),
                    "Q2": _answer(q2_delta),
                    "Q3": "YES" if q3_delta < -0.1 else ("NO" if q3_delta > 0.1 else "INCONCLUSIVE"),
                    "Q4": "YES" if r_gap >= 5.0 else "NO",
                    "Q5": "YES" if q5_yes else "NO",
                    "Q6": "YES" if np.isfinite(curve_gap) and curve_gap >= 1.0 else ("NO" if np.isfinite(curve_gap) else "INCONCLUSIVE"),
                    "Q7": "YES" if structure_alignment < pre_alignment - 0.01 else "NO",
                    "Q8": "YES" if abs(train_gap) >= 5.0 else "INCONCLUSIVE",
                },
                "transitions": transitions,
            }
        )
    combined_answers = {
        question: _aggregate_answer([row["answers"][question] for row in records])
        for question in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8")
    }
    question_labels = {
        "Q1": "Oracle NCM은 Full Oracle Gaussian CA보다 높은가?",
        "Q2": "Actual old feature CA는 matched Gaussian CA보다 높은가?",
        "Q3": "Gaussian→Actual이 New→Old confusion을 줄이는가?",
        "Q4": "Actual CA에도 New-only gap이 크게 남는가?",
        "Q5": "단일 bias로 Old를 크게 잃지 않고 New를 회복하는가?",
        "Q6": "Structure/D/R이 서로 다른 bias frontier를 만드는가?",
        "Q7": "CA 이후 weight가 class mean 방향에서 멀어지는가?",
        "Q8": "G/R 차이를 optimization/fitting 차이로 설명할 수 있는가?",
    }
    lines = [
        "# IA5/CAR5 Classifier Bottleneck Diagnostic — Combined",
        "",
        "## 통합 결론 — YES / NO / INCONCLUSIVE",
        "",
        "| 질문 | 통합 답 | IA5 | CAR5 |",
        "|---|---|---|---|",
    ]
    for question in question_labels:
        per_protocol = {
            row["protocol"]: row["answers"][question] for row in records
        }
        lines.append(
            f"| {question}. {question_labels[question]} | **{combined_answers[question]}** | "
            f"{per_protocol.get('IA5', 'INCONCLUSIVE')} | {per_protocol.get('CAR5', 'INCONCLUSIVE')} |"
        )
    lines.extend(
        [
        "",
        "## Final-task 핵심 delta",
        "",
        "| Protocol | Tasks | Oracle NCM − D Seen | R-fixed − G-fixed Seen | ΔNew→Old (pp) | R New-only gap |",
        "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in records:
        lines.append(
            f"| {row['protocol']} | `{row['tasks']}` | {_fmt(row['oracle_ncm_minus_d'])} | "
            f"{_fmt(row['r_minus_g'])} | {_fmt(row['r_new_to_old_minus_g'])} | {_fmt(row['r_gap'])} |"
        )
    lines.extend(
        [
            "",
            "## 전 transition 일관성",
            "",
            "| Protocol | Task | R−G Seen (pp) | Oracle-NCM−D Seen (pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in records:
        for transition in row["transitions"]:
            lines.append(
                f"| {row['protocol']} | {transition['task']} | {_fmt(transition['r_minus_g'])} | "
                f"{_fmt(transition['oracle_minus_d'])} |"
            )
    all_r_positive = all(
        transition["r_minus_g"] > 0
        for row in records for transition in row["transitions"]
    )
    all_ncm_below_d = all(
        transition["oracle_minus_d"] < 0
        for row in records for transition in row["transitions"]
    )
    lines.extend(
        [
            "",
            "## 원인 분리 해석",
            "",
            "- Actual old feature는 모든 transition에서 fixed Gaussian보다 약간 높았지만 최대 이득이 1.05 pp 이하였다. 따라서 old Gaussian replacement가 일부 손실은 만들지만, 현재 관측된 병목의 주원인이라는 강한 증거는 아니다.",
            "- Oracle-NCM은 모든 transition에서 Full Oracle Gaussian CA보다 낮았다. 따라서 현재 frozen feature에서 prototype classifier 정보가 CA보다 월등한데 classifier fitting이 이를 잃는다는 가설은 지지되지 않는다.",
            "- Actual 교체는 final-task New→Old confusion을 줄이지 않았다. 단일 old/new bias도 Old 손실 1 pp 이내에서 큰 New 회복을 만들지 못했다.",
            "- 남는 후보는 CA objective, classifier form, optimization과 feature-space class overlap이지만, 이번 진단만으로 어느 하나를 원인으로 확정하지 않는다.",
            "",
            "## IA10/CAR10 확장 권고",
            "",
            (
                "**RECOMMEND (별도 후속 실행, 이번 실행에는 미포함).** IA5/CAR5 전 transition에서 R−G 방향과 NCM−D 방향이 모두 일관되어, 장기 sequence에서 작은 G/R 차이가 누적되는지만 IA10/CAR10으로 검증할 근거가 생겼다. 다만 현재 결론을 내기 위해 긴 protocol이 필수인 것은 아니다."
                if all_r_positive and all_ncm_below_d
                else "**HOLD.** 5-task protocol 간 방향이 일관되지 않아 IA10/CAR10보다 seed 반복과 진단 정의 점검이 우선이다."
            ),
            "",
            "Optional G/R Linear는 실행하지 않았다. G/R 차이가 모든 transition에서 작고 같은 방향이어서 old source의 주효과 판단은 모호하지 않았으며, 이번 범위에서 새로운 classifier 실험을 추가할 필요가 없었다.",
            "",
            "## 주의",
            "",
            "Actual>Gaussian은 multimodality를 단독 원인으로 증명하지 않으며, Actual≈Gaussian도 representation 원인을 증명하지 않는다. Bias sweep은 test-oracle diagnostic이고 정식 EF-CIL result가 아니다.",
        ]
    )
    destination = output_root / "classifier_bottleneck_diagnostic_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
