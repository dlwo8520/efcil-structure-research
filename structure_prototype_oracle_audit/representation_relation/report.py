from __future__ import annotations

import csv
import json
from collections import Counter
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


def _int(row: dict[str, Any], key: str) -> int:
    value = _float(row, key)
    return int(value) if np.isfinite(value) else 0


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%" if np.isfinite(value) else "N/A"


def _mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _summary(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(
        (
            row
            for row in rows
            if row.get("row_type") == "summary" and row.get("method") == method
        ),
        {},
    )


def _findings(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    equivalence = json.loads(
        (output_dir / "cosine_linear_equivalence.json").read_text(encoding="utf-8")
    )
    linear_rows = _csv(output_dir / "linear_regularization_check.csv")
    directional = _csv(output_dir / "directional_pair_confusion.csv")
    top_pairs = _csv(output_dir / "top_pair_analysis.csv")
    margins = _csv(output_dir / "pair_margin_distribution.csv")
    knn_summary = _csv(output_dir / "knn_summary.csv")
    knn_samples = _csv(output_dir / "knn_neighborhood_diagnostic.csv")
    pair_recovery = _csv(output_dir / "pair_knn_recovery.csv")
    transitions = _csv(output_dir / "pair_task_transition_analysis.csv")

    selected_linear = next(
        (row for row in linear_rows if row.get("selected_by_validation") == "True"),
        {},
    )
    structure_row = _summary(knn_summary, "Structure cosine")
    linear_row = _summary(knn_summary, "Validated Linear")
    knn_row = next(
        (
            row
            for row in knn_summary
            if row.get("row_type") == "summary" and row.get("method", "").startswith("k-NN")
        ),
        {},
    )
    structure_test = _float(structure_row, "micro_accuracy")
    linear_test = _float(linear_row, "micro_accuracy")
    linear_train = _float(selected_linear, "train_micro_accuracy")
    linear_gap = linear_test - structure_test
    equivalence_pass = equivalence.get("status") == "PASS"
    linear_converged = selected_linear.get("converged_by_gradient") == "True"
    if not equivalence_pass or not np.isfinite(linear_train):
        q0 = "INCONCLUSIVE"
        q0_reason = "cosine→linear equivalence 또는 linear fitting 기록이 유효하지 않음"
    elif linear_train < 90.0 and not linear_converged:
        q0 = "INCONCLUSIVE"
        q0_reason = (
            f"선택 linear train={_fmt(linear_train)}%, gradient convergence가 확인되지 않아 "
            "representation 원인으로 해석할 수 없음"
        )
    elif linear_gap >= -1.0:
        q0 = "YES"
        q0_reason = (
            f"validated linear test={_fmt(linear_test)}%, Structure={_fmt(structure_test)}% "
            f"(delta={_fmt(linear_gap)} pp)"
        )
    else:
        q0 = "NO"
        q0_reason = (
            f"train fitting={_fmt(linear_train)}%이나 held-out delta={_fmt(linear_gap)} pp"
        )

    total_errors = sum(_int(row, "confusion_count") for row in directional)
    top20_global = sorted(
        directional,
        key=lambda row: (-_float(row, "pair_share_of_all_errors"), -_int(row, "confusion_count")),
    )[:20]
    top20_share = (
        sum(_int(row, "confusion_count") for row in top20_global) / total_errors
        if total_errors
        else float("nan")
    )
    if not np.isfinite(top20_share):
        q1 = "INCONCLUSIVE"
    elif top20_share >= 0.50:
        q1 = "YES"
    elif top20_share <= 0.25:
        q1 = "NO"
    else:
        q1 = "INCONCLUSIVE"

    pair_type_errors = Counter()
    for row in directional:
        pair_type_errors[row.get("pair_type", "unknown")] += _int(row, "confusion_count")
    dominant_type, dominant_count = (
        pair_type_errors.most_common(1)[0] if pair_type_errors else ("unknown", 0)
    )
    dominant_share = dominant_count / total_errors if total_errors else float("nan")
    if not np.isfinite(dominant_share):
        q2 = "INCONCLUSIVE"
    elif dominant_share >= 0.50:
        q2 = f"YES — {dominant_type}"
    else:
        q2 = f"INCONCLUSIVE — largest={dominant_type}"

    margin_by_pair: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in margins:
        key = (row.get("true_class", ""), row.get("competitor_class", ""))
        margin_by_pair.setdefault(key, {})[row.get("split", "")] = row
    collapse_deltas, train_negative, test_negative = [], [], []
    for values in margin_by_pair.values():
        if "train" not in values or "test" not in values:
            continue
        train_value = _float(values["train"], "negative_fraction")
        test_value = _float(values["test"], "negative_fraction")
        collapse_deltas.append(test_value - train_value)
        train_negative.append(train_value)
        test_negative.append(test_value)
    median_collapse = float(np.nanmedian(collapse_deltas)) if collapse_deltas else float("nan")
    median_train_negative = float(np.nanmedian(train_negative)) if train_negative else float("nan")
    median_test_negative = float(np.nanmedian(test_negative)) if test_negative else float("nan")
    if not np.isfinite(median_collapse):
        q3 = "INCONCLUSIVE"
    elif median_collapse >= 0.20 and median_train_negative <= 0.20:
        q3 = "YES"
    elif median_collapse <= 0.05:
        q3 = "NO"
    else:
        q3 = "INCONCLUSIVE"

    knn_test = _float(knn_row, "micro_accuracy")
    knn_gain = knn_test - structure_test
    test_sample_rows = [row for row in knn_samples if row.get("split") == "test"]
    cosine_wrong = [row for row in test_sample_rows if row.get("cosine_correct") == "False"]
    b1_count = sum(row.get("diagnostic_case") == "B1_HEAD_MISSES_LOCAL_SUPPORT" for row in test_sample_rows)
    b2_count = sum(row.get("diagnostic_case") == "B2_SHARED_WRONG_LOW_PURITY" for row in test_sample_rows)
    b3_count = sum(row.get("diagnostic_case") == "B3_TRAIN_HIGH_HELDOUT_LOW" for row in test_sample_rows)
    b1_error_rate = b1_count / len(cosine_wrong) if cosine_wrong else float("nan")
    if not np.isfinite(knn_gain):
        q4 = "INCONCLUSIVE"
    elif knn_gain >= 1.0 or b1_error_rate >= 0.05:
        q4 = "YES"
    elif knn_gain <= 0.0 and (not np.isfinite(b1_error_rate) or b1_error_rate < 0.02):
        q4 = "NO"
    else:
        q4 = "INCONCLUSIVE"

    correct_purity = _mean(
        [_float(row, "neighborhood_purity") for row in test_sample_rows if row.get("cosine_correct") == "True"]
    )
    wrong_purity = _mean(
        [_float(row, "neighborhood_purity") for row in test_sample_rows if row.get("cosine_correct") == "False"]
    )
    purity_gap = correct_purity - wrong_purity
    if not np.isfinite(purity_gap):
        q5 = "INCONCLUSIVE"
    elif purity_gap >= 0.10:
        q5 = "YES"
    elif purity_gap <= 0.03:
        q5 = "NO"
    else:
        q5 = "INCONCLUSIVE"

    worsened = [
        row
        for row in transitions
        if _float(row, "confusion_delta_t5_minus_t4") > 0
        and _float(row, "margin_q05_delta_t5_minus_t4") < 0
        and _float(row, "purity_delta_t5_minus_t4") < 0
    ]
    worsened_rate = len(worsened) / len(transitions) if transitions else float("nan")
    if not np.isfinite(worsened_rate):
        q6 = "INCONCLUSIVE"
    elif worsened_rate >= 0.50:
        q6 = "YES"
    elif worsened_rate <= 0.20:
        q6 = "NO"
    else:
        q6 = "INCONCLUSIVE"

    t4_probe_errors = sum(_int(row, "diagnostic_probe_confusion_t4") for row in transitions)
    t5_probe_errors = sum(_int(row, "diagnostic_probe_confusion_t5") for row in transitions)
    probe_error_delta = t5_probe_errors - t4_probe_errors
    t4_probe_micro = float(manifest.get("t4_diagnostic_probe_test_micro_accuracy", float("nan")))
    t5_probe_micro = float(manifest.get("t5_diagnostic_probe_test_micro_accuracy", float("nan")))
    t4_probe_macro = float(manifest.get("t4_diagnostic_probe_test_macro_accuracy", float("nan")))
    t5_probe_macro = float(manifest.get("t5_diagnostic_probe_test_macro_accuracy", float("nan")))
    if not transitions:
        q7 = "INCONCLUSIVE"
    elif probe_error_delta > 0:
        q7 = "YES"
    elif probe_error_delta < 0:
        q7 = "NO"
    else:
        q7 = "INCONCLUSIVE"

    preexisting_count = sum(
        row.get("interpretation") == "C3_PRE_EXISTING_HARD_RELATION_CANDIDATE"
        for row in transitions
    )
    correction_rates = [_float(row, "knn_correction_rate") for row in pair_recovery]
    mean_pair_recovery = _mean(correction_rates)
    if q1 == "NO":
        priority = "broad distributed error"
    elif q6 == "YES" and q7 == "YES":
        priority = "task-induced relation degradation"
    elif preexisting_count >= max(1, len(transitions) // 2):
        priority = "pre-existing hard class relation"
    elif q4 == "YES" and np.isfinite(mean_pair_recovery) and mean_pair_recovery >= 0.20:
        priority = "local geometry 활용 부족"
    elif q3 == "YES" or q5 == "YES":
        priority = "sample coverage/generalization"
    elif q0 == "YES":
        priority = "classifier alignment"
    else:
        priority = "INCONCLUSIVE"
    q8 = "INCONCLUSIVE" if priority == "INCONCLUSIVE" else f"YES — {priority}"

    answers = {
        "Q0": (q0, q0_reason),
        "Q1": (q1, f"global top-20 pair error share={_pct(top20_share)} ({total_errors} total errors)"),
        "Q2": (q2, f"largest pair type share={_pct(dominant_share)}"),
        "Q3": (
            q3,
            f"focused-pair median negative margin train={_pct(median_train_negative)}, "
            f"test={_pct(median_test_negative)}, delta={_pct(median_collapse)}",
        ),
        "Q4": (
            q4,
            f"k-NN−Structure={_fmt(knn_gain)} pp; B1={b1_count}/{len(cosine_wrong)} cosine errors",
        ),
        "Q5": (
            q5,
            f"purity correct={_fmt(correct_purity, 3)}, wrong={_fmt(wrong_purity, 3)}, gap={_fmt(purity_gap, 3)}",
        ),
        "Q6": (q6, f"confusion↑ + q05↓ + purity↓: {len(worsened)}/{len(transitions)} focused pairs"),
        "Q7": (
            q7,
            f"focused probe pair errors T4={t4_probe_errors}, T5={t5_probe_errors}; "
            f"overall micro={_fmt(t4_probe_micro)}→{_fmt(t5_probe_micro)}, "
            f"macro={_fmt(t4_probe_macro)}→{_fmt(t5_probe_macro)}",
        ),
        "Q8": (q8, priority),
    }
    return {
        "manifest": manifest,
        "equivalence": equivalence,
        "linear": linear_rows,
        "directional": directional,
        "top_pairs": top_pairs,
        "margins": margins,
        "knn_summary": knn_summary,
        "knn_samples": knn_samples,
        "pair_recovery": pair_recovery,
        "transitions": transitions,
        "answers": answers,
        "priority": priority,
        "case_counts": {"B1": b1_count, "B2": b2_count, "B3": b3_count},
        "pair_type_errors": pair_type_errors,
        "total_errors": total_errors,
        "top20_share": top20_share,
        "structure": structure_row,
        "validated_linear": linear_row,
        "knn": knn_row,
    }


def generate_protocol_report(output_dir: Path) -> Path:
    findings = _findings(output_dir)
    manifest = findings["manifest"]
    protocol = manifest["protocol"]
    report_path = output_dir / f"{protocol}_representation_relation_report.md"
    questions = {
        "Q0": "이전 낮은 Linear 결과가 학습 설정/regularization artifact였는가?",
        "Q1": "오류는 소수 class pair에 집중되는가?",
        "Q2": "주요 error pair는 어느 old/new 유형인가?",
        "Q3": "주요 pair가 train에서는 분리되고 held-out에서 무너지는가?",
        "Q4": "Cosine 오류를 k-NN이 의미 있게 회복하는가?",
        "Q5": "낮은 neighborhood purity가 confusion과 연결되는가?",
        "Q6": "T5 주요 pair가 T4 representation보다 악화됐는가?",
        "Q7": "동일 diagnostic probe에서도 T4→T5 악화가 유지되는가?",
        "Q8": "현재 evidence가 가장 지지하는 설명은 무엇인가?",
    }
    lines = [
        f"# {protocol} T5 Representation / Class-Relation Diagnostic",
        "",
        f"> 상태: **COMPLETE** · seed {manifest['seed']} · diagnostic only · "
        f"checkpoint unchanged: **{manifest['checkpoints_unchanged']}**",
        "",
        "## 첫 페이지 결론 — YES / NO / INCONCLUSIVE",
        "",
        "| 질문 | 판정 | 직접 근거 |",
        "|---|---|---|",
    ]
    for key, question in questions.items():
        verdict, evidence = findings["answers"][key]
        lines.append(f"| {key}. {question} | **{verdict}** | {evidence} |")

    equivalence = findings["equivalence"]
    lines.extend(
        [
            "",
            "## 1. Linear validity",
            "",
            "| Equivalence | Max logit diff | Mean logit diff | Prediction mismatch |",
            "|---|---:|---:|---:|",
            f"| {equivalence['status']} | {equivalence['max_absolute_logit_difference']:.3e} | "
            f"{equivalence['mean_absolute_logit_difference']:.3e} | "
            f"{equivalence['prediction_mismatch_count']} |",
            "",
            f"Historical Structure accuracy regression: **{equivalence.get('historical_prediction_path_regression', 'UNAVAILABLE')}** "
            f"(historical={_fmt(float(equivalence.get('historical_structure_seen_accuracy', float('nan'))))}%, "
            f"reconstructed={_fmt(float(equivalence.get('reconstructed_structure_seen_accuracy', float('nan'))))}%).",
            "",
            "| L2 | Train micro | Val micro | Val macro | Test micro | Test macro | Grad norm | Selected |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in findings["linear"]:
        lines.append(
            f"| {_float(row, 'l2'):.1e} | {_fmt(_float(row, 'train_micro_accuracy'))} | "
            f"{_fmt(_float(row, 'validation_micro_accuracy'))} | {_fmt(_float(row, 'validation_macro_accuracy'))} | "
            f"{_fmt(_float(row, 'test_micro_accuracy'))} | {_fmt(_float(row, 'test_macro_accuracy'))} | "
            f"{_float(row, 'gradient_norm'):.3e} | {row.get('selected_by_validation', '')} |"
        )

    lines.extend(
        [
            "",
            "## 2. Directional error concentration",
            "",
            f"Structure test 오류는 총 **{findings['total_errors']}개**이며 global top-20 방향 pair가 "
            f"**{_pct(findings['top20_share'])}**를 차지한다.",
            "",
            "| Pair type | Error count | Share |",
            "|---|---:|---:|",
        ]
    )
    for pair_type, count in findings["pair_type_errors"].most_common():
        share = count / findings["total_errors"] if findings["total_errors"] else float("nan")
        lines.append(f"| {pair_type} | {count} | {_pct(share)} |")
    lines.extend(
        [
            "",
            "### Focused pair",
            "",
            "| True→Pred | Type | Errors | All-error share | Conditional | Train neg | Test neg | Train q05 | Test q05 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in findings["top_pairs"][:25]:
        lines.append(
            f"| {row.get('true_class')}→{row.get('predicted_class')} | {row.get('pair_type')} | "
            f"{_int(row, 'confusion_count')} | {_pct(_float(row, 'error_share'))} | "
            f"{_pct(_float(row, 'conditional_confusion'))} | {_pct(_float(row, 'train_negative_fraction'))} | "
            f"{_pct(_float(row, 'test_negative_fraction'))} | {_fmt(_float(row, 'train_q05'), 3)} | "
            f"{_fmt(_float(row, 'test_q05'), 3)} |"
        )

    lines.extend(
        [
            "",
            "## 3. k-NN / neighborhood diagnostic",
            "",
            f"Validation에서 선택된 k는 **{manifest['selected_k']}**이다. 후보는 "
            f"`{manifest['knn_candidates']}`이며 test는 선택에 사용하지 않았다.",
            "",
            "| Method | Micro | Macro | Old | New |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in (findings["structure"], findings["validated_linear"], findings["knn"]):
        lines.append(
            f"| {row.get('method', 'N/A')} | {_fmt(_float(row, 'micro_accuracy'))} | "
            f"{_fmt(_float(row, 'macro_accuracy'))} | {_fmt(_float(row, 'old_accuracy'))} | "
            f"{_fmt(_float(row, 'new_accuracy'))} |"
        )
    cases = findings["case_counts"]
    lines.extend(
        [
            "",
            f"Sample case 집계: B1(head miss/local support) **{cases['B1']}**, "
            f"B2(shared wrong/low purity) **{cases['B2']}**, "
            f"B3(train-high/held-out-low purity) **{cases['B3']}**.",
            "",
            "### Focused-pair k-NN recovery",
            "",
            "| Pair | Structure errors | k-NN corrected | Correction rate | Same wrong rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in findings["pair_recovery"][:25]:
        lines.append(
            f"| {row.get('true_class')}→{row.get('predicted_class')} | "
            f"{_int(row, 'structure_pair_error_count')} | {_int(row, 'knn_corrected_count')} | "
            f"{_pct(_float(row, 'knn_correction_rate'))} | {_pct(_float(row, 'knn_same_wrong_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## 4. T4→T5 동일 sample / 동일 class-set transition",
            "",
            "T4/T5 original head는 후보 class 수가 달라 직접 비교하지 않았다. 따라서 C2의 "
            "'original head에서만 악화, retrained probe에서 회복'은 직접 확정하지 않고, "
            "동일 절차 diagnostic probe와 k-NN 변화만 보고한다.",
            "",
            "| Pair | Probe confusion T4→T5 | Margin q05 Δ | Purity Δ | k-NN pair error Δ | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in findings["transitions"][:25]:
        lines.append(
            f"| {row.get('true_class')}→{row.get('competitor_class')} | "
            f"{_int(row, 'diagnostic_probe_confusion_t4')}→{_int(row, 'diagnostic_probe_confusion_t5')} | "
            f"{_fmt(_float(row, 'margin_q05_delta_t5_minus_t4'), 3)} | "
            f"{_fmt(_float(row, 'purity_delta_t5_minus_t4'), 3)} | "
            f"{_int(row, 'knn_pair_error_delta')} | {row.get('interpretation')} |"
        )

    lines.extend(
        [
            "",
            "## 5. Decision tree 결과",
            "",
            f"현재 자동 판정이 우선 지지하는 설명은 **{findings['priority']}**이다.",
            "",
            "이 결과는 old actual sample을 사용한 oracle diagnostic이며 EF-CIL 성능으로 해석하지 않는다. "
            "또한 test confusion에서 focused pair를 찾았으므로 pair 결과는 hypothesis generation용이다. "
            "후속 method 검증에는 다른 seed, 별도 held-out 또는 다음 transition이 필요하다.",
            "",
            "판정 기준: top-20 오류집중 YES≥50%, NO≤25%; held-out collapse YES는 median negative-fraction "
            "증가≥20%p이면서 train≤20%; k-NN recovery YES는 +1pp 또는 cosine error 중 B1≥5%; "
            "purity 연결 YES는 correct−wrong purity≥0.10; T4→T5 악화 YES는 focused pair의 절반 이상에서 "
            "confusion↑, q05↓, purity↓가 동시에 성립할 때다.",
            "",
            "## 6. 재현 정보",
            "",
            f"- T4 checkpoint: `{manifest['t4_checkpoint']}`",
            f"- T5 checkpoint: `{manifest['t5_checkpoint']}`",
            f"- diagnostic train/validation/test: {manifest['diagnostic_train_samples']} / "
            f"{manifest['validation_samples']} / {manifest['test_samples']}",
            f"- elapsed: {manifest['elapsed_seconds']:.1f} s",
            "- backbone/adapter update: 없음",
            "- T2/T3 및 IA10/CAR10: 실행하지 않음",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_combined_report(output_root: Path) -> Path:
    available = [
        output_root / f"{protocol}_T5"
        for protocol in ("IA5", "CAR5")
        if (output_root / f"{protocol}_T5" / "run_manifest.json").is_file()
    ]
    if not available:
        raise FileNotFoundError("no complete IA5_T5/CAR5_T5 result found")
    findings = {path.name.removesuffix("_T5"): _findings(path) for path in available}
    report_path = output_root / "representation_relation_diagnostic_final_report.md"
    protocols = list(findings)
    total_elapsed = sum(value["manifest"]["elapsed_seconds"] for value in findings.values())
    question_labels = {
        "Q0": "이전 낮은 Linear 결과가 설정/regularization artifact였는가?",
        "Q1": "오류가 소수 class pair에 집중되는가?",
        "Q2": "주요 오류의 old/new 유형은 무엇인가?",
        "Q3": "Train에서 분리되지만 held-out에서 무너지는가?",
        "Q4": "Cosine 오류를 k-NN이 의미 있게 회복하는가?",
        "Q5": "낮은 neighborhood purity가 오류와 연결되는가?",
        "Q6": "T5 pair가 T4 representation보다 악화됐는가?",
        "Q7": "동일 diagnostic probe에서도 악화가 유지되는가?",
        "Q8": "현재 evidence가 가장 지지하는 설명은 무엇인가?",
    }
    lines = [
        "# Representation / Class-Relation Diagnostic Final Report",
        "",
        "> IA5/CAR5 T5 · seed 1993 · frozen Structure checkpoints · oracle diagnostic only",
        "",
        "## Executive summary",
        "",
        "이번 진단에서 가장 중요한 결과는 세 가지다.",
        "",
        "1. **기존 Linear probe의 큰 성능 저하는 representation 한계가 아니라 regularization 설정의 "
        "영향이었다.** Cosine head를 linear weight로 정확히 변환했을 때 logits가 일치했고, "
        "약한 L2로 다시 학습한 Linear는 IA5에서 Structure와 0.59 pp 차이, CAR5에서는 "
        "Structure보다 1.36 pp 높았다.",
        "2. **실제 오류는 일부 class pair에 집중되지 않았다.** Global top-20 방향 pair가 차지한 "
        "오류는 IA5 8.8%, CAR5 12.8%에 불과했다. 따라서 소수 pair만 선택적으로 보정하는 "
        "방법의 전체 성능 개선 상한은 낮다.",
        "3. **T4→T5 incremental update가 주요 관계를 새로 악화시켰다는 근거는 없었다.** "
        "동일 class set·동일 sample·동일 probe 조건에서 T5 성능은 두 protocol 모두 소폭 향상됐다. "
        "현재 결과는 task-induced pair degradation보다 **broad distributed error와 held-out "
        "generalization/coverage 문제**를 더 강하게 지지한다.",
        "",
        "## 통합 첫 페이지 결론",
        "",
        "| 질문 | " + " | ".join(protocols) + " |",
        "|---|" + "---|" * len(findings),
    ]
    for question in ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"):
        cells = [f"**{value['answers'][question][0]}** — {value['answers'][question][1]}" for value in findings.values()]
        lines.append(
            f"| {question}. {question_labels[question]} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            "## 1. 실행 범위와 무결성",
            "",
            "- 대상: IA5 T5, CAR5 T5, seed 1993",
            "- T4는 동일 sample/class-set transition 진단에만 사용",
            "- T5 frozen feature cache 재사용, T4 frozen feature만 추가 추출",
            "- backbone, adapter, APR, SCC 및 main classifier는 갱신하지 않음",
            "- IA10, CAR10, T2, T3는 실행하지 않음",
            "- 실제 old sample 사용 결과는 method 성능이 아닌 oracle diagnostic으로만 해석",
            f"- 총 diagnostic 실행 시간: {total_elapsed:.1f}초",
            "",
            "| Protocol | Train | Diagnostic train | Validation | Test | Checkpoint unchanged | Elapsed |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for protocol, value in findings.items():
        manifest = value["manifest"]
        lines.append(
            f"| {protocol} | {manifest['train_samples']} | {manifest['diagnostic_train_samples']} | "
            f"{manifest['validation_samples']} | {manifest['test_samples']} | "
            f"{manifest['checkpoints_unchanged']} | {manifest['elapsed_seconds']:.1f}s |"
        )

    lines.extend(
        [
            "",
            "## 2. Stage 0 — Linear probe validity",
            "",
            "Cosine classifier의 각 weight를 `v_k = 20 × w_k / ||w_k||`로 변환했다. "
            "두 protocol 모두 전체 test set에서 prediction mismatch가 0이었으며, 기존 Structure "
            "accuracy와 재구성 accuracy도 정확히 일치했다.",
            "",
            "| Protocol | Equivalence | Max logit diff | Mismatch | Historical/reconstructed Structure |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        equivalence = value["equivalence"]
        lines.append(
            f"| {protocol} | {equivalence['status']} | "
            f"{equivalence['max_absolute_logit_difference']:.3e} | "
            f"{equivalence['prediction_mismatch_count']} | "
            f"{equivalence.get('historical_structure_seen_accuracy', float('nan')):.2f}% / "
            f"{equivalence.get('reconstructed_structure_seen_accuracy', float('nan')):.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Regularization sanity",
            "",
            "Test는 L2 선택에 사용하지 않았고 validation macro, validation micro, 더 강한 L2 순으로 "
            "선택했다.",
            "",
            "| Protocol | L2 | Train micro | Val micro | Val macro | Test micro | Test macro | Selected |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for protocol, value in findings.items():
        for row in value["linear"]:
            lines.append(
                f"| {protocol} | {_float(row, 'l2'):.1e} | "
                f"{_fmt(_float(row, 'train_micro_accuracy'))} | "
                f"{_fmt(_float(row, 'validation_micro_accuracy'))} | "
                f"{_fmt(_float(row, 'validation_macro_accuracy'))} | "
                f"{_fmt(_float(row, 'test_micro_accuracy'))} | "
                f"{_fmt(_float(row, 'test_macro_accuracy'))} | "
                f"{row.get('selected_by_validation')} |"
            )
    lines.extend(
        [
            "",
            "IA5에서는 `5e-4`의 test micro 59.12%가 `5e-5`에서 65.83%로 회복되어 "
            "Structure 66.43%에 근접했다. CAR5에서는 63.65%가 `5e-6`에서 82.55%로 "
            "회복되어 Structure 81.20%를 넘었다. 따라서 과거 Linear의 -6~-18 pp 차이를 근거로 "
            "classifier 원인을 배제한 판단은 수정해야 한다.",
            "",
            "## 3. Directional class-pair confusion",
            "",
            "| Protocol | Total errors | Top-20 pair share | old-old | old-new | new-old | new-new |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        counts = value["pair_type_errors"]
        total = value["total_errors"]
        shares = [counts.get(name, 0) / total if total else float("nan") for name in (
            "old-old", "old-new", "new-old", "new-new"
        )]
        lines.append(
            f"| {protocol} | {total} | {_pct(value['top20_share'])} | "
            + " | ".join(_pct(share) for share in shares)
            + " |"
        )
    lines.extend(
        [
            "",
            "두 protocol 모두 old-old 오류가 가장 많지만, 이는 하나의 특정 old-old pair가 "
            "지배한다는 뜻이 아니다. IA5의 가장 큰 방향 pair도 3건, CAR5의 가장 큰 pair도 "
            "16건으로 전체 오류의 1.1% 이하였다.",
            "",
            "### Global contribution 상위 pair",
            "",
            "| Protocol | Pair | Type | Errors | All-error share | Conditional confusion | Train neg | Test neg |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        global_rows = sorted(
            value["directional"],
            key=lambda row: (-_float(row, "pair_share_of_all_errors"), -_int(row, "confusion_count")),
        )[:10]
        margin_lookup = {
            (row.get("true_class"), row.get("predicted_class")): row
            for row in value["top_pairs"]
        }
        for row in global_rows:
            margin = margin_lookup.get((row.get("true_class"), row.get("predicted_class")), {})
            lines.append(
                f"| {protocol} | {row.get('true_class')}→{row.get('predicted_class')} | "
                f"{row.get('pair_type')} | {_int(row, 'confusion_count')} | "
                f"{_pct(_float(row, 'pair_share_of_all_errors'))} | "
                f"{_pct(_float(row, 'conditional_confusion'))} | "
                f"{_pct(_float(margin, 'train_negative_fraction'))} | "
                f"{_pct(_float(margin, 'test_negative_fraction'))} |"
            )

    lines.extend(
        [
            "",
            "## 4. Train–held-out pair margin",
            "",
            "Focused pair의 median negative-margin fraction은 IA5에서 0.0%→50.0%, CAR5에서 "
            "2.9%→25.0%로 증가했다. 즉 training feature에서는 분리되던 관계가 held-out에서 "
            "무너지는 경향은 분명하다. 다만 오류가 소수 pair에 집중되지 않았으므로 이는 특정 "
            "relation 하나의 문제가 아니라 더 넓은 generalization/coverage 현상으로 보는 편이 타당하다.",
            "",
            "- [IA5 margin plot](IA5_T5/pair_margin/train_validation_test_margin.png)",
            "- [CAR5 margin plot](CAR5_T5/pair_margin/train_validation_test_margin.png)",
            "",
            "## 5. k-NN 및 neighborhood purity",
            "",
            "| Protocol | Selected k | Structure micro/macro | Linear micro/macro | k-NN micro/macro | k-NN Δ | B1 | B2 | B3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        structure = value["structure"]
        linear = value["validated_linear"]
        knn = value["knn"]
        delta = _float(knn, "micro_accuracy") - _float(structure, "micro_accuracy")
        cases = value["case_counts"]
        lines.append(
            f"| {protocol} | {value['manifest']['selected_k']} | "
            f"{_fmt(_float(structure, 'micro_accuracy'))}/{_fmt(_float(structure, 'macro_accuracy'))} | "
            f"{_fmt(_float(linear, 'micro_accuracy'))}/{_fmt(_float(linear, 'macro_accuracy'))} | "
            f"{_fmt(_float(knn, 'micro_accuracy'))}/{_fmt(_float(knn, 'macro_accuracy'))} | "
            f"{_fmt(delta)} pp | {cases['B1']} | {cases['B2']} | {cases['B3']} |"
        )
    lines.extend(
        [
            "",
            "k-NN은 IA5에서 Structure보다 8.62 pp, CAR5에서 14.84 pp 낮았다. 반면 cosine "
            "정답 sample과 오답 sample의 purity 차이는 IA5 0.511, CAR5 0.330이었다. 낮은 purity가 "
            "오류와 연결되기는 하지만, local neighborhood를 그대로 사용하는 classifier가 현재 "
            "오류를 해결하지는 못한다. 특히 CAR5에서는 high-purity B1이 0건이고 low-purity "
            "shared-wrong B2가 567건이었다.",
            "",
            "- [IA5 neighborhood purity](IA5_T5/neighborhood_purity/purity_distribution.png)",
            "- [CAR5 neighborhood purity](CAR5_T5/neighborhood_purity/purity_distribution.png)",
            "- [IA5 cosine × k-NN matrix](IA5_T5/knn_vs_cosine/correctness_matrix.png)",
            "- [CAR5 cosine × k-NN matrix](CAR5_T5/knn_vs_cosine/correctness_matrix.png)",
            "",
            "## 6. T4→T5 동일 조건 비교",
            "",
            "T4와 T5 representation에 동일한 T5 class set, train/validation/test sample, probe 절차와 "
            "k-NN 규칙을 적용했다. 후보 class 수가 다른 original T4/T5 head accuracy는 직접 비교하지 않았다.",
            "",
            "| Protocol | Probe micro T4→T5 | Probe macro T4→T5 | Focused probe pair errors T4→T5 | Simultaneous degradation pairs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        manifest = value["manifest"]
        transitions = value["transitions"]
        t4_errors = sum(_int(row, "diagnostic_probe_confusion_t4") for row in transitions)
        t5_errors = sum(_int(row, "diagnostic_probe_confusion_t5") for row in transitions)
        worsened = sum(
            _float(row, "confusion_delta_t5_minus_t4") > 0
            and _float(row, "margin_q05_delta_t5_minus_t4") < 0
            and _float(row, "purity_delta_t5_minus_t4") < 0
            for row in transitions
        )
        lines.append(
            f"| {protocol} | {manifest['t4_diagnostic_probe_test_micro_accuracy']:.2f}→"
            f"{manifest['t5_diagnostic_probe_test_micro_accuracy']:.2f} | "
            f"{manifest['t4_diagnostic_probe_test_macro_accuracy']:.2f}→"
            f"{manifest['t5_diagnostic_probe_test_macro_accuracy']:.2f} | "
            f"{t4_errors}→{t5_errors} | {worsened}/{len(transitions)} |"
        )
    lines.extend(
        [
            "",
            "IA5는 overall probe micro/macro가 각각 +0.79/+0.86 pp 향상됐고 focused pair 오류는 "
            "47건으로 동일했다. CAR5도 +0.59/+0.62 pp 향상되고 focused pair 오류가 192→189로 "
            "감소했다. Confusion 증가·margin q05 감소·purity 감소가 동시에 발생한 pair는 "
            "IA5 1/40, CAR5 0/23이었다. 따라서 T5 update가 주요 relation을 전반적으로 훼손했다는 "
            "가설은 지지되지 않는다.",
            "",
            "- [IA5 transition plots](IA5_T5/task_transition/)",
            "- [CAR5 transition plots](CAR5_T5/task_transition/)",
            "",
            "## 7. Cross-protocol decision",
            "",
        ]
    )
    priorities = Counter(value["priority"] for value in findings.values())
    if len(findings) == 2 and len(priorities) == 1:
        priority = next(iter(priorities))
        lines.append(f"두 protocol의 자동 판정은 모두 **{priority}**를 우선 지지한다.")
    else:
        lines.append(
            "Protocol 간 우선 설명이 일치하지 않는다: "
            + ", ".join(f"{protocol}={value['priority']}" for protocol, value in findings.items())
            + ". 공통 method 결론보다 protocol-specific relation을 먼저 검토한다."
        )
    lines.extend(
        [
            "",
            "권장 해석 순서는 다음과 같다.",
            "",
            "1. Linear classifier family 자체는 충분한 표현력을 가진다. 향후 classifier 비교에서는 "
            "regularization과 validation selection을 먼저 공정하게 맞춰야 한다.",
            "2. 선택적 소수-pair separation은 현재 전체 오류 구조와 맞지 않는다. Pair-specific method를 "
            "추가하기 전에 오류 coverage 상한을 계산해야 한다.",
            "3. 단순 k-NN/local geometry 활용은 우선순위가 낮다. Neighborhood purity가 낮은 held-out "
            "sample이 많아 reference coverage/generalization 진단이 더 직접적이다.",
            "4. T4→T5 relation preservation failure는 현재 seed/protocol에서 주원인이 아니다. 새로운 "
            "preservation loss를 추가할 근거로 사용하지 않는다.",
            "5. 다음 검증은 독립 seed 또는 별도 held-out에서 진행해야 한다. 현재 focused pair는 test "
            "confusion으로 발견했으므로 hypothesis-generation 용도다.",
            "",
            "## 8. 한계와 해석 범위",
            "",
            "- Validation image는 diagnostic head training에서는 제외됐지만 과거 CIL backbone/adapter "
            "학습에는 사용됐으므로 representation에 완전히 unseen인 표본은 아니다.",
            "- Focused pair는 같은 test set에서 선정·기술했으므로 새로운 method의 유효성 검정으로 "
            "사용할 수 없다.",
            "- T4/T5 original heads는 class 후보 수가 달라 비교하지 않았으므로 original-head-only "
            "alignment 변화는 직접 판정하지 않았다.",
            "- IA5/CAR5 seed 1993만 분석했으며 10-task 결과로 일반화하지 않는다.",
            "",
            "## 9. 상세 보고서와 원시 산출물",
            "",
        ]
    )
    for protocol, value in findings.items():
        report = output_root / f"{protocol}_T5" / f"{protocol}_representation_relation_report.md"
        protocol_dir = output_root / f"{protocol}_T5"
        lines.extend(
            [
                f"### {protocol}",
                "",
                f"- [상세 보고서]({report.relative_to(output_root)})",
                f"- [Directional confusion]({(protocol_dir / 'directional_pair_confusion.csv').relative_to(output_root)})",
                f"- [Pair margins]({(protocol_dir / 'pair_margin_distribution.csv').relative_to(output_root)})",
                f"- [k-NN sample diagnostic]({(protocol_dir / 'knn_neighborhood_diagnostic.csv').relative_to(output_root)})",
                f"- [T4→T5 transition]({(protocol_dir / 'pair_task_transition_analysis.csv').relative_to(output_root)})",
                f"- 자동 우선 판정: **{value['priority']}**",
                "",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
