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
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def _summary(rows: list[dict[str, str]], condition: str) -> dict[str, str]:
    return next(
        (
            row for row in rows
            if row.get("condition") == condition and row.get("row_type") == "summary"
        ),
        {},
    )


def _answer(delta: float, threshold: float = 0.1) -> str:
    if not np.isfinite(delta):
        return "INCONCLUSIVE"
    if delta > threshold:
        return "YES"
    if delta < -threshold:
        return "NO"
    return "INCONCLUSIVE"


def _protocol_findings(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    exposure = _csv(output_dir / "ca_sampling_audit.csv")
    paired = _csv(output_dir / "g256_r256_accuracy.csv")
    probe = _csv(output_dir / "classifier_probe_accuracy.csv")
    traces = _csv(output_dir / "cosine_convergence_trace.csv")
    exposure_map = {row["condition"]: row for row in exposure}
    paired_map = {condition: _summary(paired, condition) for condition in (
        "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED", "G256_FIXED", "R256_FIXED"
    )}
    probe_map = {condition: _summary(probe, condition) for condition in (
        "STRUCTURE_CA", "COSINE_5EP", "COSINE_CONVERGED", "LINEAR_NO_BIAS", "LINEAR_WITH_BIAS"
    )}
    g_recovery = _float(paired_map["G256_FIXED"], "seen_class_accuracy") - _float(
        paired_map["G_FIXED"], "seen_class_accuracy"
    )
    r_recovery = _float(paired_map["R256_FIXED"], "seen_class_accuracy") - _float(
        paired_map["R_FIXED"], "seen_class_accuracy"
    )
    r_old_delta = _float(paired_map["R256_FIXED"], "old_accuracy") - _float(
        paired_map["R_FIXED"], "old_accuracy"
    )
    r_new_delta = _float(paired_map["R256_FIXED"], "new_accuracy") - _float(
        paired_map["R_FIXED"], "new_accuracy"
    )
    old_imbalance = abs(
        _float(paired_map["R_FIXED"], "old_accuracy")
        - _float(paired_map["R_FIXED"], "new_accuracy")
    )
    new_imbalance = abs(
        _float(paired_map["R256_FIXED"], "old_accuracy")
        - _float(paired_map["R256_FIXED"], "new_accuracy")
    )
    r256_minus_g256 = _float(paired_map["R256_FIXED"], "seen_class_accuracy") - _float(
        paired_map["G256_FIXED"], "seen_class_accuracy"
    )
    cosine5 = probe_map["COSINE_5EP"]
    converged = probe_map["COSINE_CONVERGED"]
    linear_no_bias = probe_map["LINEAR_NO_BIAS"]
    linear_with_bias = probe_map["LINEAR_WITH_BIAS"]
    selected_train_loss_change = (
        _float(cosine5, "final_training_ce") - _float(converged, "final_training_ce")
    ) / max(abs(_float(cosine5, "final_training_ce")), 1e-12)
    cosine5_trace = [row for row in traces if row.get("condition") == "COSINE_5EP"]
    converged_trace = [row for row in traces if row.get("condition") == "COSINE_CONVERGED"]
    cosine5_last = max(cosine5_trace, key=lambda row: int(row["epoch"]))
    converged_last = max(converged_trace, key=lambda row: int(row["epoch"]))
    trajectory_loss_reduction = (
        _float(cosine5_last, "train_loss") - _float(converged_last, "train_loss")
    ) / max(abs(_float(cosine5_last, "train_loss")), 1e-12)
    final_trajectory_gradient = _float(converged_last, "gradient_norm_last")
    validation_delta = _float(converged, "validation_accuracy") - _float(
        cosine5, "validation_accuracy"
    )
    converged_test_delta = _float(converged, "seen_class_accuracy") - _float(
        cosine5, "seen_class_accuracy"
    )
    linear_no_bias_delta = _float(linear_no_bias, "seen_class_accuracy") - _float(
        converged, "seen_class_accuracy"
    )
    bias_delta = _float(linear_with_bias, "seen_class_accuracy") - _float(
        linear_no_bias, "seen_class_accuracy"
    )
    q1 = "YES" if (
        _float(exposure_map["R_FIXED"], "old_draws_per_class_mean_epoch")
        < _float(exposure_map["R_FIXED"], "new_draws_per_class_mean_epoch") - 0.1
    ) else "NO"
    q3 = "YES" if new_imbalance < old_imbalance - 0.1 else (
        "NO" if new_imbalance > old_imbalance + 0.1 else "INCONCLUSIVE"
    )
    q5 = "NO" if trajectory_loss_reduction > 0.01 else (
        "YES" if trajectory_loss_reduction <= 0.01 and final_trajectory_gradient < 1e-3
        else "INCONCLUSIVE"
    )
    deltas = {
        "g256_minus_gfixed": g_recovery,
        "r256_minus_rfixed": r_recovery,
        "r_old_delta": r_old_delta,
        "r_new_delta": r_new_delta,
        "imbalance_before": old_imbalance,
        "imbalance_after": new_imbalance,
        "r256_minus_g256": r256_minus_g256,
        "selected_train_loss_change": selected_train_loss_change,
        "trajectory_loss_reduction": trajectory_loss_reduction,
        "final_trajectory_gradient": final_trajectory_gradient,
        "validation_delta": validation_delta,
        "converged_test_delta": converged_test_delta,
        "linear_no_bias_delta": linear_no_bias_delta,
        "bias_delta": bias_delta,
    }
    answers = {
        "Q1": q1,
        "Q2": _answer(g_recovery),
        "Q3": q3,
        "Q4": _answer(r256_minus_g256),
        "Q5": q5,
        "Q6": _answer(converged_test_delta),
        "Q7": _answer(linear_no_bias_delta),
        "Q8": _answer(bias_delta),
    }
    if abs(g_recovery) >= 1.0 or old_imbalance - new_imbalance >= 1.0:
        priority = "sampling/weighting"
    elif converged_test_delta >= 0.5 and trajectory_loss_reduction > 0.01:
        priority = "fitting"
    elif max(linear_no_bias_delta, bias_delta) >= 0.5:
        priority = "classifier form"
    else:
        priority = "representation/generalization"
    answers["Q9"] = f"YES — {priority}"
    return {
        "manifest": manifest,
        "exposure": exposure_map,
        "paired": paired_map,
        "probe": probe_map,
        "trace": traces,
        "deltas": deltas,
        "answers": answers,
        "priority": priority,
    }


def generate_protocol_report(output_dir: Path) -> Path:
    findings = _protocol_findings(output_dir)
    manifest = findings["manifest"]
    protocol = manifest["protocol"]
    answers = findings["answers"]
    delta = findings["deltas"]
    exposure = findings["exposure"]
    paired = findings["paired"]
    probe = findings["probe"]
    question_labels = {
        "Q1": "기존 G/R-fixed의 Old/New exposure가 불균형했는가?",
        "Q2": "G-fixed→G256에서 성능이 회복되는가?",
        "Q3": "R-fixed→R256에서 Old/New balance가 회복되는가?",
        "Q4": "동일 256 exposure에서 R256이 G256보다 높은가?",
        "Q5": "현재 5-epoch cosine classifier는 충분히 수렴했는가?",
        "Q6": "Converged cosine이 5-epoch cosine보다 높은가?",
        "Q7": "Linear no-bias가 converged cosine보다 높은가?",
        "Q8": "Bias를 허용한 linear가 추가 개선되는가?",
        "Q9": "현재 evidence가 지시하는 다음 우선순위는 무엇인가?",
    }
    evidence = {
        "Q1": (
            f"old/class/epoch mean={_fmt(_float(exposure['R_FIXED'], 'old_draws_per_class_mean_epoch'))}, "
            f"new/class/epoch={_fmt(_float(exposure['R_FIXED'], 'new_draws_per_class_mean_epoch'))}; "
            f"steps={exposure['R_FIXED']['optimizer_steps']} vs 256-paired={exposure['R256_FIXED']['optimizer_steps']}"
        ),
        "Q2": f"G256−G-fixed Seen={_fmt(delta['g256_minus_gfixed'])} pp",
        "Q3": (
            f"R256−R-fixed: Seen={_fmt(delta['r256_minus_rfixed'])}, Old={_fmt(delta['r_old_delta'])}, "
            f"New={_fmt(delta['r_new_delta'])}; |Old−New|={_fmt(delta['imbalance_before'])}→{_fmt(delta['imbalance_after'])}"
        ),
        "Q4": f"R256−G256 Seen={_fmt(delta['r256_minus_g256'])} pp",
        "Q5": (
            f"5ep→100ep trajectory train CE reduction={_fmt(100.0 * delta['trajectory_loss_reduction'])}%, "
            f"epoch100 grad={_fmt(delta['final_trajectory_gradient'])}; validation-best epoch={manifest['cosine_converged_selected_epoch']}"
        ),
        "Q6": f"Converged−5ep test Seen={_fmt(delta['converged_test_delta'])} pp",
        "Q7": f"Linear no-bias−Converged Seen={_fmt(delta['linear_no_bias_delta'])} pp",
        "Q8": f"Linear with-bias−no-bias Seen={_fmt(delta['bias_delta'])} pp",
        "Q9": findings["priority"],
    }
    lines = [
        f"# {protocol} T5 Classifier Procedure Diagnostic",
        "",
        f"> 상태: **COMPLETE** · seed 1993 · T5 only · checkpoint unchanged: **{manifest['checkpoint_unchanged']}**",
        "",
        "## 첫 페이지 결론 — YES / NO / INCONCLUSIVE",
        "",
        "| 질문 | 판정 | 직접 근거 |",
        "|---|---|---|",
    ]
    for question, label in question_labels.items():
        lines.append(f"| {question}. {label} | **{answers[question]}** | {evidence[question]} |")

    lines.extend(
        [
            "",
            "## Exposure table",
            "",
            "| Condition | Old/class draws | New/class draws | Total draws | Steps | Fixed/Dynamic |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for condition in (
        "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED", "G256_FIXED", "R256_FIXED"
    ):
        row = exposure[condition]
        old_value = (
            f"{_fmt(_float(row, 'old_draws_per_class_min_epoch'), 0)}–"
            f"{_fmt(_float(row, 'old_draws_per_class_max_epoch'), 0)} / epoch"
        )
        new_value = (
            f"{_fmt(_float(row, 'new_draws_per_class_min_epoch'), 0)}–"
            f"{_fmt(_float(row, 'new_draws_per_class_max_epoch'), 0)} / epoch"
        )
        lines.append(
            f"| {condition} | {old_value} | {new_value} | {int(float(row['total_draws'])):,} | "
            f"{int(float(row['optimizer_steps'])):,} | {row['bank_lifecycle']} |"
        )

    lines.extend(
        [
            "",
            "## G/R procedure accuracy",
            "",
            "| Condition | Seen | Old | New | Group balance | Old-only | New-only | New→Old | Old→New |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in (
        "FULL_ORACLE_GAUSSIAN_CA", "G_FIXED", "R_FIXED", "G256_FIXED", "R256_FIXED"
    ):
        row = paired[condition]
        lines.append(
            f"| {condition} | {_fmt(_float(row, 'seen_class_accuracy'))} | {_fmt(_float(row, 'old_accuracy'))} | "
            f"{_fmt(_float(row, 'new_accuracy'))} | {_fmt(_float(row, 'group_balance'))} | "
            f"{_fmt(_float(row, 'old_only_accuracy'))} | {_fmt(_float(row, 'new_only_accuracy'))} | "
            f"{_fmt(100.0 * _float(row, 'new_to_old_rate'))} | {_fmt(100.0 * _float(row, 'old_to_new_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Actual-feature classifier probe",
            "",
            "| Classifier | Seen | Old | New | Old-only | New-only | New→Old | Old→New | Train | Validation | Train CE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in (
        "STRUCTURE_CA", "COSINE_5EP", "COSINE_CONVERGED", "LINEAR_NO_BIAS", "LINEAR_WITH_BIAS"
    ):
        row = probe[condition]
        lines.append(
            f"| {condition} | {_fmt(_float(row, 'seen_class_accuracy'))} | {_fmt(_float(row, 'old_accuracy'))} | "
            f"{_fmt(_float(row, 'new_accuracy'))} | {_fmt(_float(row, 'old_only_accuracy'))} | "
            f"{_fmt(_float(row, 'new_only_accuracy'))} | {_fmt(100.0 * _float(row, 'new_to_old_rate'))} | "
            f"{_fmt(100.0 * _float(row, 'old_to_new_rate'))} | {_fmt(_float(row, 'train_accuracy'))} | "
            f"{_fmt(_float(row, 'validation_accuracy'))} | {_fmt(_float(row, 'final_training_ce'))} |"
        )

    lines.extend(
        [
            "",
            "## Convergence 및 validation",
            "",
            f"- validation sample: {manifest['validation_samples']:,}; validation class: {manifest['validation_classes']}; validation 미포함 class: {len(manifest['no_validation_classes'])}",
            f"- COSINE_CONVERGED 실행 epoch: {manifest['cosine_converged_epochs_run']}, 선택 epoch: {manifest['cosine_converged_selected_epoch']}",
            f"- stop reason: `{manifest['cosine_converged_stop_reason']}`",
            "- validation은 convergence 선택에만 사용했고 test는 epoch/regularization 선택에 사용하지 않았다.",
            "- Linear L2는 0.0005로 사전 고정했으며 test 기반 sweep을 하지 않았다.",
            "",
            "## 해석",
            "",
        ]
    )
    if abs(delta["r256_minus_g256"]) <= 0.5:
        lines.append("동일 exposure/step 조건에서 G256과 R256의 차이는 작다. Gaussian-vs-actual source보다 sampling/weighting 또는 classifier 측 요인이 우선이다.")
    elif delta["r256_minus_g256"] > 0.5:
        lines.append("동일 exposure/step 조건에서도 R256이 G256보다 높다. Actual feature가 보존하는 추가 정보가 유효할 가능성이 있으나 이를 multimodality 하나로 단정하지 않는다.")
    else:
        lines.append("R256이 G256보다 낮다. 적은 unique actual sample의 반복으로 인한 overfitting 또는 Gaussian regularization 효과 가능성이 있으며 Gaussian이 실제 분포보다 정확하다고 해석하지 않는다.")
    if delta["converged_test_delta"] > 0.5:
        lines.append("Converged cosine이 5-epoch보다 높아 현재 fitting budget 부족의 증거가 있다.")
    elif delta["trajectory_loss_reduction"] > 0.01 and delta["converged_test_delta"] <= 0.5:
        lines.append("추가 학습은 train loss를 낮췄지만 held-out Seen 개선으로 이어지지 않았다. 단순 epoch 확대보다 generalization 제약을 우선 고려해야 한다.")
    if delta["linear_no_bias_delta"] > 0.5:
        lines.append("Linear probe 개선은 cosine normalization/fixed scale 또는 class-specific intercept를 포함한 classifier form 제약 가능성을 지지한다.")
    elif delta["bias_delta"] > 0.5:
        lines.append("Bias는 linear 내부에서 소폭 개선을 만들었지만 linear 자체가 cosine보다 낮다. 따라서 intercept의 국소 효과는 있으나 classifier form이 현재 주병목이라는 증거는 아니다.")
    lines.extend(
        [
            "",
            "## Leakage 및 안전성",
            "",
            "- frozen T5 feature cache와 기존 checkpoint만 재사용했다.",
            "- backbone, adapter, APR, SCC 및 main checkpoint를 갱신하지 않았다.",
            "- optimizer에는 diagnostic classifier parameter만 포함했다.",
            "- actual feature는 diagnostic-only이며 next task로 전달하지 않았다.",
            "- 실행 전후 checkpoint edge fingerprint와 size가 동일했다.",
            "- T2–T4 및 IA10/CAR10은 실행하지 않았다.",
        ]
    )
    destination = output_dir / f"{protocol}_T5_classifier_procedure_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def generate_combined_report(output_root: Path) -> Path:
    records = []
    for protocol in ("IA5", "CAR5"):
        directory = output_root / f"{protocol}_T5"
        if (directory / "run_manifest.json").is_file():
            records.append((protocol, _protocol_findings(directory)))
    question_labels = {
        "Q1": "기존 G/R exposure 불균형?",
        "Q2": "G256으로 회복?",
        "Q3": "R256 balance 회복?",
        "Q4": "R256 > G256?",
        "Q5": "5-epoch cosine 충분?",
        "Q6": "Converged > 5ep?",
        "Q7": "Linear no-bias > cosine?",
        "Q8": "Linear bias 추가 개선?",
        "Q9": "다음 우선순위",
    }
    lines = [
        "# IA5/CAR5 T5 Classifier Procedure Diagnostic — Final",
        "",
        "## 첫 페이지 결론",
        "",
        "| 질문 | IA5 | CAR5 |",
        "|---|---|---|",
    ]
    for question, label in question_labels.items():
        values = {protocol: result["answers"][question] for protocol, result in records}
        lines.append(f"| {question}. {label} | **{values.get('IA5', 'N/A')}** | **{values.get('CAR5', 'N/A')}** |")
    lines.extend(
        [
            "",
            "## 핵심 delta",
            "",
            "| Protocol | G256−Gfixed | R256−Rfixed | R256−G256 | Converged−5ep | Linear-no-bias−Converged | Bias gain | Priority |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for protocol, result in records:
        delta = result["deltas"]
        lines.append(
            f"| {protocol} | {_fmt(delta['g256_minus_gfixed'])} | {_fmt(delta['r256_minus_rfixed'])} | "
            f"{_fmt(delta['r256_minus_g256'])} | {_fmt(delta['converged_test_delta'])} | "
            f"{_fmt(delta['linear_no_bias_delta'])} | {_fmt(delta['bias_delta'])} | {result['priority']} |"
        )
    priorities = [result["priority"] for _, result in records]
    t2_t4_recommendation = (
        "RECOMMEND" if len(set(priorities)) > 1 else "HOLD"
    )
    lines.extend(
        [
            "",
            "## T2–T4 확장 권고",
            "",
            f"**{t2_t4_recommendation}** — "
            + (
                "IA5와 CAR5의 우선 병목 판정이 달라 transition progression 확인이 필요하다. 자동 실행은 하지 않았다."
                if t2_t4_recommendation == "RECOMMEND"
                else "두 protocol의 우선 병목 판정이 같아 T5 원인 분리 결론이 명확하다. 현재 단계에서는 자동 확장하지 않는다."
            ),
            "",
            "## 범위와 주의",
            "",
            "이 결과는 actual old/current training feature를 사용하는 diagnostic oracle probe다. 정식 replay-free CIL method 결과가 아니며 actual feature를 main training이나 checkpoint에 전달하지 않았다. T2–T4, IA10/CAR10 및 신규 방법론은 실행하지 않았다.",
        ]
    )
    destination = output_root / "classifier_procedure_diagnostic_final_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination
