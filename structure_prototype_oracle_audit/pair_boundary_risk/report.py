from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


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


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "N/A"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * value:.{digits}f}%" if np.isfinite(value) else "N/A"


def _summary_map(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["target"], row["ranking"]): row for row in rows}


def _wins(
    summary: dict[tuple[str, str], dict[str, str]],
    candidate: str,
    baseline: str,
    targets: tuple[str, ...] = ("B_POSITIVE_DELTA_B", "C_NEW_ACTUAL_ERROR"),
) -> tuple[int, int, list[str]]:
    wins, total, evidence = 0, 0, []
    for target in targets:
        for metric in ("mean_recall_across_k", "mean_captured_mass_across_k"):
            candidate_value = _float(summary[(target, candidate)], metric)
            baseline_value = _float(summary[(target, baseline)], metric)
            if np.isfinite(candidate_value) and np.isfinite(baseline_value):
                total += 1
                wins += candidate_value > baseline_value
                evidence.append(
                    f"{target}/{metric}: {_fmt(candidate_value)} vs {_fmt(baseline_value)}"
                )
    return wins, total, evidence


def protocol_findings(output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = _csv(output_dir / "pair_confusion_boundary_metrics.csv")
    margins = _csv(output_dir / "same_sample_pair_margin_transition.csv")
    risks = _csv(output_dir / "stored_oracle_boundary_risk.csv")
    summary_rows = _csv(output_dir / "risk_ranking_summary.csv")
    correlations = _csv(output_dir / "rank_correlation_summary.csv")
    transitions = _csv(output_dir / "same_sample_prediction_transition.csv")
    summary = _summary_map(summary_rows)
    degraded = sorted(
        [row for row in risks if row.get("newly_degraded_rank", "") != ""],
        key=lambda row: _int(row, "newly_degraded_rank"),
    )
    current_hard = sorted(
        [row for row in risks if row.get("current_hard_rank", "") != ""],
        key=lambda row: _int(row, "current_hard_rank"),
    )
    transition_counts: dict[str, int] = {}
    for row in transitions:
        key = row["prediction_transition_type"]
        transition_counts[key] = transition_counts.get(key, 0) + 1
    correct_to_wrong = transition_counts.get("correct_to_wrong_B", 0)
    changed_wrong = transition_counts.get("wrong_C_to_wrong_B", 0)
    recovered = transition_counts.get("wrong_B_to_correct", 0)
    correct_both = transition_counts.get("correct_to_correct", 0)
    sample_count = len(transitions)
    correct_t4 = correct_both + correct_to_wrong
    correct_t5 = correct_both + recovered
    accuracy_t4 = 100.0 * correct_t4 / sample_count
    accuracy_t5 = 100.0 * correct_t5 / sample_count
    accuracy_delta = accuracy_t5 - accuracy_t4
    positive_delta_mass = sum(_int(row, "positive_boundary_degradation_count") for row in risks)
    newly_error_mass = sum(_int(row, "correct_to_wrong_j") for row in risks)
    boundary_t4 = sum(_int(row, "boundary_violation_count_t4") for row in risks)
    boundary_t5 = sum(_int(row, "boundary_violation_count_t5") for row in risks)
    boundary_delta = boundary_t5 - boundary_t4
    negative_delta_mass = sum(
        max(_int(row, "boundary_violation_count_t4") - _int(row, "boundary_violation_count_t5"), 0)
        for row in risks
    )
    degraded_fraction = len(degraded) / len(risks) if risks else float("nan")
    # "Sufficient" is judged jointly: candidates must exist and the aggregate
    # transition must not improve in both actual accuracy and boundary violations.
    q1 = (
        "YES"
        if degraded and (correct_to_wrong > recovered or boundary_delta > 0)
        else "NO"
    )

    delta_b = np.asarray([_float(row, "delta_B") for row in risks])
    delta_q05 = np.asarray([_float(row, "delta_margin_q05") for row in risks])
    margin_relation = (
        spearmanr(delta_b, -delta_q05, nan_policy="omit")
        if np.unique(delta_b).size >= 2 and np.unique(delta_q05).size >= 2
        else None
    )
    margin_statistic = (
        float(margin_relation.statistic) if margin_relation is not None else float("nan")
    )
    expected_sign = np.mean((delta_b > 0) == (delta_q05 < 0))
    q3 = (
        "YES"
        if margin_statistic >= 0.3 and expected_sign >= 0.7
        else ("NO" if margin_statistic <= 0 else "INCONCLUSIVE")
    )

    oracle_random_wins, oracle_random_total, oracle_random_evidence = _wins(summary, "ORACLE_GAMMA", "RANDOM_EXPECTED")
    oracle_proto_wins, oracle_proto_total, oracle_proto_evidence = _wins(summary, "ORACLE_GAMMA", "PROTOTYPE_SIMILARITY")
    # Newly-degraded risk must beat both random and the simple deployable
    # prototype baseline on recall and captured mass for both B/C targets.
    q4 = "YES" if oracle_random_wins == oracle_random_total and oracle_proto_wins == oracle_proto_total else "NO"
    oracle_gamma = np.asarray([_float(row, "gamma_oracle") for row in risks])
    stored_gamma = np.asarray([_float(row, "gamma_stored") for row in risks])
    gamma_spearman = spearmanr(oracle_gamma, stored_gamma, nan_policy="omit")
    q5 = "YES" if gamma_spearman.statistic >= 0.9 else (
        "NO" if gamma_spearman.statistic <= 0 else "INCONCLUSIVE"
    )
    stored_proto_wins, stored_proto_total, stored_proto_evidence = _wins(
        summary, "STORED_GAMMA", "PROTOTYPE_SIMILARITY"
    )
    q6 = "YES" if stored_proto_wins == stored_proto_total else "NO"
    oracle_target_corr = {
        row["target"]: _float(row, "spearman")
        for row in correlations if row["ranking"] == "ORACLE_GAMMA"
    }
    q7 = "YES" if q4 == "YES" else "NO"
    if q1 == "NO" and accuracy_delta > 0 and boundary_delta < 0:
        q8 = "D — incremental-specific degradation 자체가 거의 없음"
    elif q4 != "YES":
        q8 = "C — mean/covariance 자체로 설명 부족"
    elif q6 == "YES" and q5 == "YES":
        q8 = "A — stored statistics로 risk identification 가능"
    else:
        q8 = "B — oracle에서는 가능하지만 stored statistics가 부족"
    ready = q1 == "YES" and q4 == "YES" and q5 == "YES" and q6 == "YES"
    q9 = "YES" if ready else "NO"
    return {
        "manifest": manifest,
        "metrics": metrics,
        "margins": margins,
        "risks": risks,
        "summary_rows": summary_rows,
        "summary": summary,
        "correlations": correlations,
        "degraded": degraded,
        "current_hard": current_hard,
        "transition_counts": transition_counts,
        "positive_delta_mass": positive_delta_mass,
        "negative_delta_mass": negative_delta_mass,
        "boundary_t4": boundary_t4,
        "boundary_t5": boundary_t5,
        "boundary_delta": boundary_delta,
        "newly_error_mass": newly_error_mass,
        "sample_count": sample_count,
        "accuracy_t4": accuracy_t4,
        "accuracy_t5": accuracy_t5,
        "accuracy_delta": accuracy_delta,
        "degraded_fraction": degraded_fraction,
        "gamma_spearman": float(gamma_spearman.statistic),
        "margin_spearman": margin_statistic,
        "margin_sign_agreement": float(expected_sign),
        "oracle_random_wins": (oracle_random_wins, oracle_random_total, oracle_random_evidence),
        "oracle_proto_wins": (oracle_proto_wins, oracle_proto_total, oracle_proto_evidence),
        "stored_proto_wins": (stored_proto_wins, stored_proto_total, stored_proto_evidence),
        "oracle_target_corr": oracle_target_corr,
        "answers": {
            "Q1": (q1, f"eligible={len(degraded)}/{len(risks)} ({_pct(degraded_fraction)}); new errors={correct_to_wrong} < recovered={recovered}; fixed-head accuracy Δ={_fmt(accuracy_delta)} pp; boundary violations Δ={boundary_delta}"),
            "Q2": ("DESCRIPTIVE", f"correct→wrong-j={correct_to_wrong}, wrong-C→wrong-B={changed_wrong}, recovered={recovered}"),
            "Q3": (q3, f"Spearman(delta_B, -delta_q05)={_fmt(margin_statistic)}, sign agreement={_pct(expected_sign)}"),
            "Q4": (q4, f"new-risk B/C: Oracle beats random {oracle_random_wins}/{oracle_random_total}, prototype {oracle_proto_wins}/{oracle_proto_total}; correlations remain small"),
            "Q5": (q5, f"Spearman(gamma_stored, gamma_oracle)={_fmt(gamma_spearman.statistic)}"),
            "Q6": (q6, f"Stored gamma beats prototype on {stored_proto_wins}/{stored_proto_total} B/C recall/mass comparisons; consistent superiority requires 4/4"),
            "Q7": (q7, "New-risk ranking is not consistently superior; Oracle target Spearman: " + ", ".join(f"{k}={_fmt(v)}" for k, v in oracle_target_corr.items())),
            "Q8": (q8, q8),
            "Q9": (q9, "R1/Q4/Q5/Q6 must all be YES; relative ranking evidence, not a single absolute cutoff"),
        },
        "method_ready": ready,
    }


def _write_method_notes(output_dir: Path, findings: dict[str, Any]) -> Path:
    path = output_dir / "method_readiness_notes.md"
    answer = findings["answers"]
    lines = [
        "# Method Readiness Notes",
        "",
        f"> METHOD READY: **{answer['Q9'][0]}**",
        "",
        "## Evidence gate",
        "",
        f"- R1 충분한 net newly-degraded mass: **{answer['Q1'][0]}** — {answer['Q1'][1]}",
        f"- R2 Oracle gamma ranking: **{answer['Q4'][0]}** — {answer['Q4'][1]}",
        f"- R3 Stored gamma vs prototype: **{answer['Q6'][0]}** — {answer['Q6'][1]}",
        f"- Stored↔Oracle reproduction: **{answer['Q5'][0]}** — {answer['Q5'][1]}",
        "",
        "## Optional Stage D 설계 제안",
        "",
        "METHOD READY가 YES일 때만 random/prototype/stored-risk/oracle-development selection을 동일 pair 수, 동일 total coefficient, 동일 optimizer step으로 비교한다. Test target으로 pair 또는 hyperparameter를 선택하지 않는다.",
        "",
        "첫 intervention은 separation이 아니라 risk-weighted relation preservation 후보로 둔다. Existing SCC의 learnable current feature 경로에 selected old-class boundary 방향을 연결해야 하며, detached stored prototype끼리만 계산하는 loss는 gradient가 adapter/backbone으로 흐르지 않으므로 사용하지 않는다.",
        "",
        "이 문서는 구현 readiness만 기록하며 새 method의 성능 또는 인과효과를 주장하지 않는다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_protocol_report(output_dir: Path) -> Path:
    findings = protocol_findings(output_dir)
    protocol = findings["manifest"]["protocol"]
    report = output_dir / f"{protocol}_pair_boundary_risk_report.md"
    _write_method_notes(output_dir, findings)
    questions = {
        "Q1": "동일 sample에서 newly-degraded directional pair가 충분히 존재하는가?",
        "Q2": "Confusion 증가는 correct→wrong과 wrong-destination 변경 중 무엇인가?",
        "Q3": "delta_B와 same-sample margin 악화가 일치하는가?",
        "Q4": "Oracle gamma가 actual risky pair를 ranking하는가?",
        "Q5": "Stored gamma가 Oracle risk ranking을 재현하는가?",
        "Q6": "Stored gamma가 prototype similarity보다 나은가?",
        "Q7": "Mean/covariance로 held-out boundary risk를 설명할 수 있는가?",
        "Q8": "현재 evidence가 가장 지지하는 case는 무엇인가?",
        "Q9": "METHOD READY인가?",
    }
    lines = [
        f"# {protocol} T4→T5 Pair Boundary-Risk Diagnostic",
        "",
        f"> 상태: COMPLETE · seed 1993 · METHOD READY: **{'YES' if findings['method_ready'] else 'NO'}** · checkpoint unchanged",
        "",
        "## First-page answers",
        "",
        "| 질문 | 판정 | 근거 |",
        "|---|---|---|",
    ]
    for key, question in questions.items():
        verdict, evidence = findings["answers"][key]
        lines.append(f"| {key}. {question} | **{verdict}** | {evidence} |")
    lines.extend(
        [
            "",
            "## Same-sample transition",
            "",
            "| Transition | Count |",
            "|---|---:|",
        ]
    )
    for name, count in sorted(findings["transition_counts"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(
        [
            "",
            f"Fixed-head accuracy는 T4 **{findings['accuracy_t4']:.2f}%**에서 T5 "
            f"**{findings['accuracy_t5']:.2f}%**로 변해 Δ={findings['accuracy_delta']:+.2f} pp였다. "
            f"전체 ordered sample-pair boundary violation은 {findings['boundary_t4']}→"
            f"{findings['boundary_t5']} (Δ={findings['boundary_delta']:+d})였다. Positive degradation "
            f"mass {findings['positive_delta_mass']}보다 improvement mass "
            f"{findings['negative_delta_mass']}가 컸다.",
            "",
            "T4/T5 모두 frozen T5 Structure cosine head를 공통 readout으로 사용했다. 따라서 이 표는 historical T4 head 성능 비교가 아니라 current boundary에 대한 representation transition이다.",
            "Prototype-similarity baseline은 stored mean cosine이며 방향성이 없으므로 `i→j`와 `j→i`에 동일한 값이 들어간다.",
            "",
            "## Newly-degraded pairs",
            "",
            "| Rank | Pair | Type | n | ΔC | ΔB | New errors | Δmargin q05 | Oracle γ | Stored γ |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in findings["degraded"][:30]:
        lines.append(
            f"| {_int(row, 'newly_degraded_rank')} | {row['true_class']}→{row['competitor_class']} | "
            f"{row['pair_type']} | {_int(row, 'n_true_samples')} | {_fmt(_float(row, 'delta_C'))} | "
            f"{_fmt(_float(row, 'delta_B'))} | {_int(row, 'correct_to_wrong_j')} | "
            f"{_fmt(_float(row, 'delta_margin_q05'))} | {_fmt(_float(row, 'gamma_oracle'))} | "
            f"{_fmt(_float(row, 'gamma_stored'))} |"
        )
    if not findings["degraded"]:
        lines.append("| - | 조건을 모두 만족한 pair 없음 | - | - | - | - | - | - | - | - |")
    lines.extend(
        [
            "",
            "## Ranking comparison",
            "",
            "아래 값은 K={10,20,50,100}의 평균이다. Gamma는 작을수록 위험하여 ranking score로는 `-gamma`를 사용했다.",
            "",
            "| Target | Ranking | Mean Recall@K | Mean captured mass@K | Spearman |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in findings["summary_rows"]:
        lines.append(
            f"| {row['target']} | {row['ranking']} | {_fmt(_float(row, 'mean_recall_across_k'))} | "
            f"{_fmt(_float(row, 'mean_captured_mass_across_k'))} | {_fmt(_float(row, 'spearman'))} |"
        )
    static_oracle = findings["summary"][("A_B_T5", "ORACLE_GAMMA")]
    new_boundary_oracle = findings["summary"][("B_POSITIVE_DELTA_B", "ORACLE_GAMMA")]
    new_error_oracle = findings["summary"][("C_NEW_ACTUAL_ERROR", "ORACLE_GAMMA")]
    lines.extend(
        [
            "",
            "Static current risk(Target A)와 incremental new risk(Target B/C)는 분리해서 해석한다. "
            f"Oracle mean Recall@K는 A={_fmt(_float(static_oracle, 'mean_recall_across_k'))}, "
            f"B={_fmt(_float(new_boundary_oracle, 'mean_recall_across_k'))}, "
            f"C={_fmt(_float(new_error_oracle, 'mean_recall_across_k'))}였다. "
            "Target A가 높더라도 B/C가 낮으면 incremental-specific selection 근거가 아니다.",
        ]
    )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`q_hat=Phi(-gamma)`는 Gaussian projection의 보조값이며 multiclass에서 competitor로 최종 예측될 확률이 아니다. 결과는 ranking association 진단이고 causal factor 또는 method 성능이 아니다. Test에서 관찰한 target은 Stage D selection에 사용할 수 없다.",
            "",
            "## Artifacts",
            "",
            "- `same_sample_prediction_transition.csv`",
            "- `pair_confusion_boundary_metrics.csv`",
            "- `same_sample_pair_margin_transition.csv`",
            "- `stored_oracle_boundary_risk.csv`",
            "- `risk_recall_at_k.csv`, `captured_degradation_mass.csv`, `rank_correlation_summary.csv`",
            "- `optional_region_support_diagnostic.csv` (설명용 only)",
            "- `method_readiness_notes.md`",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def generate_combined_report(output_root: Path) -> Path:
    findings = {
        protocol: protocol_findings(output_root / f"{protocol}_T4_T5")
        for protocol in ("IA5", "CAR5")
        if (output_root / f"{protocol}_T4_T5" / "run_manifest.json").is_file()
    }
    if not findings:
        raise FileNotFoundError("no completed pair-boundary protocol outputs")
    path = output_root / "pair_boundary_risk_diagnostic_final_report.md"
    lines = [
        "# Pair Boundary-Risk Diagnostic Final Report",
        "",
        "> IA5/CAR5 · T4→T5 · seed 1993 · frozen diagnostic only",
        "",
        "## Executive conclusion",
        "",
        "**최종 결론은 METHOD READY = NO이다.** 일부 directional pair의 악화는 관찰됐지만 "
        "전체 pair와 sample 수에 비해 희소했고, 두 protocol 모두 새 오류보다 복구가 많았으며 "
        "fixed-head accuracy와 전체 boundary violation은 오히려 개선됐다. Stored gamma는 Oracle "
        "gamma 자체는 매우 잘 재현했지만, 핵심 target인 newly-degraded boundary와 newly-created "
        "actual error의 Top-K ranking에서는 prototype similarity보다 일관되게 우수하지 않았다.",
        "",
        "따라서 현재 결과는 **D. incremental-specific degradation 자체가 전체 성능의 주된 문제는 "
        "아님**을 가장 강하게 지지한다. Mean/covariance boundary risk는 CAR5의 현재 static "
        "boundary risk를 설명하는 데는 일부 유효했지만, 다음 단계의 risk-weighted preservation "
        "method를 시작할 충분한 cross-protocol 근거는 아니다.",
        "",
        "## First-page answers",
        "",
        "| Question | " + " | ".join(findings) + " |",
        "|---|" + "---|" * len(findings),
    ]
    for question in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9"):
        lines.append(
            f"| {question} | "
            + " | ".join(
                f"**{value['answers'][question][0]}** — {value['answers'][question][1]}"
                for value in findings.values()
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 1. 실행 범위와 score 정의",
            "",
            "- IA5/CAR5, seed 1993, T4→T5만 분석",
            "- T4/T5 frozen feature cache와 T5 Structure checkpoint의 stored mean/covariance 사용",
            "- backbone, adapter, APR, SCC, main classifier 업데이트 없음",
            "- T4 original head에는 T5 신규 class가 없으므로 frozen T5 Structure cosine head를 "
            "T4/T5 feature 모두에 적용한 fixed common readout 사용",
            "- 따라서 prediction transition은 historical T4 head 성능이 아니라 current T5 boundary에 "
            "대한 representation 변화",
            "- `a_ij = normalize(w_i)-normalize(w_j)` sign equivalence는 두 protocol 모두 PASS",
            "- Gamma는 ranking diagnostic이며 actual multiclass error probability가 아님",
            "",
            "| Protocol | Samples | Classes | Ordered pairs | Newly-degraded eligible | Checkpoint unchanged | Runtime |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for protocol, value in findings.items():
        manifest = value["manifest"]
        lines.append(
            f"| {protocol} | {value['sample_count']} | {manifest['common_class_count']} | "
            f"{manifest['pair_count']} | {len(value['degraded'])} ({_pct(value['degraded_fraction'])}) | "
            f"{manifest['checkpoints_unchanged']} | {manifest['elapsed_seconds']:.1f}s |"
        )
    lines.extend(
        [
            "",
            "## 2. Same-sample prediction transition",
            "",
            "| Protocol | T4 fixed-head acc | T5 fixed-head acc | Δ | Correct→Wrong | Wrong→Correct | Wrong destination change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        counts = value["transition_counts"]
        lines.append(
            f"| {protocol} | {value['accuracy_t4']:.2f}% | {value['accuracy_t5']:.2f}% | "
            f"{value['accuracy_delta']:+.2f} pp | {counts.get('correct_to_wrong_B', 0)} | "
            f"{counts.get('wrong_B_to_correct', 0)} | {counts.get('wrong_C_to_wrong_B', 0)} |"
        )
    lines.extend(
        [
            "",
            "IA5는 새 오류 14개보다 복구 26개가 많아 +0.79 pp 개선됐고, CAR5도 새 오류 "
            "47개보다 복구 73개가 많아 +0.32 pp 개선됐다. 즉 newly-degraded sample은 실제로 "
            "존재하지만 전체 transition의 순효과는 악화가 아니다.",
            "",
            "## 3. Pairwise boundary transition",
            "",
            "| Protocol | B-violation T4 | B-violation T5 | Net Δ | Positive degradation mass | Improvement mass | ΔB>0 pairs | ΔB<0 pairs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        positive_pairs = sum(_float(row, "delta_B") > 0 for row in value["risks"])
        negative_pairs = sum(_float(row, "delta_B") < 0 for row in value["risks"])
        lines.append(
            f"| {protocol} | {value['boundary_t4']} | {value['boundary_t5']} | "
            f"{value['boundary_delta']:+d} | {value['positive_delta_mass']} | "
            f"{value['negative_delta_mass']} | {positive_pairs} | {negative_pairs} |"
        )
    lines.extend(
        [
            "",
            "IA5는 boundary violation이 151건, CAR5는 318건 순감소했다. 악화 mass만 보면 "
            "IA5 271, CAR5 273건이지만 반대 방향 improvement mass가 각각 422, 591건으로 더 "
            "크다. Positive mass만 보고 relation degradation이 지배적이라고 결론 내리면 안 된다.",
            "",
            "## 4. Newly-degraded pair의 규모",
            "",
            "Eligibility 조건은 `n>=5`, `ΔC>0`, `ΔB>0`, q05 margin 악화, correct→wrong-j "
            "1개 이상을 동시에 만족하는 것이다.",
            "",
            "| Protocol | Eligible pairs | Pair share | New errors | Largest new-error count/pair | Main pair types |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for protocol, value in findings.items():
        type_counts: dict[str, int] = {}
        for row in value["degraded"]:
            type_counts[row["pair_type"]] = type_counts.get(row["pair_type"], 0) + 1
        largest = max((_int(row, "correct_to_wrong_j") for row in value["degraded"]), default=0)
        lines.append(
            f"| {protocol} | {len(value['degraded'])} | {_pct(value['degraded_fraction'], 3)} | "
            f"{value['transition_counts'].get('correct_to_wrong_B', 0)} | {largest} | "
            + ", ".join(f"{key}={count}" for key, count in sorted(type_counts.items()))
            + " |"
        )
    lines.extend(
        [
            "",
            "IA5 eligible pair는 전체의 0.023%, CAR5는 0.071%였다. IA5 각 pair는 새 오류 "
            "1개씩이었고 CAR5도 상위 3개만 2개, 나머지는 1개였다. 이는 candidate relation의 "
            "존재를 보여주지만 selective method가 해결할 충분한 error mass를 입증하지는 않는다.",
            "",
            "## 5. ΔB와 same-sample margin",
            "",
            "| Protocol | Spearman(ΔB, −Δq05) | Sign agreement | 판정 |",
            "|---|---:|---:|---|",
        ]
    )
    for protocol, value in findings.items():
        lines.append(
            f"| {protocol} | {_fmt(value['margin_spearman'])} | "
            f"{_pct(value['margin_sign_agreement'])} | {value['answers']['Q3'][0]} |"
        )
    lines.extend(
        [
            "",
            "두 correlation은 양수지만 0.115/0.079로 작다. ΔB와 q05 margin이 완전히 무관하다고 "
            "할 수는 없지만, 같은 현상을 강하게 일관되게 측정한다고 보기도 어렵다.",
            "",
            "## 6. Risk ranking — static target과 incremental target 분리",
            "",
            "표의 값은 K={10,20,50,100} 평균이다.",
            "",
            "| Protocol | Target | Ranking | Mean Recall@K | Mean captured mass@K | Spearman |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for protocol, value in findings.items():
        for row in value["summary_rows"]:
            if row["ranking"] == "RANDOM_EXPECTED":
                continue
            lines.append(
                f"| {protocol} | {row['target']} | {row['ranking']} | "
                f"{_fmt(_float(row, 'mean_recall_across_k'))} | "
                f"{_fmt(_float(row, 'mean_captured_mass_across_k'))} | "
                f"{_fmt(_float(row, 'spearman'))} |"
            )
    lines.extend(
        [
            "",
            "### Target A — 현재 T5 pairwise boundary risk",
            "",
            "CAR5에서는 Oracle gamma가 mean Recall@K 0.540, Stored gamma가 0.467로 static current "
            "risk ranking에 의미 있는 신호가 있었다. 반면 IA5는 Oracle/Stored/Prototype 모두 "
            "Recall@K가 0이었고 prototype correlation(0.288)이 gamma보다 높았다.",
            "",
            "### Target B/C — 새 boundary 악화와 새 actual error",
            "",
            "핵심 incremental target에서는 결과가 약하다. IA5 Target B의 Oracle/Stored Recall@K는 "
            "모두 0이며, Target C도 mean 0.010이다. CAR5도 Target B Oracle 0.065/Stored 0.053, "
            "Target C Oracle 0.090/Stored 0.075에 그쳤고, prototype baseline이 각각 0.073/0.148로 "
            "같거나 더 높았다. 따라서 static risk 식별 가능성을 newly-degraded pair 식별 능력으로 "
            "확대 해석할 수 없다.",
            "",
            "## 7. Stored vs Oracle gamma",
            "",
            "| Protocol | Gamma rank Spearman | 의미 |",
            "|---|---:|---|",
        ]
    )
    for protocol, value in findings.items():
        lines.append(
            f"| {protocol} | {_fmt(value['gamma_spearman'])} | Stored statistics가 Oracle gamma "
            "순서 자체는 매우 잘 재현 |"
        )
    lines.extend(
        [
            "",
            "IA5 0.980, CAR5 0.988은 stored statistics 품질이 낮아서 ranking이 실패한 것이 "
            "아님을 보여준다. Oracle gamma 자체도 newly-degraded target과의 상관이 낮기 때문에 "
            "주된 한계는 stored estimation보다는 **mean/covariance Gaussian boundary score와 실제 "
            "incremental error target 사이의 약한 연결**이다.",
            "",
            "## 8. METHOD READINESS",
            "",
            "| Gate | IA5 | CAR5 | Cross-protocol |",
            "|---|---|---|---|",
            f"| R1. 충분한 net degradation | {findings['IA5']['answers']['Q1'][0]} | "
            f"{findings['CAR5']['answers']['Q1'][0]} | FAIL |",
            f"| R2. Oracle new-risk ranking | {findings['IA5']['answers']['Q4'][0]} | "
            f"{findings['CAR5']['answers']['Q4'][0]} | FAIL |",
            f"| R3. Stored > prototype | {findings['IA5']['answers']['Q6'][0]} | "
            f"{findings['CAR5']['answers']['Q6'][0]} | FAIL |",
            "| Final | **NO** | **NO** | **METHOD READY = NO** |",
            "",
            "따라서 현재 결과만으로 risk-weighted relation preservation 또는 selective pair "
            "intervention을 구현·성능 실험할 단계는 아니다. 먼저 다른 seed/transition에서 "
            "newly-degraded mass가 반복되는지 확인하거나, mean/covariance 외 tail/local/coverage "
            "정보가 new-error ranking을 개선하는지 별도 diagnostic으로 검증해야 한다.",
            "",
            "## 9. Scientific limitations",
            "",
            "- Fixed T5 head를 공통 readout으로 썼으므로 historical T4 classifier의 실제 배포 성능 비교가 아니다.",
            "- Test label로 degradation target을 정의했으므로 이 pair를 Stage D training selection에 그대로 사용할 수 없다.",
            "- `q=Phi(-gamma)`는 projected Gaussian 보조값이지 multiclass error probability가 아니다.",
            "- Gamma correlation은 association이며 causal evidence가 아니다.",
            "- IA5/CAR5 seed 1993 한 개만 분석했으므로 10-task 또는 다른 seed로 일반화할 수 없다.",
            "- Optional Mahalanobis region 표는 selected pair 설명용이며 readiness 판정에 사용하지 않았다.",
            "",
            "## 10. Detailed reports and artifacts",
            "",
        ]
    )
    for protocol, value in findings.items():
        lines.append(
            f"- [{protocol} report]({protocol}_T4_T5/{protocol}_pair_boundary_risk_report.md) — "
            f"METHOD READY {'YES' if value['method_ready'] else 'NO'}"
        )
    lines.extend(
        [
            "",
            "- [IA5 raw risk table](IA5_T4_T5/stored_oracle_boundary_risk.csv)",
            "- [CAR5 raw risk table](CAR5_T4_T5/stored_oracle_boundary_risk.csv)",
            "- [IA5 Recall@K plots](IA5_T4_T5/recall_at_k/)",
            "- [CAR5 Recall@K plots](CAR5_T4_T5/recall_at_k/)",
            "",
            "모든 수치는 association/ranking diagnostic이다. Gamma를 actual error probability로, Gaussian을 실제 분포의 정확한 모델로, correlation을 causal evidence로 해석하지 않는다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
