"""Write final Markdown only after both paired pilots pass audit assertions."""
from __future__ import annotations

from analyze_pilot import ROOT, PROTOCOLS, BRANCHES, analyze, _read_csv


def _n(value):
    return f"{value:.2f}"


def generate() -> None:
    result = analyze()
    phase_a = result["phase_a"]
    phase_b = result["phase_b"]
    lines = [
        "# Final CEOS Diagnostic + Current-Task Variation Consistency Pilot",
        "",
        "## 범위와 비교 원칙",
        "",
        "- IA10/CAR10, seed 1993. 기존 Structure T8 task-end checkpoint에서 시작해 T9→T10만 학습했다. 기존 checkpoint/result는 읽기만 했다.",
        "- 핵심 인과 비교는 **동일한 두-view supervised Control(C)** 대 **동일한 두-view + JS(J)** 이다. 기존 one-view Structure는 참고값일 뿐이다.",
        "- 원래 Structure train augmentation을 같은 current image에 독립적으로 두 번 적용했다. 두 branch는 sample/order/view SHA-256, optimizer step, stats/APR/CA 호출 수가 정확히 일치해야 보고서를 생성한다.",
        "- SCC loss는 각 view에 적용 후 평균했다. APR 통계는 task 종료 시 원래 경로에서 한 번만 계산했다. Gaussian CA sampling·optimizer·schedule·256/class exposure는 원본을 그대로 사용했다.",
        "- JS는 current-only unscaled cosine logits, temperature 1.0, lambda 0.1. 다른 teacher/selector/replay sampler는 없다.",
        "",
        "## Phase A — CEOS New 오류의 목적지",
        "",
        "| Protocol | New test | 새 오류 (New%) | Recovery (New%) | Old 경쟁(Type 1; New%) | New-New(Type 2; New%) | 둘 다(Type 3; New%) | 평균 Δg | 평균 Δw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        row = phase_a[protocol]
        types = row["type_counts"]["loss"]
        n = row["N_new_test"]
        loss = row["gaussian_correct_ceos_wrong_count"]
        recovered = row["gaussian_wrong_ceos_correct_count"]
        typed = [f"{types[key]} ({100*types[key]/n:.2f}%)" for key in
                 ("type1_old_competition", "type2_new_internal", "type3_both")]
        lines.append(f"| {protocol} | {n} | {loss} ({100*loss/n:.2f}%) | {recovered} ({100*recovered/n:.2f}%) | {typed[0]} | {typed[1]} | {typed[2]} | {_n(row['delta_g']['all_new']['mean'])} | {_n(row['delta_w']['all_new']['mean'])} |")
    lines += [
        "",
        "기존 C1 head/logit이 저장되지 않아 사용자가 승인한 **동일 C1 설정 1회 재실행**으로 복원했다. 이전 C1 accuracy와 train loss가 일치하지 않으면 이 분석이 중단되도록 했다. C0는 기존 저장된 paired logits을 사용했다.",
        "",
        "새 New 오류는 두 protocol 모두 전부 **Type 1: Old competition**이었다. New-New 단독 악화나 recovery는 관찰되지 않았다. Δg는 전반적으로 음수이고 Δw 변화는 더 작다. 따라서 CEOS의 New 손실은 이 조건에서 Old/New score competition으로 설명되며, 추가 boundary sampler 튜닝은 하지 않는다. 전체·손실·회복 집합의 Δg/Δw 분포와 sample별 로그는 `phaseA_ceos_error_destination/`에 있다.",
        "",
        "![CEOS 신규 New 오류 목적지](phaseA_ceos_error_destination/new_error_destination_stacked.png)",
        "",
        "## Phase B — T9 직후 (%)",
        "",
        "| Protocol | Branch | Seen | T1–T8 Old | T9 New | Macro | Old→New | New→Old |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for branch in BRANCHES:
            row = phase_b[protocol][branch]
            lines.append(f"| {protocol} | {branch} | {_n(row['t9_seen'])} | {_n(row['t9_prior_old'])} | {_n(row['t9_new'])} | {_n(row['t9_macro'])} | {row['t9_old_to_new']:.3f} | {row['t9_new_to_old']:.3f} |")
    lines += ["", "## Phase B — T10 직후 (%)", "",
              "| Protocol | Branch | Seen | T1–T8 Prior Old | T9 Newly Old | T10 New | Macro |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for branch in BRANCHES:
            row = phase_b[protocol][branch]
            lines.append(f"| {protocol} | {branch} | {_n(row['t10_seen'])} | {_n(row['t10_prior_old'])} | {_n(row['t10_newly_old_t9'])} | {_n(row['t10_new'])} | {_n(row['t10_macro'])} |")
    lines += ["", "## T9 cohort retention", "",
              "`Retention = T9 class accuracy after T10 − immediately after T9` (percentage points).", "",
              "| Protocol | Branch | T9 Immediate | T9 after T10 | Retention |",
              "|---|---|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for branch in BRANCHES:
            row = phase_b[protocol][branch]
            lines.append(f"| {protocol} | {branch} | {_n(row['t9_new'])} | {_n(row['t10_newly_old_t9'])} | {row['retention_t9']:+.2f} |")
    lines += ["", "## JS − Control 핵심 delta (pp)", "",
              "| Protocol | T9 New | T9 after T10 | ΔRetention | T1–T8 after T10 | T10 New | T10 Seen |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        row = phase_b[protocol]["delta"]
        lines.append(f"| {protocol} | {row['t9_new']:+.2f} | {row['t10_newly_old_t9']:+.2f} | {row['retention_t9']:+.2f} | {row['t10_prior_old']:+.2f} | {row['t10_new']:+.2f} | {row['t10_seen']:+.2f} |")
    lines += ["", "### 기존 one-view Structure (참고 전용)", "",
              "| Protocol | T9 cohort after T9 | T9 cohort after T10 | T10 Seen |",
              "|---|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        original = ROOT.parent / "structure_prototype_oracle_audit" / "training_outputs" / protocol / "Structure_AUDIT" / "seed_1993"
        tasks = _read_csv(original / "task_accuracy.csv")
        summaries = _read_csv(original / "task_summary.csv")
        immediate = next(float(row["accuracy"]) for row in tasks if int(row["trained_task"]) == 9 and int(row["evaluated_task"]) == 9)
        later = next(float(row["accuracy"]) for row in tasks if int(row["trained_task"]) == 10 and int(row["evaluated_task"]) == 9)
        seen = next(float(row["accuracy"]) for row in summaries if int(row["task_id"]) == 10)
        lines.append(f"| {protocol} | {immediate:.2f} | {later:.2f} | {seen:.2f} |")
    lines += ["", "이 one-view 값은 augmentation/view exposure가 달라 JS 효과 추정에 사용하지 않는다."]
    for protocol in PROTOCOLS:
        lines += ["", f"### {protocol} 시각화", "",
                  f"![T9 New]({protocol}/plots/t9_new_control_vs_js.png)", "",
                  f"![T9 retention]({protocol}/plots/t9_retention.png)", "",
                  f"![T10 cohorts]({protocol}/plots/t10_cohorts.png)", "",
                  f"![T9 per-class delta]({protocol}/plots/t9_per_class_delta.png)"]
    lines += ["", "## 판정", ""]
    for protocol in PROTOCOLS:
        row = phase_b[protocol]["delta"]
        if row["t9_new"] <= 0:
            status = "S1/S2: 즉시 held-out T9 이득 없음"
        elif row["t10_newly_old_t9"] <= 0:
            status = "S3: T10에서 T9 이득 소실"
        elif row["t10_seen"] <= 0 or row["t10_prior_old"] < -row["t10_newly_old_t9"]:
            status = "S4: 전체 또는 prior-Old 비용이 더 큼"
        else:
            status = "개별 protocol에서는 추가 확인 후보"
        lines.append(f"- {protocol}: {status}.")
    if result["consistency_prototype_promising"]:
        lines.append("- IA/CAR 양쪽에서 사전 기준을 충족했으므로 full 10-task와 독립 seed 검증으로만 확장한다. 아직 방법론 성립을 주장하지 않는다.")
    else:
        lines.append("- 두 protocol 공통 성공 조건을 충족하지 못해 full 10-task/독립 seed 확대를 보류한다. teacher·selector·stronger augmentation을 추가해 이 pilot을 구제하지 않는다.")
    lines += ["", f"- BOUNDARY AXIS CLOSED = {'YES' if result['boundary_axis_closed'] else 'NO'}",
              f"- CONSISTENCY PROTOTYPE PROMISING = {'YES' if result['consistency_prototype_promising'] else 'NO'}",
              "- FULL METHOD READY = NO", "",
              "## 제한", "",
              "단일 seed, T9→T10 두 transition만 관찰했다. 10-task 전체의 평균 성능을 재측정한 것이 아니다. 이 pilot의 unscaled cosine logit JS는 대략 10⁻⁴~10⁻³ 크기였으므로, null 결과는 지정된 T=1·λ=0.1 설정에 대한 것이지 모든 consistency regularization의 실패를 뜻하지 않는다. 다만 명세의 stop rule에 따라 이번 pilot을 추가 tuning하지 않는다. 원본 one-view Structure와의 수치 차이는 augmentation/view exposure까지 달라 인과 비교가 아니다.", "",
              "코드 경로와 동등성 검사는 `audit.md`, 실행·사전 판정 기준은 `implementation_notes.md`, 원자료는 `run_manifest.json`과 protocol별 CSV에 있다.", ""]
    (ROOT / "final_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    generate()
