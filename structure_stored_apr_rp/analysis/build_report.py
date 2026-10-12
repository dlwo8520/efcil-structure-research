#!/usr/bin/env python3
"""Build a read-only comparison against existing Structure and RP outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTIVE = ROOT.parent / "structure_selective_similarity_sweep" / "outputs"
OLD_RP = ROOT.parent / "structure_rp_protoclad_task5" / "outputs"
IA_STRUCTURE = (
    ROOT.parent / "structure_e3_followup" / "outputs" / "imageneta"
    / "B0_STRUCTURE" / "seed_1993"
)
RUN_ID = "StoredAPR_RP_C_P020_L030"
METRICS = ("Inc-Acc", "Last-Acc", "Forgetting", "Old", "New")


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_csv(path):
    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, signed=False):
    value = number(value)
    return "—" if value is None else f"{value:+.3f}" if signed else f"{value:.3f}"


def mean(rows, key):
    values = [number(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def performance(run_dir):
    summary = read_json(run_dir / "summary.json")
    metadata = read_json(run_dir / "run_metadata.json")
    final = [row for row in read_csv(run_dir / "task_summary.csv") if row.get("eval_stage") == "task_final"]
    final.sort(key=lambda row: int(row["task_id"]))
    complete = (
        summary.get("status") == "complete"
        and int(summary.get("completed_tasks", 0)) == 5
        and metadata.get("status") == "complete"
        and len(final) == 5
    )
    return {
        "status": "COMPLETE" if complete else metadata.get("status", "NOT_STARTED"),
        "Inc-Acc": number(summary.get("Inc-Acc")) if complete else None,
        "Last-Acc": number(summary.get("Last-Acc")) if complete else None,
        "Forgetting": number(summary.get("Forgetting")) if complete else None,
        "Old": number(final[-1].get("old_accuracy")) if complete else None,
        "New": number(final[-1].get("new_accuracy")) if complete else None,
        "A_by_task": summary.get("A_by_task", []) if complete else [],
    }


def baseline_dir(protocol, name):
    if protocol == "IA5" and name == "Structure":
        return IA_STRUCTURE
    if name == "RP_C_P020_L030":
        return OLD_RP / protocol / name / "seed_1993"
    if name == RUN_ID:
        return ROOT / "outputs" / protocol / name / "seed_1993"
    return SELECTIVE / protocol / name / "seed_1993"


def build_report():
    rows = []
    lines = [
        "# StoredAPR RP-ProtoCLAD 비교 보고서", "",
        "실험은 기존 `RP_C_P020_L030`을 보존한 별도 ablation이다. 새 방식은 "
        "`mean(ReLU(z_birth R)) + ReLU(mu_APR R) - ReLU(mu_birth R)`를 old class에 사용한다.",
        "원공간 Structure/APR/SCC/classifier adaptation과 Top-P 선택은 변경하지 않았다.", "",
        "## 성능 (단위: percentage point)", "",
        "| Protocol | Run | 상태 | Inc | Last | Forgetting | Old | New | ΔStruct Inc | ΔOldRP Inc |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    performances = {}
    for protocol in ("IA5", "CAR5"):
        for name in ("Structure", "C_P020_L030", "RP_C_P020_L030", RUN_ID):
            performances[(protocol, name)] = performance(baseline_dir(protocol, name))
        struct, old, new = (
            performances[(protocol, name)]
            for name in ("Structure", "RP_C_P020_L030", RUN_ID)
        )
        for name in ("Structure", "C_P020_L030", "RP_C_P020_L030", RUN_ID):
            item = performances[(protocol, name)]
            def delta(reference, key):
                return item[key] - reference[key] if item[key] is not None and reference[key] is not None else None
            lines.append(
                f"| {protocol} | {name} | {item['status']} | "
                + " | ".join(fmt(item[key]) for key in METRICS)
                + f" | {fmt(delta(struct, 'Inc-Acc'), True)} | {fmt(delta(old, 'Inc-Acc'), True)} |"
            )
            if name == RUN_ID:
                row = {"protocol": protocol, "run_id": name, "status": item["status"]}
                row.update({key: item[key] for key in METRICS})
                for key in METRICS:
                    row[f"delta_Struct_{key}"] = delta(struct, key)
                    row[f"delta_OldRP_{key}"] = delta(old, key)
                rows.append(row)
    lines += ["", "전체 항목의 `ΔStruct_*`, `ΔOldRP_*`는 "
              "[`outputs/comparison_metrics.csv`](outputs/comparison_metrics.csv)에 저장한다.", "",
              "## Prototype 및 geometry diagnostic", "",
              "| Protocol | Birth rel-L2 (oldstyle vs true) | Stored vs oldstyle rel-L2 | APR Δq norm | Selected RP cosine | RP gap | Gradient cos(cls, RP) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in ("IA5", "CAR5"):
        run_dir = baseline_dir(protocol, RUN_ID)
        proto = read_csv(run_dir / "stored_rp_prototype_diagnostics.csv")
        birth = [row for row in proto if row.get("stage") == "birth"]
        old = [row for row in proto if row.get("stage") == "old_epoch"]
        pairs = read_csv(run_dir / "rp_pair_geometry_diagnostics.csv")
        gaps = read_csv(run_dir / "rp_gap_diagnostics.csv")
        gradients = read_csv(run_dir / "rp_gradient_diagnostics.csv")
        lines.append(
            f"| {protocol} | {fmt(mean(birth, 'birth_relative_l2'))} | "
            f"{fmt(mean(old, 'stored_vs_oldstyle_relative_l2'))} | "
            f"{fmt(mean(old, 'apr_delta_norm'))} | "
            f"{fmt(mean(pairs, 'rp_cosine'))} | "
            f"{fmt(mean(gaps, 'gap_rp'))} | "
            f"{fmt(mean(gradients, 'cos_grad_cls_rp_clad'))} |"
        )
    lines += ["", "`birth_relative_l2`의 분모는 정확한 `q_birth`; "
              "`stored_vs_oldstyle_relative_l2`의 분모는 해당 epoch의 old-style prototype이다.",
              "Geometry와 accuracy 사이의 인과관계는 이 두 run만으로 확정할 수 없다.", "",
              "## 핵심 질문", ""]
    both = all(
        performances[(protocol, name)]["status"] == "COMPLETE"
        for protocol in ("IA5", "CAR5")
        for name in (RUN_ID, "RP_C_P020_L030")
    )
    if both:
        delta_inc = {}
        for protocol in ("IA5", "CAR5"):
            new, old = performances[(protocol, RUN_ID)], performances[(protocol, "RP_C_P020_L030")]
            delta_inc[protocol] = new["Inc-Acc"] - old["Inc-Acc"]
            lines.append(f"- {protocol}: ΔOldRP Inc {fmt(new['Inc-Acc'] - old['Inc-Acc'], True)}, "
                         f"Last {fmt(new['Last-Acc'] - old['Last-Acc'], True)}, "
                         f"Forgetting {fmt(new['Forgetting'] - old['Forgetting'], True)}, "
                         f"Old {fmt(new['Old'] - old['Old'], True)}, "
                         f"New {fmt(new['New'] - old['New'], True)}.")
        improved = [protocol for protocol, value in delta_inc.items() if value > 0]
        lines.append(f"- Q1: Inc 개선 protocol은 {', '.join(improved) if improved else '없음'}이다. "
                     "동일 seed 단일 run의 관측 차이이며 유의성 검정은 아니다.")
        lines.append("- Q2: Last/Forgetting/Old/New 변화는 위 ΔOldRP 다섯 지표를 참조한다. "
                     "Forgetting은 낮을수록 좋다.")
        lines.append("- Q3: birth-time oldstyle 근사를 학습용 old prototype에서 제거한 것은 코드상 확정이다. "
                     "birth 상대 L2와 stored-vs-oldstyle 상대 L2는 위 diagnostic을 참조한다.")
        lines.append("- Q4: APR displacement는 매 epoch 반영하며 위 APR Δq norm을 기록한다. "
                     "zero-drift 식은 단위시험에서 검증한다.")
        guards = []
        for protocol in ("IA5", "CAR5"):
            new, old = performances[(protocol, RUN_ID)], performances[(protocol, "RP_C_P020_L030")]
            passed = (new["New"] >= old["New"] - 0.5 and new["Old"] > old["Old"]
                      and new["Forgetting"] < old["Forgetting"])
            guards.append(f"{protocol} {'충족' if passed else '미충족'}")
        lines.append("- Q5: New 감소 ≤0.5pp, Old 증가, Forgetting 감소의 탐색적 조건: "
                     + ", ".join(guards) + ".")
        lines.append("- Q6: geometry/gradient/confusion의 동반 변화는 기술적 연관이며 "
                     "이 ablation만으로 인과관계를 확정할 수 없다.")
        same_direction = delta_inc["IA5"] * delta_inc["CAR5"] > 0
        lines.append(f"- Q7: 두 protocol의 Inc 변화 방향은 "
                     f"{'같다' if same_direction else '같지 않다 또는 한쪽이 0이다'}.")
    else:
        lines.append("- Q1–Q7: IA5/CAR5의 새 run과 기존 RP run이 모두 완료되기 전이므로 PENDING. "
                     "부분 결과로 결론 내리지 않는다.")
    lines += ["", "## 원자료", "",
              "각 새 run의 `task_summary.csv`, `class_accuracy.csv`, "
              "`stored_rp_prototype_diagnostics.csv`, `rp_pair_geometry_diagnostics.csv`, "
              "`rp_gap_diagnostics.csv`, `rp_gradient_diagnostics.csv`, "
              "`rp_confusion_diagnostics.csv`, `rp_selected_confusion.csv`를 보존한다.", ""]
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "comparison_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = ROOT / "stored_apr_rp_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(build_report())
