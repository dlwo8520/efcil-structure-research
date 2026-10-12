#!/usr/bin/env python3
"""Build a non-fabricated Stage-1/2 Markdown report from completed metrics."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PROTOCOLS = ("IA5", "CAR5")
HOLD_PROTOCOLS = ("IA10", "CAR10")
REPORT_RUN_IDS = (
    "Structure", "A_P005", "B_P020_A100", "C_P020_L030", "C_P020_L100"
)


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display(value, digits=3):
    if value in (None, ""):
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "outputs" / "metrics_per_seed.csv"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "selective_similarity_sweep" / "report_auto.md"),
        help="generated summary path; the hand-checked report.md is never overwritten by default",
    )
    cli = parser.parse_args()
    rows = read_csv(cli.metrics)
    lines = [
        "# Similarity-Selective Old-New Separation 결과 보고서",
        "",
        "Primary baseline: **vanilla Structure**. 모든 delta는 동일 protocol/seed의 "
        "Structure 대비 값이다.",
        "",
    ]
    if not rows:
        lines.extend(["아직 완료된 run이 없어 수치를 작성하지 않았습니다.", ""])
    counts = Counter(row["protocol"] for row in rows)
    lines.extend(["## 실행 완결성", "", "| Protocol | Completed metric rows |", "|---|---:|"])
    for protocol in (*ACTIVE_PROTOCOLS, *HOLD_PROTOCOLS):
        lines.append(f"| {protocol} | {counts[protocol]} |")
    for protocol in ACTIVE_PROTOCOLS:
        subset = [
            row for row in rows
            if row["protocol"] == protocol
            and row.get("seed") == "1993"
            and row.get("run_id") in REPORT_RUN_IDS
        ]
        if not subset:
            continue
        structure_available = any(row.get("run_id") == "Structure" for row in subset)
        lines.extend(["", f"## {protocol} Stage-1", ""])
        if not structure_available:
            lines.extend(
                [
                    "> 이 metrics snapshot에는 Structure row가 없어 기존 delta/guardrail을 "
                    "표시하지 않는다. Structure 완료 후 aggregate를 다시 생성해야 한다.",
                    "",
                ]
            )
        lines.extend(
            ["| Run | Method | P | alpha/lambda | Inc | ΔInc | Last | ΔLast | F↓ | ΔF | Old | ΔOld | New | ΔNew | Guardrail |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        )
        for row in subset:
            parameter = row.get("alpha") or row.get("lambda_clad") or "—"
            delta = lambda field: display(row.get(field)) if structure_available else "—"
            guardrail = row.get("guardrail_pass", "—") if structure_available else "—"
            lines.append(
                f"| {row['run_id']} | {row['method']} | {display(row.get('P'))} | {display(parameter)} | "
                f"{display(row.get('Inc-Acc'))} | {delta('delta_Inc-Acc')} | "
                f"{display(row.get('Last-Acc'))} | {delta('delta_Last-Acc')} | "
                f"{display(row.get('Forgetting'))} | {delta('delta_Forgetting')} | "
                f"{display(row.get('Old'))} | {delta('delta_Old')} | "
                f"{display(row.get('New'))} | {delta('delta_New')} | {guardrail} |"
            )
    lines.extend(
        ["", "## 해석 원칙", "",
         "- FROM CLAD CONCEPT: Top-P conflict selection, class-aware similarity, disentanglement.",
         "- ADAPTED FOR EXEMPLAR-FREE STRUCTURE: APR prototype, Selective AM-Drift, similarity-enlarged tau, Proto-CLAD.",
         "- Reduced Stage 1은 IA5/CAR5, seed 1993의 방법 탐색 결과입니다.",
         "- IA10 기존 결과는 참고용으로만 보존하고 CAR10과 함께 HOLD합니다.",
         "- 추가 seed와 10-task 검증은 5-task 최종 후보 1–2개를 정한 뒤 별도로 계획합니다."]
    )
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
