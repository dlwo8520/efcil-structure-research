#!/usr/bin/env python3
"""Build a non-fabricated Stage-1/2 Markdown report from completed metrics."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    lines = ["# Similarity-Selective Old-New Separation 결과 보고서", ""]
    if not rows:
        lines.extend(["아직 완료된 run이 없어 수치를 작성하지 않았습니다.", ""])
    counts = Counter(row["protocol"] for row in rows)
    lines.extend(["## 실행 완결성", "", "| Protocol | Completed metric rows |", "|---|---:|"])
    for protocol in ("IA5", "IA10", "CAR5", "CAR10"):
        lines.append(f"| {protocol} | {counts[protocol]} |")
    for protocol in ("IA5", "IA10", "CAR5", "CAR10"):
        subset = [row for row in rows if row["protocol"] == protocol and row.get("seed") == "1993"]
        if not subset:
            continue
        lines.extend(
            ["", f"## {protocol} Stage-1", "",
             "| Run | Method | P | alpha/lambda | Inc | ΔInc | Last | ΔLast | F↓ | ΔF | Old | ΔOld | New | ΔNew | Guardrail |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        )
        for row in subset:
            parameter = row.get("alpha") or row.get("lambda_clad") or "—"
            lines.append(
                f"| {row['run_id']} | {row['method']} | {display(row.get('P'))} | {display(parameter)} | "
                f"{display(row.get('Inc-Acc'))} | {display(row.get('delta_Inc-Acc'))} | "
                f"{display(row.get('Last-Acc'))} | {display(row.get('delta_Last-Acc'))} | "
                f"{display(row.get('Forgetting'))} | {display(row.get('delta_Forgetting'))} | "
                f"{display(row.get('Old'))} | {display(row.get('delta_Old'))} | "
                f"{display(row.get('New'))} | {display(row.get('delta_New'))} | {row.get('guardrail_pass', '—')} |"
            )
    lines.extend(
        ["", "## 해석 원칙", "",
         "- FROM CLAD CONCEPT: Top-P conflict selection, class-aware similarity, disentanglement.",
         "- ADAPTED FOR EXEMPLAR-FREE STRUCTURE: APR prototype, Selective AM-Drift, similarity-enlarged tau, Proto-CLAD.",
         "- Stage 1은 seed 1993 개발 결과이며, 최종 주장은 동결된 Stage-2 seed 1996/1997 확인 뒤 작성합니다.",
         "- 5-task/10-task 및 ImageNet-A/Cars-196 비교는 네 protocol 결과가 모두 완결된 뒤 판단합니다."]
    )
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
