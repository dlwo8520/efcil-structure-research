#!/usr/bin/env python3
"""Build the live RP-ProtoCLAD report from baseline and RP artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rp.audit import RP_OUTPUT_ROOT, audit_rp, baseline_dir, classify


METRICS = ("Inc", "Last", "F", "Old", "New")


def _json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {} if default is None else default


def _csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _mean(rows, field):
    values = [_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _final_metrics(run_dir):
    run_dir = Path(run_dir)
    status, _ = classify(run_dir)
    if status != "COMPLETE":
        return None
    summary = _json(run_dir / "summary.json")
    final = [row for row in _csv(run_dir / "task_summary.csv") if row.get("eval_stage") == "task_final"]
    final = max(final, key=lambda row: int(row["task_id"]))
    return {
        "Inc": _number(summary.get("Inc-Acc")),
        "Last": _number(summary.get("Last-Acc")),
        "F": _number(summary.get("Forgetting")),
        "Old": _number(final.get("old_accuracy")),
        "New": _number(final.get("new_accuracy")),
    }


def _run_dir(protocol, run_id):
    if run_id in {"Structure", "C_P020_L030", "C_P020_L100"}:
        return baseline_dir(protocol, run_id)
    return RP_OUTPUT_ROOT / protocol / run_id / "seed_1993"


def _metric_rows(protocol):
    ids = (
        "Structure", "C_P020_L030", "RP_C_P020_L030",
        "C_P020_L100", "RP_C_P020_L100",
    )
    values = {run_id: _final_metrics(_run_dir(protocol, run_id)) for run_id in ids}
    rows = []
    for run_id in ids:
        current = values[run_id]
        parent_id = run_id.removeprefix("RP_") if run_id.startswith("RP_") else None
        struct = values["Structure"]
        parent = values.get(parent_id) if parent_id else None
        row = {"protocol": protocol, "run_id": run_id, "status": "COMPLETE" if current else "PENDING"}
        row.update(current or {key: None for key in METRICS})
        for key in METRICS:
            row[f"delta_struct_{key}"] = (
                current[key] - struct[key] if current and struct and current[key] is not None and struct[key] is not None else None
            )
            row[f"delta_parent_{key}"] = (
                current[key] - parent[key] if current and parent and current[key] is not None and parent[key] is not None else None
            )
        row["guardrail"] = (
            current is not None and struct is not None
            and row["delta_struct_Old"] >= -0.5
            and row["delta_struct_New"] >= -0.5
            and row["delta_struct_F"] <= 0.5
        ) if current and struct else None
        rows.append(row)
    return rows


def _fmt(value, signed=False):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _write_metrics_csv(rows):
    target = RP_OUTPUT_ROOT / "metrics_per_seed.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _geometry_rows(protocol, run_id):
    run_dir = _run_dir(protocol, run_id)
    pair = _csv(run_dir / "rp_pair_geometry_diagnostics.csv")
    gaps = _csv(run_dir / "rp_gap_diagnostics.csv")
    features = _csv(run_dir / "rp_feature_diagnostics.csv")
    gradients = _csv(run_dir / "rp_gradient_diagnostics.csv")
    tasks = sorted({int(row["task"]) for row in [*pair, *gaps, *features] if row.get("task", "").isdigit()})
    output = []
    for task in [*tasks, None]:
        select = lambda rows: rows if task is None else [row for row in rows if int(row.get("task", -1)) == task]
        pair_t, gap_t, feature_t, grad_t = map(select, (pair, gaps, features, gradients))
        output.append(
            {
                "protocol": protocol,
                "run_id": run_id,
                "task": "ALL" if task is None else task,
                "orig_cos": _mean(pair_t, "original_cosine"),
                "rp_cos": _mean(pair_t, "rp_cosine"),
                "delta_cos": _mean(pair_t, "delta_cosine"),
                "orig_gap": _mean(gap_t, "gap_original"),
                "rp_gap": _mean(gap_t, "gap_rp"),
                "delta_gap": _mean(gap_t, "delta_gap"),
                "zero_ratio": _mean(feature_t, "rp_zero_ratio"),
                "grad_cos": _mean(grad_t, "cos_grad_cls_rp_clad"),
            }
        )
    return output


def _answer(label, evidence):
    return f"- **{label}: {evidence[0]}** — {evidence[1]}"


def _judge_improvement(rp_row):
    if rp_row["status"] != "COMPLETE" or rp_row["delta_parent_Inc"] is None:
        return "INCONCLUSIVE", "RP 또는 parent 결과가 아직 완료되지 않았다."
    acceptable = (
        rp_row["delta_parent_New"] >= -0.5
        and rp_row["delta_parent_Old"] >= -0.5
        and rp_row["delta_parent_F"] <= 0.5
    )
    if rp_row["delta_parent_Inc"] > 0 and acceptable:
        return "YES", f"ΔParent Inc={rp_row['delta_parent_Inc']:+.3f} pp이고 parent-relative trade-off 허용범위도 만족한다."
    return "NO", f"ΔParent Inc={rp_row['delta_parent_Inc']:+.3f} pp 또는 parent-relative stability/plasticity 조건을 만족하지 못했다."


def build_report():
    metric_rows = [row for protocol in ("IA5", "CAR5") for row in _metric_rows(protocol)]
    _write_metrics_csv(metric_rows)
    rp_audit = audit_rp()
    geometry = [
        row for protocol in ("IA5", "CAR5")
        for run_id in ("RP_C_P020_L030", "RP_C_P020_L100")
        for row in _geometry_rows(protocol, run_id)
    ]
    by_key = {(row["protocol"], row["run_id"]): row for row in metric_rows}
    overall_geom = {
        (row["protocol"], row["run_id"]): row for row in geometry if row["task"] == "ALL"
    }

    lines = [
        "# RP-ProtoCLAD Task-5 결과 보고서",
        "",
        "> 이 문서는 output이 추가될 때 재생성되는 live report다. `—`는 아직 완료 결과가 없다는 뜻이다.",
        "",
        "## 방법 구분",
        "",
        "Prior work idea: fixed random projection을 이용한 고차원 nonlinear random feature representation.",
        "",
        "Our experimental adaptation: Structure의 classifier/SCC/APR/classifier adaptation은 원 feature 공간에 유지하고, frozen RP를 Proto-CLAD old-new conflict regularization cosine에만 적용한다. 기존 논문의 직접 재현이 아니다.",
        "",
        "Top-P conflict selection 역시 기존 original-space frozen similarity를 그대로 사용한다.",
        "",
        "## 실행 상태",
        "",
        "| Protocol | Run | Parent | Status | Action |",
        "|---|---|---|---|---|",
    ]
    for row in rp_audit:
        lines.append(f"| {row['protocol']} | {row['run_id']} | {row['parent_run_id']} ({row['parent_status']}) | {row['status']} | {row['action']} |")

    for protocol in ("IA5", "CAR5"):
        lines.extend(
            [
                "",
                f"## {protocol}",
                "",
                "| Method | Inc | ΔStruct | ΔParent | Last | F↓ | Old | New | Guardrail |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in [item for item in metric_rows if item["protocol"] == protocol]:
            lines.append(
                f"| {row['run_id']} | {_fmt(row['Inc'])} | {_fmt(row['delta_struct_Inc'], True)} | "
                f"{_fmt(row['delta_parent_Inc'], True)} | {_fmt(row['Last'])} | {_fmt(row['F'])} | "
                f"{_fmt(row['Old'])} | {_fmt(row['New'])} | {_fmt(row['guardrail'])} |"
            )
        lines.append("")
        lines.append("전체 delta(Last/F/Old/New 포함)는 `outputs/metrics_per_seed.csv`에 저장한다.")

    lines.extend(
        [
            "",
            "## RP 전후 geometry",
            "",
            "| Protocol | Run | Task | orig selected cosine | RP selected cosine | Δcos | orig gap | RP gap | Δgap | RP zero ratio | grad cos(cls, RP) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in geometry:
        lines.append(
            f"| {row['protocol']} | {row['run_id']} | {row['task']} | {_fmt(row['orig_cos'])} | "
            f"{_fmt(row['rp_cos'])} | {_fmt(row['delta_cos'], True)} | {_fmt(row['orig_gap'])} | "
            f"{_fmt(row['rp_gap'])} | {_fmt(row['delta_gap'], True)} | {_fmt(row['zero_ratio'])} | "
            f"{_fmt(row['grad_cos'])} |"
        )

    complete_rp = [row for row in metric_rows if row["run_id"].startswith("RP_") and row["status"] == "COMPLETE"]
    q1_values = [_judge_improvement(by_key[(p, "RP_C_P020_L030")]) for p in ("IA5", "CAR5")]
    q2_values = [_judge_improvement(by_key[(p, "RP_C_P020_L100")]) for p in ("IA5", "CAR5")]
    q1 = ("INCONCLUSIVE", "두 dataset 결과가 모두 필요하다.") if len(complete_rp) < 4 else (
        "YES" if all(value[0] == "YES" for value in q1_values) else "NO",
        "; ".join(f"{p}: {v[0]}" for p, v in zip(("IA5", "CAR5"), q1_values)),
    )
    q2 = ("INCONCLUSIVE", "두 dataset 결과가 모두 필요하다.") if len(complete_rp) < 4 else (
        "YES" if all(value[0] == "YES" for value in q2_values) else "NO",
        "; ".join(f"{p}: {v[0]}" for p, v in zip(("IA5", "CAR5"), q2_values)),
    )
    geom_ready = len(overall_geom) == 4 and all(row["delta_cos"] is not None for row in overall_geom.values())
    geom_improved = geom_ready and all(
        row["delta_cos"] < 0 or (row["delta_gap"] is not None and row["delta_gap"] > 0)
        for row in overall_geom.values()
    )
    q3 = (
        ("YES" if geom_improved else "NO"),
        "모든 RP run에서 Δcos<0 또는 Δgap>0인지로 판정했다.",
    ) if geom_ready else ("INCONCLUSIVE", "geometry diagnostic이 아직 완결되지 않았다.")
    accuracy_improved = len(complete_rp) == 4 and all(row["delta_parent_Inc"] > 0 for row in complete_rp)
    q4 = (
        ("YES" if geom_improved and accuracy_improved else "NO"),
        "geometry와 ΔParent Inc의 동시 개선 여부를 사용했다.",
    ) if geom_ready and len(complete_rp) == 4 else ("INCONCLUSIVE", "geometry/accuracy 결과가 아직 완결되지 않았다.")
    l100 = [by_key[(p, "RP_C_P020_L100")] for p in ("IA5", "CAR5")]
    q5 = (
        "YES" if all(row["delta_parent_New"] >= 0 for row in l100) else "NO",
        "lambda=1.0의 두 dataset에서 ΔParent New>=0인지로 판정했다.",
    ) if all(row["status"] == "COMPLETE" for row in l100) else ("INCONCLUSIVE", "lambda=1.0 결과가 아직 완결되지 않았다.")
    q6 = (
        "YES" if all(row["delta_parent_Old"] >= 0 and row["delta_parent_F"] <= 0 and row["delta_parent_New"] >= 0 for row in complete_rp) else "NO",
        "각 RP run에서 Parent 대비 Old>=0, F<=0, New>=0을 동시에 요구했다.",
    ) if len(complete_rp) == 4 else ("INCONCLUSIVE", "4개 RP 결과가 모두 필요하다.")
    q7_ready = len(complete_rp) == 4
    q7_same = q7_ready and all(
        math.copysign(1, by_key[("IA5", run)]["delta_parent_Inc"] or 0)
        == math.copysign(1, by_key[("CAR5", run)]["delta_parent_Inc"] or 0)
        for run in ("RP_C_P020_L030", "RP_C_P020_L100")
    )
    q7 = (("YES" if q7_same else "NO"), "lambda별 ΔParent Inc 방향을 IA5/CAR5에서 비교했다.") if q7_ready else ("INCONCLUSIVE", "두 dataset 결과가 아직 완결되지 않았다.")
    approximation = []
    for protocol in ("IA5", "CAR5"):
        for run_id in ("RP_C_P020_L030", "RP_C_P020_L100"):
            approximation.extend(_csv(_run_dir(protocol, run_id) / "rp_prototype_approximation.csv"))
    approx_cos = _mean(approximation, "cos_a_b")
    approx_rel = _mean(approximation, "relative_l2_a_b")
    q8 = (
        ("YES" if approx_cos >= 0.99 and approx_rel <= 0.10 else "NO"),
        f"사전 명시한 보수적 기준 mean cosine>=0.99 및 mean relative-L2<=0.10; 관측값={_fmt(approx_cos)}, {_fmt(approx_rel)}.",
    ) if approximation and len(complete_rp) == 4 else ("INCONCLUSIVE", "prototype approximation 결과가 아직 완결되지 않았다.")

    lines.extend(
        [
            "",
            "## 필수 질문 판정",
            "",
            _answer("Q1 — RP_C_P020_L030이 parent보다 좋아졌는가?", q1),
            _answer("Q2 — RP_C_P020_L100이 parent보다 좋아졌는가?", q2),
            _answer("Q3 — RP가 selected cosine 또는 correct-new gap을 개선했는가?", q3),
            _answer("Q4 — geometry 개선이 accuracy 개선으로 이어졌는가?", q4),
            _answer("Q5 — lambda=1.0의 New 손실이 회복됐는가?", q5),
            _answer("Q6 — Old/F를 유지하며 New도 유지·개선했는가?", q6),
            _answer("Q7 — IA5/CAR5에서 같은 방향으로 반복됐는가?", q7),
            _answer("Q8 — nonlinear prototype approximation gap이 충분히 작았는가?", q8),
            "",
            "## 해석 원칙",
            "",
            "RP 자체 효과는 Structure delta보다 동일 lambda parent의 `ΔParent_*`를 우선한다. Inc가 상승해도 New가 크게 하락하면 긍정 결과로 해석하지 않으며, Old만 상승하고 New가 하락하면 stability-plasticity 이동으로 기록한다.",
        ]
    )
    target = ROOT / "rp_protoclad_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"wrote {build_report()}")
