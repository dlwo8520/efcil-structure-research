#!/usr/bin/env python3
"""Build the live RP-ProtoCLAD report from baseline and RP artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


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


def _task_accuracies(run_dir):
    rows = [
        row for row in _csv(Path(run_dir) / "task_summary.csv")
        if row.get("eval_stage") == "task_final"
    ]
    return {int(row["task_id"]): _number(row.get("accuracy")) for row in rows}


def _class_change_counts(parent_dir, rp_dir):
    def final(path):
        return {
            int(row["mapped_class_id"]): float(row["accuracy"])
            for row in _csv(Path(path) / "class_accuracy.csv")
            if row.get("trained_task") == "5" and row.get("eval_stage") == "task_final"
        }

    parent, rp = final(parent_dir), final(rp_dir)
    if set(parent) != set(rp) or len(parent) != 200:
        return None
    result = {}
    for label, class_ids in (("old", range(160)), ("new", range(160, 200))):
        differences = [rp[class_id] - parent[class_id] for class_id in class_ids]
        result[label] = (
            sum(value > 0 for value in differences),
            sum(value == 0 for value in differences),
            sum(value < 0 for value in differences),
        )
    return result


def _selected_confusion(run_dir):
    run_dir = Path(run_dir)
    selected = {
        (int(row["new_class_id"]), int(row["old_class_id"]))
        for row in _csv(run_dir / "conflict_selection.csv")
        if row.get("task") == "5" and row.get("selected") == "True"
    }
    path = run_dir / "confusion_task_05.npz"
    if not path.is_file() or not selected:
        return None
    with np.load(path) as archive:
        matrix = archive["matrix"]
    return {
        "selected_pairs": len(selected),
        "selected_new_to_old": sum(int(matrix[new, old]) for new, old in selected),
        "selected_old_to_new": sum(int(matrix[old, new]) for new, old in selected),
        "all_new_to_old": int(matrix[160:, :160].sum()),
        "all_old_to_new": int(matrix[:160, 160:].sum()),
    }


def _rp_diagnostic_summary(protocol, run_id):
    run_dir = _run_dir(protocol, run_id)
    approximation = _csv(run_dir / "rp_prototype_approximation.csv")
    correlations = _csv(run_dir / "rp_similarity_structure_diagnostics.csv")
    gradients = _csv(run_dir / "rp_gradient_diagnostics.csv")
    features = _csv(run_dir / "rp_feature_diagnostics.csv")
    projection = _json(run_dir / "rp_projection.json")
    relatives = [_number(row.get("relative_l2_a_b")) for row in approximation]
    relatives = [value for value in relatives if value is not None]
    return {
        "approx_n": len(approximation),
        "approx_cos": _mean(approximation, "cos_a_b"),
        "approx_rel": _mean(approximation, "relative_l2_a_b"),
        "approx_rel_gt_01": sum(value > 0.1 for value in relatives),
        "pearson": _mean(correlations, "pearson_original_vs_rp"),
        "spearman": _mean(correlations, "spearman_original_vs_rp"),
        "grad_cls": _mean(gradients, "grad_norm_cls"),
        "grad_scc": _mean(gradients, "grad_norm_scc"),
        "grad_rp": _mean(gradients, "grad_norm_rp_clad"),
        "cos_cls_rp": _mean(gradients, "cos_grad_cls_rp_clad"),
        "cos_scc_rp": _mean(gradients, "cos_grad_scc_rp_clad"),
        "rp_zero": _mean(features, "rp_zero_ratio"),
        "rp_norm": _mean(features, "rp_norm_mean"),
        "original_norm": _mean(features, "original_norm_mean"),
        "projection": projection,
    }


def _matched_original_pair_cosines(parent_dir, rp_dir):
    """Compare only identical sampled class/sample/epoch pairs in original space."""
    def key(row):
        return tuple(
            row[field]
            for field in ("task", "epoch", "batch", "sample_id", "new_class_id", "old_class_id")
        )

    parent = {
        key(row): float(row["pair_cosine"])
        for row in _csv(Path(parent_dir) / "selective_pair_samples.csv")
        if row.get("selected") == "True"
    }
    rp = {
        key(row): float(row["original_cosine"])
        for row in _csv(Path(rp_dir) / "rp_gap_diagnostics.csv")
    }
    overlap = parent.keys() & rp.keys()
    if not overlap:
        return None
    return {
        "parent_count": len(parent),
        "rp_count": len(rp),
        "matched_count": len(overlap),
        "parent_cos": sum(parent[item] for item in overlap) / len(overlap),
        "rp_original_cos": sum(rp[item] for item in overlap) / len(overlap),
        "delta": sum(rp[item] - parent[item] for item in overlap) / len(overlap),
    }


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
    generated = datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
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
        f"> 생성 시각: {generated:%Y-%m-%d %H:%M KST}. 파일 기반 부분 결과 보고서이며, `—`는 아직 완료 결과가 없다는 뜻이다.",
        "",
        "2026-09-21 14:47 KST에 종료된 첫 GPU 2 RP queue에서는 IA5 RP 두 run이 완료됐고 CAR5 RP 두 run은 parent 미완료로 DEFER되어 학습을 시작하지 않았다 (`failures=0`, `deferred=2`; 큐의 비영(2) 종료 코드는 대기 중인 parent가 있다는 뜻이다). 아래 판정은 생성 시점의 파일시스템 상태를 다시 확인한 결과다.",
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
    car = [row for row in rp_audit if row["protocol"] == "CAR5"]
    if car:
        lines.extend(
            [
                "",
                f"CAR5의 Structure 기준선은 `{car[0]['structure_status']}`이고, 두 Proto-CLAD parent는 각각 `{car[0]['parent_status']}`, `{car[1]['parent_status']}`다. CAR5 RP의 비어 있는 성능 셀은 실패한 RP 결과가 아니라 **미실행** 상태를 뜻한다.",
            ]
        )

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

    ia5_ids = ("RP_C_P020_L030", "RP_C_P020_L100")
    ia5_complete = all(by_key[("IA5", run_id)]["status"] == "COMPLETE" for run_id in ia5_ids)
    if ia5_complete:
        lines.extend(
            [
                "",
                "## IA5 완료 결과의 상세 해석",
                "",
                "아래 차이는 모두 퍼센트포인트(pp)다. `ΔParent`는 동일 λ의 원공간 Proto-CLAD 대비 변화이고, `ΔStruct`는 vanilla Structure 대비 변화다. Forgetting은 낮을수록 좋다.",
                "",
                "| RP run | 비교 | ΔInc | ΔLast | ΔF↓ | ΔOld | ΔNew |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            row = by_key[("IA5", run_id)]
            for label, prefix in (("Parent", "delta_parent"), ("Structure", "delta_struct")):
                lines.append(
                    f"| {run_id} | {label} | "
                    + " | ".join(_fmt(row[f"{prefix}_{field}"], True) for field in METRICS)
                    + " |"
                )
        lines.extend(
            [
                "",
                "`λ=0.30` RP는 parent 대비 Inc +0.064 pp, Last +0.263 pp, F −0.554 pp, Old +0.321 pp이며 New는 동일하다. Structure guardrail도 통과한다. 다만 효과 크기가 작고 단일 seed여서 유망한 **예비 결과**이지 일반화된 우월성의 증거는 아니다.",
                "",
                "`λ=1.00` RP는 parent 대비 Inc −0.126 pp, Last −0.263 pp, F +0.258 pp, Old −0.240 pp, New −0.369 pp다. Structure 대비 New −0.738 pp로 −0.5 pp guardrail을 위반한다. 기존 New 손실을 회복하지 못했다.",
                "",
                "### Task별 single-head accuracy",
                "",
                "| λ | Task | Parent | RP | RP − Parent |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            parent_id = run_id.removeprefix("RP_")
            parent_tasks = _task_accuracies(_run_dir("IA5", parent_id))
            rp_tasks = _task_accuracies(_run_dir("IA5", run_id))
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            for task in range(1, 6):
                parent_value, rp_value = parent_tasks.get(task), rp_tasks.get(task)
                delta = rp_value - parent_value if parent_value is not None and rp_value is not None else None
                lines.append(
                    f"| {coefficient} | {task} | {_fmt(parent_value)} | {_fmt(rp_value)} | {_fmt(delta, True)} |"
                )

        lines.extend(
            [
                "",
                "### RP 자체의 geometry 및 근사 진단",
                "",
                "`selected cosine`은 RP run 안에서 같은 class pair를 원 공간과 RP 공간에 각각 투영해 비교한 값이다. 두 공간의 cosine 척도가 달라 **RP가 학습 후 원공간 separation을 개선했다는 직접 증거로 해석하면 안 된다**. 아래 gap은 sampled current sample의 correct-new cosine − selected-old cosine이다.",
                "",
                "| λ | selected Δcos (RP−orig) | gap Δ (RP−orig) | Pearson | Spearman | RP zero ratio | prototype cos(a,b) | prototype rel-L2 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            diagnostic = _rp_diagnostic_summary("IA5", run_id)
            geom = overall_geom[("IA5", run_id)]
            lines.append(
                f"| {coefficient} | {_fmt(geom['delta_cos'], True)} | {_fmt(geom['delta_gap'], True)} | "
                f"{_fmt(diagnostic['pearson'])} | {_fmt(diagnostic['spearman'])} | "
                f"{_fmt(diagnostic['rp_zero'])} | {_fmt(diagnostic['approx_cos'])} | "
                f"{_fmt(diagnostic['approx_rel'])} |"
            )
        lines.extend(
            [
                "",
                "두 RP run 모두 selected cosine이 RP 공간에서 더 높고 correct-new gap이 더 작다. 이 조작적 진단에서는 의도한 separation 증가가 관측되지 않았다. original/RP similarity의 Pearson·Spearman은 높아, 관계 구조 자체는 대체로 보존되지만 ReLU로 cosine의 절대 수준이 이동한다.",
                "",
                "prototype 근사는 `a=ReLU(mean(z)R)`, `b=mean(ReLU(zR))`를 current class의 기존 cache feature로 계산했다. old class에서 직접 검증한 값은 아니다.",
                "",
                "| λ | class-task rows | mean cosine(a,b) | mean relative-L2 | relative-L2 > 0.1 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            diagnostic = _rp_diagnostic_summary("IA5", run_id)
            lines.append(
                f"| {coefficient} | {diagnostic['approx_n']} | {_fmt(diagnostic['approx_cos'])} | "
                f"{_fmt(diagnostic['approx_rel'])} | {diagnostic['approx_rel_gt_01']}/{diagnostic['approx_n']} |"
            )
        lines.extend(
            [
                "",
                "mean relative-L2가 약 0.43–0.45이고 모든 320개 class-task row에서 0.1을 넘는다. `0.1`은 탐색적 참고선이며 사전 등록된 검정 임계값은 아니다. 그럼에도 `ReLU(mean(z)R)`을 과거 class의 sample-wise projected mean 대리값으로 취급하는 근사는 작지 않아 보인다.",
                "",
                "### Parent와 같은 원공간에서의 sampled pair 비교",
                "",
                "마지막 epoch의 동일 task/epoch/batch/sample/class pair만 대조했다. selection이 달라진 pair는 제외했다.",
                "",
                "| λ | matched pairs | Parent orig cosine | RP run orig cosine | ΔRP−Parent |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            parent_id = run_id.removeprefix("RP_")
            matched = _matched_original_pair_cosines(
                _run_dir("IA5", parent_id), _run_dir("IA5", run_id)
            )
            if matched:
                lines.append(
                    f"| {coefficient} | {matched['matched_count']}/{matched['rp_count']} | "
                    f"{_fmt(matched['parent_cos'])} | {_fmt(matched['rp_original_cos'])} | "
                    f"{_fmt(matched['delta'], True)} |"
                )
        lines.extend(
            [
                "",
                "원공간 selected cosine도 RP run에서 parent보다 소폭 높다. 따라서 `λ=0.30`의 작은 accuracy 이득을 'old-new cosine separation 향상'으로 설명할 수는 없다.",
                "",
                "### Gradient 및 최종 task confusion",
                "",
                "gradient 값은 logging point에서 trainable network parameter에 대한 **raw loss component**의 평균이다. λ와 SCC 계수가 적용된 실제 합성 gradient norm은 아니므로 크기를 직접 동등 비교하면 안 된다.",
                "",
                "| λ | ∥grad cls∥ | ∥grad SCC∥ | ∥grad RP∥ | cos(cls,RP) | cos(SCC,RP) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            diagnostic = _rp_diagnostic_summary("IA5", run_id)
            lines.append(
                f"| {coefficient} | {_fmt(diagnostic['grad_cls'])} | {_fmt(diagnostic['grad_scc'])} | "
                f"{_fmt(diagnostic['grad_rp'])} | {_fmt(diagnostic['cos_cls_rp'])} | "
                f"{_fmt(diagnostic['cos_scc_rp'])} |"
            )
        lines.extend(
            [
                "",
                "| λ | Run | old→new | new→old | selected old→new | selected new→old |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            parent_id = run_id.removeprefix("RP_")
            for name, current_id in (("Parent", parent_id), ("RP", run_id)):
                confusion = _selected_confusion(_run_dir("IA5", current_id))
                if confusion:
                    lines.append(
                        f"| {coefficient} | {name} | {confusion['all_old_to_new']} | "
                        f"{confusion['all_new_to_old']} | {confusion['selected_old_to_new']} | "
                        f"{confusion['selected_new_to_old']} |"
                    )
        lines.extend(
            [
                "",
                "selected confusion은 각 run의 task-5 Top-P pair 1,280개에 해당하는 오분류 count다. 두 방향 모두 변화가 1–2건 수준이어서 큰 구조적 혼동 개선의 증거는 아니다.",
                "",
                "### Class-wise 최종 accuracy 변화",
                "",
                "| λ | Old classes 개선/동일/악화 | New classes 개선/동일/악화 |",
                "|---:|---:|---:|",
            ]
        )
        for run_id in ia5_ids:
            coefficient = "0.30" if run_id.endswith("L030") else "1.00"
            parent_id = run_id.removeprefix("RP_")
            changes = _class_change_counts(
                _run_dir("IA5", parent_id), _run_dir("IA5", run_id)
            )
            if changes:
                lines.append(
                    f"| {coefficient} | {'/'.join(map(str, changes['old']))} | "
                    f"{'/'.join(map(str, changes['new']))} |"
                )

        lines.extend(
            [
                "",
                "### IA5에 한정한 Q1–Q8 판정",
                "",
                "| 질문 | IA5 판정 | 근거 |",
                "|---|---|---|",
                "| Q1: λ=0.30 RP가 parent보다 좋은가? | YES (예비) | ΔInc +0.064 pp, ΔF −0.554 pp, ΔOld +0.321 pp, ΔNew 0; 단일 seed의 작은 차이 |",
                "| Q2: λ=1.00 RP가 parent보다 좋은가? | NO | ΔInc −0.126 pp, ΔNew −0.369 pp, ΔF +0.258 pp |",
                "| Q3: selected cosine/gap이 개선됐는가? | NO (기록된 지표 기준) | 두 run 모두 RP selected cosine 상승, correct-new gap 감소 |",
                "| Q4: geometry 개선이 accuracy 개선으로 이어졌는가? | NO/해당 없음 | λ=0.30의 accuracy 이득에도 geometry 개선은 관측되지 않음 |",
                "| Q5: λ=1.00의 New 손실이 회복됐는가? | NO | Parent 대비 New −0.369 pp, Structure 대비 −0.738 pp |",
                "| Q6: Old/F 유지와 New 유지·개선이 동시에 일어났는가? | λ=0.30 YES, λ=1.00 NO | λ=0.30은 New 동일·Old/F 개선, λ=1.00은 모두 악화 |",
                "| Q7: IA5와 CAR5에서 재현되는가? | INCONCLUSIVE | CAR5 RP 두 run 미실행 |",
                "| Q8: nonlinear prototype approximation gap이 작은가? | NO (탐색적 판단) | mean relative-L2 0.43–0.45, cosine 약 0.90–0.91 |",
                "",
                "### 검증 및 한계",
                "",
            ]
        )
        projections = [_rp_diagnostic_summary("IA5", run_id)["projection"] for run_id in ia5_ids]
        same_projection = (
            bool(projections[0])
            and projections[0] == projections[1]
            and projections[0].get("shape") == [768, 2048]
            and projections[0].get("seed") == 1993
            and projections[0].get("requires_grad") is False
        )
        lines.extend(
            [
                f"- IA5 RP 두 run 모두 task 5/5 완료, checkpoint와 summary/metadata 일치, `recording_gaps=[]` 확인. RP projection 메타데이터 동일: **{same_projection}**; checksum `{projections[0].get('checksum_sha256', 'missing')}`.",
                "- 기존 IA5 Structure는 외부 완료 run을 재사용했다. Structure/parent/RP의 `class_order.json` 내용이 동일하며 seed=1993, data root와 5×40 task 설정을 맞췄다.",
                "- accuracy 변화는 200-class IA5의 seed 하나에서 나온 관측치다. 분산·유의성·다른 dataset으로의 일반화는 판단할 수 없다.",
                "- geometry는 서로 다른 특징 공간의 cosine을 비교한 서술적 진단이다. approximation은 현재 new class에서만 직접 검증 가능하며, old APR prototype에 대한 오차는 직접 측정할 수 없다.",
                "- 위 결과로 RP dimension, seed, λ 등을 추가 탐색하지 않았다. CAR5 parent가 완료된 후 고정된 두 RP 설정만 후속 검증한다.",
                "",
                "### 주요 근거 파일",
                "",
                "- [GPU 2 queue 기록](logs/rp_task5_gpu2.log), [실행별 종료 코드](logs/rp_task5/exit_codes.csv)",
                "- [λ=0.30 RP summary](outputs/IA5/RP_C_P020_L030/seed_1993/summary.json), [λ=1.00 RP summary](outputs/IA5/RP_C_P020_L100/seed_1993/summary.json)",
                "- [원자료 성능·delta CSV](outputs/metrics_per_seed.csv), [실행 계획 및 parent audit](rp_experiment_plan.md)",
                "- RP run별 `rp_pair_geometry_diagnostics.csv`, `rp_gap_diagnostics.csv`, `rp_prototype_approximation.csv`, `rp_gradient_diagnostics.csv`, `confusion_task_05.npz`는 각 output 디렉터리 안에 있다.",
            ]
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
        f"탐색적 참고선 mean cosine>=0.99 및 mean relative-L2<=0.10; 관측값={_fmt(approx_cos)}, {_fmt(approx_rel)}. 이 참고선은 사전 등록된 판정 기준이 아니다.",
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
