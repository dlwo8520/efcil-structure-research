#!/usr/bin/env python3
"""Regenerate the RP experiment plan from current filesystem state."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rp.audit import RP_OUTPUT_ROOT, SELECTIVE_OUTPUT_ROOT, audit_baselines, audit_rp


def _table(rows, fields):
    lines = ["| " + " | ".join(label for _, label in fields) + " |"]
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "—") or "—") for key, _ in fields) + " |")
    return lines


def build_plan(output_root=RP_OUTPUT_ROOT, selective_output_root=SELECTIVE_OUTPUT_ROOT):
    baselines = audit_baselines(selective_output_root)
    rp_rows = audit_rp(output_root, selective_output_root)
    counts = Counter(row["action"] for row in rp_rows)
    lines = [
        "# RP-ProtoCLAD Task-5 실행 계획",
        "",
        "## 고정 범위",
        "",
        "- protocol: IA5, CAR5만 사용",
        "- 새 run: RP_C_P020_L030, RP_C_P020_L100만 사용(총 4개)",
        "- seed=1993, P=0.20, RP dim=2048, RP seed=1993, ReLU, bias 없음, frozen",
        "- IA10/CAR10, 추가 seed/dimension/P/lambda/alpha sweep은 실행하지 않음",
        "- output은 기존 selective sweep과 분리하여 기존 결과를 덮어쓰지 않음",
        "",
        "## 현재 filesystem audit",
        "",
        f"RP action count: {dict(counts)}",
        "",
    ]
    lines.extend(
        _table(
            baselines,
            (("protocol", "Protocol"), ("run_id", "Baseline"), ("status", "Status"),
             ("evidence", "Evidence"), ("output_dir", "Output")),
        )
    )
    lines.extend(["", "## RP queue", ""])
    lines.extend(
        _table(
            rp_rows,
            (("protocol", "Protocol"), ("run_id", "Run"), ("parent_run_id", "Parent"),
             ("lambda_clad", "lambda"), ("status", "Status"), ("structure_status", "Structure"),
             ("parent_status", "Parent status"), ("action", "Action"),
             ("resume_from", "Resume"), ("reason", "Reason"), ("output_dir", "Output")),
        )
    )
    lines.extend(
        [
            "",
            "## 구현 불변조건",
            "",
            "1. Top-P mask/rank/cutoff는 기존 frozen original-space similarity table로 계산한다.",
            "2. 선택된 pair의 Proto-CLAD cosine만 frozen Gaussian RP → ReLU → L2 normalize 공간에서 계산한다.",
            "3. original-space Proto-CLAD를 RP loss와 동시에 더하지 않는다.",
            "4. classifier, SCC, APR, covariance/tau, classifier adaptation은 parent와 동일하다.",
            "5. R은 N(0,1/d), shape=[768,2048], requires_grad=False이며 seed/shape/checksum을 resolved config와 checkpoint config에 저장한다.",
            "6. nonlinear prototype approximation은 parent cache가 이미 수집한 current features를 재사용하여 task당 한 번 기록한다.",
            "7. baseline 및 RP COMPLETE run은 다시 실행하지 않는다. PARTIAL RP만 동일 checkpoint에서 resume한다.",
            "",
            "## 실행 전 검증 결과 (2026-09-21)",
            "",
            "- projection/loss assertion test: 3/3 PASS",
            "- 네 고정 config validate-only: 4/4 PASS",
            "- RP shape: [768, 2048]",
            "- RP checksum (seed 1993): `849337f236a4df355a5d74c27d015c45a80c27cd4f7fc0e32957a02c7619aa2d`",
            "- 동일 seed 재생성 checksum 일치 및 다른 seed checksum 불일치 확인",
            "- matrix requires_grad=False, ReLU 위치, cosine 전 L2 normalize, original-space Top-P mask 유지 확인",
            "- physical GPU 2 (RTX A6000)에서 [4,768]→[4,2048] forward/backward CUDA sanity PASS",
            "- dry-run에서 허용된 네 run 외 명령이 생성되지 않음을 확인",
            "- 실제 GPU training은 이 검증 과정에서 시작하지 않음",
            "",
            "## 생성 diagnostic",
            "",
            "- rp_pair_geometry_diagnostics.csv: selected class-pair original/RP cosine와 Δcos",
            "- rp_gap_diagnostics.csv: sampled current sample의 correct-new vs selected-old gap",
            "- rp_prototype_approximation.csv: ReLU(mean·R)와 mean(ReLU(z·R)) 차이",
            "- rp_feature_diagnostics.csv: original/RP norm 및 RP sparsity",
            "- rp_similarity_structure_diagnostics.csv: Pearson/Spearman",
            "- rp_gradient_diagnostics.csv 및 rp_loss_diagnostics.csv",
            "- rp_confusion_diagnostics.csv 및 rp_selected_confusion.csv",
            "",
            "## GPU 2 백그라운드 실행",
            "",
            "```bash",
            ("cd ${RESEARCH_WORKSPACE}/structure_rp_protoclad_task5").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
            "nohup bash run_rp_protoclad_task5.sh 2 > logs/rp_task5_gpu2.log 2>&1 &",
            "echo $! > logs/rp_task5_gpu2.pid",
            "```",
            "",
            "실행기는 각 RP run 직전에 parent와 RP output을 다시 audit한다. parent가 아직 완료되지 않았으면 해당 run은 DEFER하고 기존 결과를 건드리지 않는다.",
        ]
    )
    target = ROOT / "rp_experiment_plan.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target, rp_rows


if __name__ == "__main__":
    path, rows = build_plan()
    print(f"wrote {path}")
    for row in rows:
        print(f"{row['action']:13s} {row['protocol']}/{row['run_id']} parent={row['parent_status']}")
