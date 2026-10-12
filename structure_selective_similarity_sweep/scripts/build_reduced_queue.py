#!/usr/bin/env python3
"""Audit existing outputs and generate the reduced Stage-1 plan and queue."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import argparse
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from selective.reduced import DEFAULT_OUTPUT_ROOT, audit_all


def value(item):
    return "—" if item is None else str(item)


def cell(item):
    return str(item or "—").replace("|", "\\|").replace("\n", " ")


def table(records):
    lines = [
        "| protocol | run_id | method | P | alpha | lambda | current_status | action | resume_from | reason | command | output_dir |",
        "|---|---|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            "| " + " | ".join(
                cell(item)
                for item in (
                    row.protocol, row.run_id, row.method, value(row.top_p), value(row.alpha),
                    value(row.lambda_clad), row.current_status, row.action, row.resume_from,
                    row.reason, f"`{row.command}`" if row.command else "—", row.output_dir,
                )
            ) + " |"
        )
    return lines


def old_sweep_processes():
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="], text=True, capture_output=True, check=True
    )
    return [
        line.strip() for line in result.stdout.splitlines()
        if "structure_selective_similarity_sweep/scripts/run_sweeps.py" in line
        or "structure_selective_similarity_sweep/run.py" in line
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, choices=(1993,), default=1993)
    parser.add_argument("--plan", default=str(ROOT / "reduced_experiment_plan.md"))
    parser.add_argument("--queue", default=str(ROOT / "reduced_experiment_queue.md"))
    cli = parser.parse_args()

    records = audit_all(Path(cli.output_root), cli.seed)
    selected = [row for row in records if row.selected]
    actions = Counter(row.action for row in records)
    statuses = Counter(row.current_status for row in records if row.legacy_grid_run)
    selected_actions = Counter(row.action for row in selected)
    skipped_pending = sum(
        row.action == "SKIP"
        and row.current_status == "NOT_STARTED"
        and row.legacy_grid_run
        for row in records
    )
    skipped_breakdown = Counter(
        row.current_status for row in records if row.action == "SKIP"
    )
    processes = old_sweep_processes()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    selected_by_protocol = []
    for protocol in ("IA5", "CAR5"):
        selected_by_protocol.extend(row for row in selected if row.protocol == protocol)
    execution = [row for row in selected_by_protocol if row.action in {"RUN_NEW", "RESUME"}]

    plan_lines = [
        "# Reduced Stage 1 실행 계획",
        "",
        f"Filesystem audit: {timestamp}  ",
        f"Seed: `{cli.seed}`  ",
        "운영 범위: **IA5 + CAR5 only**  ",
        "Primary baseline: **vanilla Structure**",
        "",
        "## 결론",
        "",
        f"- 이미 완료된 선택 run: **{selected_actions['KEEP_COMPLETE']}개**",
        f"- 새로 실행할 run: **{selected_actions['RUN_NEW']}개**",
        f"- checkpoint에서 resume할 run: **{selected_actions['RESUME']}개**",
        f"- 기존 108-grid 중 미실행 상태로 SKIP할 run: **{skipped_pending}개**",
        f"- 전체 audit row action: KEEP_COMPLETE {actions['KEEP_COMPLETE']}, "
        f"RUN_NEW {actions['RUN_NEW']}, RESUME {actions['RESUME']}, SKIP {actions['SKIP']}",
        "",
        "실제 새 GPU 작업은 CAR5의 `Structure`, `B_P020_A100`, `C_P020_L030`, "
        "`C_P020_L100` 네 개다. IA5 선택 run은 모두 완료되어 다시 실행하지 않는다.",
        "",
        "## 기존 대규모 queue 중단 확인",
        "",
    ]
    if processes:
        plan_lines.append("다음 기존 sweep 프로세스가 아직 발견되므로 reduced queue를 실행하면 안 된다:")
        plan_lines.extend(f"- `{line}`" for line in processes)
    else:
        plan_lines.append("기존 `run_sweeps.py` 및 sweep `run.py` 프로세스는 발견되지 않았다.")
    plan_lines.extend(
        [
            "",
            "중단 시 실행 중이던 비선택 `CAR5/B_P010_A100`은 삭제하지 않았다. "
            "`checkpoint_latest.pt`와 partial metadata를 그대로 보존했으며 새 queue에서는 resume하지 않는다.",
            "",
            "## Audit 판정 기준",
            "",
            "COMPLETE는 `summary.json`, `run_metadata.json`, `resolved_config.json`, "
            "`task_summary.csv`, `checkpoint_latest.pt`가 모두 완결되고 예상 task 수와 모든 "
            "`task_final` 행이 일치해야 한다. `metrics_per_seed.csv`도 교차 확인하되, 학습 직후 "
            "aggregate export만 실패한 경우 완료 run을 재학습하지 않도록 보조 증거로 사용한다. "
            "IA5 Structure는 외부의 검증된 `B0_STRUCTURE` 결과다.",
            "",
            f"기존 108-grid 상태: COMPLETE {statuses['COMPLETE']}, PARTIAL {statuses['PARTIAL']}, "
            f"FAILED {statuses['FAILED']}, NOT_STARTED {statuses['NOT_STARTED']}.",
            "",
            "## 선택 matrix 상태",
            "",
            *table(selected_by_protocol),
            "",
            "## 실행 순서",
            "",
        ]
    )
    if execution:
        for index, row in enumerate(execution, 1):
            plan_lines.append(f"{index}. `{row.protocol}/{row.run_id}` — `{row.action}`")
    else:
        plan_lines.append("실행할 항목 없음.")
    plan_lines.extend(
        [
            "",
            "IA10은 완료/실패/partial 결과를 모두 참고용으로 보존하지만 추가 실행, 실패 복구, "
            "matched Structure 생성을 하지 않는다. CAR10은 전체 HOLD다.",
            "",
            "## 진단 수집 유지",
            "",
            "실험 개수만 줄이고 각 선택 run은 기존 `run.py`와 동일한 recorder/export 경로를 "
            "사용한다. 따라서 performance/task/class accuracy, similarity selection/rank/cutoff, "
            "APR drift/tau, separation D2/intrusion/loss, Proto-CLAD cosine/loss, gradient cosine/norm, "
            "feature norm, old-new 및 full confusion matrix, wall-time/GPU/RAM 진단을 계속 저장한다. "
            "Structure에서는 비활성 방법의 loss 항목만 해당 없음으로 남는다.",
            "",
            "## 실행 방법",
            "",
            "GPU 번호를 첫 번째 인자로 지정한다.",
            "",
            "```bash",
            ("cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
            "bash run_reduced_experiments.sh 1",
            "```",
            "",
            "백그라운드 실행:",
            "",
            "```bash",
            ("cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
            "nohup bash run_reduced_experiments.sh 1 > logs/reduced_stage1_gpu1.log 2>&1 &",
            "echo $! > logs/reduced_stage1_gpu1.pid",
            "```",
            "",
            "이 plan 생성 단계에서는 새 학습을 시작하지 않았다. PositiveDefinite fallback도 아직 "
            "적용하지 않았으며, 선택 run에서 실제 오류가 발생할 때만 적용하고 횟수를 기록한다.",
            "",
            "## SKIP 보존 현황",
            "",
            f"- COMPLETE 보존: {skipped_breakdown['COMPLETE']}",
            f"- PARTIAL 보존: {skipped_breakdown['PARTIAL']}",
            f"- FAILED 보존: {skipped_breakdown['FAILED']}",
            f"- NOT_STARTED/HOLD: {skipped_breakdown['NOT_STARTED']}",
        ]
    )

    queue_lines = [
        "# Reduced Stage 1 queue",
        "",
        f"Generated from filesystem audit: {timestamp}  ",
        "실행 protocol: `IA5`, `CAR5`  ",
        "HOLD protocol: `IA10`, `CAR10`  ",
        "실행 대상은 `RUN_NEW` 또는 `RESUME` row뿐이다. 기존 결과 디렉터리는 삭제하거나 덮어쓰지 않는다.",
        "",
        "## 선택 matrix",
        "",
        *table(selected_by_protocol),
        "",
        "## 전체 filesystem audit",
        "",
        "아래에는 기존 108-grid와 protocol별 Structure row를 모두 기록한다. `SKIP`은 파일 삭제가 아니라 "
        "향후 실행 queue에서 제외한다는 뜻이다.",
        "",
        *table(records),
    ]

    Path(cli.plan).write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
    Path(cli.queue).write_text("\n".join(queue_lines) + "\n", encoding="utf-8")
    print(
        f"selected_complete={selected_actions['KEEP_COMPLETE']} "
        f"run_new={selected_actions['RUN_NEW']} resume={selected_actions['RESUME']} "
        f"skip_existing_pending={skipped_pending} total_skip={actions['SKIP']}"
    )
    for row in execution:
        print(f"{row.action:8s} {row.protocol}/{row.run_id} -> {row.output_dir}")
    print(f"wrote {Path(cli.plan).resolve()}")
    print(f"wrote {Path(cli.queue).resolve()}")


if __name__ == "__main__":
    main()
