"""Render the predeclared diagnostic summaries; never train or tune a model."""
from __future__ import annotations

import csv
from pathlib import Path

from metrics import write_csv

PROTOCOLS = ("IA10", "CAR10")
SCOPES = ("all_final_classes", "retention_birth_le9")
TYPE_LABELS = {"A": "Persistent wrong", "B": "Birth correct → final wrong",
               "C": "Recovered → final wrong"}
TARGETS = {
    "A": "current-task acquisition / initial class learning",
    "B1": "all-seen candidate competition (후발 class 경쟁)",
    "B2": "existing-class discrimination preservation (표현 및 head 변화 분리 전)",
    "C": "dynamic competition / ranking instability",
    "MIXED": "혼합 양상; 단일 개입 위치를 지정하지 않음",
}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty diagnostic table: {path}")
    return rows


def _one(rows: list[dict], scope: str, grouping: str, group_id: str) -> dict:
    matches = [row for row in rows if row["scope"] == scope and
               row["grouping"] == grouping and row["group_id"] == group_id]
    if len(matches) != 1:
        raise ValueError(f"expected one summary for {scope}/{grouping}/{group_id}; found {len(matches)}")
    return matches[0]


def _float(value: str) -> float | None:
    return None if value in ("", "None", None) else float(value)


def _pct(value: str | float | None) -> str:
    value = _float(value) if isinstance(value, str) else value
    return "N/A" if value is None else f"{100 * value:.2f}%"


def _pp(value: str | float | None) -> str:
    value = _float(value) if isinstance(value, str) else value
    return "N/A" if value is None else f"{value:.2f} pp"


def _cell(row: dict, letter: str) -> str:
    return (f"{row['n_' + letter]} / {_pct(row[letter + '_final_error_share'])} / "
            f"{_pp(row[letter + '_all_test_burden_pp'])}")


def _mode(comp: dict, restriction: dict) -> str:
    shares = {letter: _float(comp[f"{letter}_final_error_share"]) for letter in "ABC"}
    if all(value is None for value in shares.values()):
        return "MIXED"
    largest = max(value for value in shares.values() if value is not None)
    leaders = [letter for letter, value in shares.items() if value is not None and abs(value - largest) < 1e-12]
    if len(leaders) != 1:
        return "MIXED"
    if leaders[0] != "B":
        return leaders[0]
    b1, b2 = int(restriction["restored_b1_n"]), int(restriction["still_wrong_b2_n"])
    return "B1" if b1 > b2 else "B2" if b2 > b1 else "MIXED"


def _plots(root: Path, summary: dict[str, dict[str, dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = root / "cross_protocol" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = list(PROTOCOLS)
    positions = np.arange(2)
    colors = {"A": "#457b9d", "B": "#e76f51", "C": "#8d99ae"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(2)
    for letter in "ABC":
        values = np.asarray([100 * float(summary[name]["all_final_classes"]["comp"][f"{letter}_final_error_share"])
                             for name in names])
        ax.bar(positions, values, bottom=bottom, label=TYPE_LABELS[letter], color=colors[letter])
        bottom += values
    ax.set_xticks(positions, names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of final wrong (%)")
    ax.set_title("Final-error trajectory composition · all final classes")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "01_final_error_composition.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    width = 0.34
    for offset, key, label, color in ((-width / 2, "loss_rate", "Loss L", "#e76f51"),
                                      (width / 2, "recovery_rate", "Recovery R", "#2a9d8f")):
        values = [100 * float(summary[name]["retention_birth_le9"]["lr"][key]) for name in names]
        ax.bar(positions + offset, values, width, label=label, color=color)
    ax.set_xticks(positions, names)
    ax.set_ylabel("Rate (%) · different denominators")
    ax.set_title("Birth-to-final loss and recovery · birth ≤ T9")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_loss_vs_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bottom = np.zeros(2)
    for key, label, color in (("restored_b1_n", "B1: restored", "#2a9d8f"),
                              ("still_wrong_b2_n", "B2: still wrong", "#e76f51")):
        values = np.asarray([int(summary[name]["retention_birth_le9"]["br"][key]) for name in names])
        ax.bar(positions, values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(positions, names)
    ax.set_ylabel("Type B test samples (count)")
    ax.set_title("T10 logits restricted to birth candidate classes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "03_typeB_birth_candidate_restriction.png", dpi=180)
    plt.close(fig)


def create_report(root: Path, manifest: dict) -> None:
    """Make a single comparison only after both vanilla baseline sequences complete."""
    if any(manifest["protocols"][name]["status"] != "complete" for name in PROTOCOLS):
        raise ValueError("both protocols must have completed inference")
    output = root / "cross_protocol"
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    summary: dict[str, dict[str, dict]] = {}
    summary_rows = []
    decision_rows = []
    for protocol in PROTOCOLS:
        folder = root / protocol
        comp = _read(folder / "final_error_composition.csv")
        lr = _read(folder / "loss_recovery.csv")
        br = _read(folder / "typeB_candidate_restriction.csv")
        tables[protocol] = {"comp": comp, "lr": lr, "br": br}
        summary[protocol] = {}
        for scope in SCOPES:
            for grouping in ("micro", "class_macro"):
                c = _one(comp, scope, grouping, "all")
                l = _one(lr, scope, grouping, "all")
                b = _one(br, scope, grouping, "all")
                if grouping == "micro":
                    if sum(int(c[f"n_{letter}"]) for letter in "ABC") != int(c["n_final_wrong"]):
                        raise AssertionError(f"{protocol} {scope}: A+B+C != final wrong")
                    if int(b["restored_b1_n"]) + int(b["still_wrong_b2_n"]) != int(b["type_b_n"]):
                        raise AssertionError(f"{protocol} {scope}: B1+B2 != B")
                    summary[protocol][scope] = {"comp": c, "lr": l, "br": b}
                summary_rows.append({
                    "protocol": protocol, "scope": scope, "grouping": grouping,
                    "n_test": c["n_test"], "n_final_wrong": c["n_final_wrong"],
                    "n_A": c["n_A"], "A_final_error_share": c["A_final_error_share"],
                    "A_all_test_burden_pp": c["A_all_test_burden_pp"],
                    "n_B": c["n_B"], "B_final_error_share": c["B_final_error_share"],
                    "B_all_test_burden_pp": c["B_all_test_burden_pp"],
                    "n_C": c["n_C"], "C_final_error_share": c["C_final_error_share"],
                    "C_all_test_burden_pp": c["C_all_test_burden_pp"],
                    "birth_correct_n": l["birth_correct_n"], "lost_at_final_n": l["lost_at_final_n"],
                    "loss_rate": l["loss_rate"], "birth_wrong_n": l["birth_wrong_n"],
                    "recovered_at_final_n": l["recovered_at_final_n"], "recovery_rate": l["recovery_rate"],
                    "restored_b1_n": b["restored_b1_n"], "still_wrong_b2_n": b["still_wrong_b2_n"],
                    "restored_b1_share": b["restored_b1_share"],
                    "restored_b1_all_test_burden_pp": b["restored_b1_all_test_burden_pp"],
                    "still_wrong_b2_all_test_burden_pp": b["still_wrong_b2_all_test_burden_pp"],
                })
        retention = summary[protocol]["retention_birth_le9"]
        mode = _mode(retention["comp"], retention["br"])
        decision_rows.append({
            "protocol": protocol, "dominant_mode_retention": mode,
            "dominant_A_share": retention["comp"]["A_final_error_share"],
            "dominant_B_share": retention["comp"]["B_final_error_share"],
            "dominant_C_share": retention["comp"]["C_final_error_share"],
            "B1_share_within_B": retention["br"]["restored_b1_share"],
            "B2_share_within_B": retention["br"]["still_wrong_b2_share"],
            "loss_rate": retention["lr"]["loss_rate"],
            "recovery_rate": retention["lr"]["recovery_rate"],
            "intervention_target": TARGETS[mode], "method_ready": "NO",
        })
    write_csv(output / "summary.csv", summary_rows)
    write_csv(output / "decision_table.csv", decision_rows)
    _plots(root, summary)

    modes = {row["protocol"]: row["dominant_mode_retention"] for row in decision_rows}
    common = modes["IA10"] == modes["CAR10"] and modes["IA10"] != "MIXED"
    target = TARGETS[modes["IA10"]] if common else (
        f"공통 단일 target 없음; IA10={TARGETS[modes['IA10']]}, CAR10={TARGETS[modes['CAR10']]}"
    )
    parity_notes = []
    for protocol in PROTOCOLS:
        changed = [(entry["task"], entry["historical_correct_delta"])
                   for entry in manifest["protocols"][protocol]["checkpoints"]
                   if entry["historical_correct_delta"]]
        parity_notes.append(f"{protocol}: " + (
            ", ".join(f"T{task} {delta:+d}" for task, delta in changed) if changed else "전 task 일치"
        ))
    lines = [
        "# Final Error Trajectory Diagnostic — IA10 / CAR10",
        "", "진단 전용: seed 1993의 vanilla Structure task-end checkpoint T1~T10을 고정하여 "
        "official test sample의 birth 이후 예측만 추적했다. 학습·보정·튜닝은 하지 않았다.", "",
        "## 판정 요약", "",
        f"DOMINANT IA10 FAILURE MODE = {modes['IA10']}",
        f"DOMINANT CAR10 FAILURE MODE = {modes['CAR10']}",
        f"COMMON FAILURE MODE = {'YES' if common else 'NO'}",
        f"NEXT INTERVENTION TARGET = {target}",
        "METHOD READY = NO", "",
        "지배 유형은 retention subset (birth ≤ T9)의 최종 오류 A/B/C 최대 비율로 정했다. "
        "B가 최대이면 B1/B2의 다수 쪽으로 나눴으며 동률은 MIXED다. "
        "이는 탐색 우선순위일 뿐 인과 확정이나 방법론 승인 근거가 아니다.", "",
        "## Table A — 최종 오류 구성", "",
        "각 칸: count / final errors 중 비율 / 전체 test 중 burden (pp).", "",
        "| Protocol | Scope | A 지속 오답 | B birth 정답→final 오답 | C 중간 회복→final 오답 |",
        "|---|---|---:|---:|---:|",
    ]
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            c = summary[protocol][scope]["comp"]
            lines.append(f"| {protocol} | {scope} | {_cell(c, 'A')} | {_cell(c, 'B')} | {_cell(c, 'C')} |")
    lines += ["", "## Table B — Birth→Final loss / recovery", "",
              "L과 R의 분모가 다르므로 크기만 비교해 최종 accuracy 변화로 해석하지 않는다.", "",
              "| Protocol | Scope | Birth correct N | Lost N | L | Birth wrong N | Recovered N | R |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            l = summary[protocol][scope]["lr"]
            lines.append(f"| {protocol} | {scope} | {l['birth_correct_n']} | {l['lost_at_final_n']} | "
                         f"{_pct(l['loss_rate'])} | {l['birth_wrong_n']} | {l['recovered_at_final_n']} | "
                         f"{_pct(l['recovery_rate'])} |")
    lines += ["", "## Table C — Type B의 birth candidate restriction", "",
              "T10 표현과 head/logit은 고정하고 birth 때 존재하던 후보 class만 남겼다.", "",
              "| Protocol | Scope | Type B N | B1 복원 N (%) | B2 여전히 오답 N (%) | B1 / all test | B2 / all test |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for scope in SCOPES:
            b = summary[protocol][scope]["br"]
            lines.append(f"| {protocol} | {scope} | {b['type_b_n']} | "
                         f"{b['restored_b1_n']} ({_pct(b['restored_b1_share'])}) | "
                         f"{b['still_wrong_b2_n']} ({_pct(b['still_wrong_b2_share'])}) | "
                         f"{_pp(b['restored_b1_all_test_burden_pp'])} | "
                         f"{_pp(b['still_wrong_b2_all_test_burden_pp'])} |")
    lines += ["", "## Class-macro 및 birth-task 분해", "",
              "Class-macro는 분모가 정의되는 class의 비율을 평균한다. "
              "최종 오답이 없는 class는 오류 구성 비율 평균에서 제외하지만 all-test burden 평균에는 포함한다. "
              "상세 class-macro 값은 각 protocol CSV와 [cross-protocol summary](cross_protocol/summary.csv)에 있다. "
              "아래 birth-task 행은 micro 기준이며 T10-born의 L=0은 이후 task가 없기 때문이다.", ""]
    for protocol in PROTOCOLS:
        lines += [f"### {protocol}", "",
                  "| Birth | N test | Final wrong | A/B/C | L | R | B1/B2 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for task in range(1, 11):
            c = _one(tables[protocol]["comp"], "all_final_classes", "birth_task_micro", f"T{task}")
            l = _one(tables[protocol]["lr"], "all_final_classes", "birth_task_micro", f"T{task}")
            b = _one(tables[protocol]["br"], "all_final_classes", "birth_task_micro", f"T{task}")
            lines.append(f"| T{task} | {c['n_test']} | {c['n_final_wrong']} | "
                         f"{c['n_A']}/{c['n_B']}/{c['n_C']} | {_pct(l['loss_rate'])} | "
                         f"{_pct(l['recovery_rate'])} | {b['restored_b1_n']}/{b['still_wrong_b2_n']} |")
        lines.append("")
    lines += ["## 시각화", "",
              "![Final-error composition](cross_protocol/plots/01_final_error_composition.png)", "",
              "![Loss and recovery](cross_protocol/plots/02_loss_vs_recovery.png)", "",
              "![Type B candidate restriction](cross_protocol/plots/03_typeB_birth_candidate_restriction.png)", "",
              "## 해석의 경계", "",
              "A는 관측한 task-end checkpoint에서 계속 오답이었다는 뜻이지 학습 중 어느 시점에도 정답이 아니었다는 뜻은 아니다. "
              "B1은 후발 class competitor가 최종 오류에 필요했다는 증거이지 class 추가만이 원인이라는 증거는 아니다. "
              "B2는 기존 class 간 최종 ranking도 틀렸다는 뜻이며 backbone 표현과 classifier head 변화는 분리하지 못한다. "
              "C는 checkpoint 사이 예측 상태가 비단조적임을 뜻한다. "
              "T10-born class의 최종 오류는 retention 실패가 아니므로 전체와 birth≤T9 결과를 분리했다.", "",
              "## 재현성", "",
              "[audit.md](audit.md)와 [run_manifest.json](run_manifest.json)에 체크포인트 SHA-256, "
              "training branch/seed, task·class 순서, transform, 후보 class, canonical sample ID digest 및 샘플 수를 기록했다. "
              "추론은 원본 checkpoint의 frozen network와 원본 test transform으로 수행했다. "
              "이전 feature cache는 logit source로 사용하지 않았다.", ""]
    lines += ["원본 task-summary의 평가는 `after_task()` 직전이고 본 진단은 저장된 "
              "`task_end` checkpoint(`after_task()` 직후)를 평가한다. Correct-count 차이(진단 − 원본): "
              + "; ".join(parity_notes) + ". 0이 아니면 해당 task의 비교 해석에 주의한다.", ""]
    (root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
