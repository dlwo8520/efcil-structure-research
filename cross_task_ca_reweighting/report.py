"""Fixed success/stop audit and five plots; no hyperparameter selection."""
from __future__ import annotations

import json
from pathlib import Path

from analysis import write_csv

PROTOCOLS = ("IA10", "CAR10")
SEED = 1993


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def _read_summary(root: Path, protocol: str) -> dict:
    return json.loads((root / protocol / "summary.json").read_text(encoding="utf-8"))


def _decision(protocol: str, data: dict) -> dict:
    overall, b1, branch = data["overall"], data["b1"], data["branch"]
    audit = data["audit"]
    control, intervention = branch["control"], branch["intervention"]
    macro_delta = intervention["class_macro_accuracy"] - control["class_macro_accuracy"]
    old_delta = intervention["old_accuracy"] - control["old_accuracy"]
    new_delta = intervention["new_accuracy"] - control["new_accuracy"]
    cross_reduction = control["cross_task_error_n"] - intervention["cross_task_error_n"]
    within_increase = intervention["same_task_error_n"] - control["same_task_error_n"]
    # All thresholds were fixed in implementation_notes.md before observing outcomes.
    checks = {
        "S0_control_parity": (audit["matched_control_vs_original_final_head_weight_max_abs"] <= 1e-3
                              and abs(audit["matched_control_vs_vanilla_cached_accuracy_delta_pp"]) <= 0.5),
        "S1_repair_gt_harm": overall["repair_n"] > overall["harm_n"],
        "S2_seen_accuracy_up": overall["delta_accuracy"] > 0,
        "S3_clear_B1_gain": b1["net_gain_n"] >= 5 and b1["delta_accuracy"] >= 0.05,
        "S4_macro_accuracy_up": macro_delta > 0,
        "S5_cross_reduction_not_offset": cross_reduction > max(0, within_increase),
        "S6_no_large_old_new_loss": old_delta >= -0.01 and new_delta >= -0.01,
    }
    return {
        "protocol": protocol, **{key: "YES" if value else "NO" for key, value in checks.items()},
        "all_satisfied": "YES" if all(checks.values()) else "NO",
        "delta_seen_accuracy_pp": 100 * overall["delta_accuracy"],
        "delta_macro_accuracy_pp": 100 * macro_delta,
        "delta_old_accuracy_pp": 100 * old_delta,
        "delta_new_accuracy_pp": 100 * new_delta,
        "cross_task_error_reduction_n": cross_reduction,
        "same_task_error_increase_n": within_increase,
        "b1_net_gain_n": b1["net_gain_n"],
        "b1_repair_gt_overall_harm": "YES" if b1["repair_n"] > overall["harm_n"] else "NO",
    }


def _make_plots(root: Path, values: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    folder = root / "cross_protocol" / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(PROTOCOLS))
    width = 0.35

    def grouped(filename: str, title: str, ylabel: str, left: list[float], right: list[float],
                left_name: str, right_name: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.bar(positions - width/2, left, width, label=left_name, color="#2a9d8f")
        ax.bar(positions + width/2, right, width, label=right_name, color="#e76f51")
        ax.set_xticks(positions, PROTOCOLS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=180)
        plt.close(fig)

    grouped("01_overall_repair_harm.png", "Overall paired outcomes", "Real test samples (count)",
            [values[p]["overall"]["repair_n"] for p in PROTOCOLS],
            [values[p]["overall"]["harm_n"] for p in PROTOCOLS], "Repair", "Harm")
    grouped("02_b1_repair_harm.png", "Fixed baseline B1 paired outcomes", "B1 samples (count)",
            [values[p]["b1"]["repair_n"] for p in PROTOCOLS],
            [values[p]["b1"]["harm_n"] for p in PROTOCOLS], "Repair", "Harm")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        data = values[protocol]["branch"]
        for offset, name, color in ((-width/2, "control", "#457b9d"),
                                    (width/2, "intervention", "#e76f51")):
            ax.bar(np.arange(2) + offset,
                   [100*data[name]["same_task_error_rate"], 100*data[name]["cross_task_error_rate"]],
                   width, label=name, color=color)
        ax.set_xticks([0, 1], ["Same-task", "Cross-task"])
        ax.set_title(protocol)
        ax.legend()
    axes[0].set_ylabel("Error / all test (%)")
    fig.suptitle("Same-task vs cross-task error")
    fig.tight_layout()
    fig.savefig(folder / "03_error_group.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        data = values[protocol]["branch"]
        for offset, name, color in ((-width/2, "control", "#457b9d"),
                                    (width/2, "intervention", "#e76f51")):
            ax.bar(np.arange(2) + offset,
                   [100*data[name]["accuracy"], 100*data[name]["class_macro_accuracy"]],
                   width, label=name, color=color)
        ax.set_xticks([0, 1], ["Seen", "Class macro"])
        ax.set_title(protocol)
        ax.legend()
    axes[0].set_ylabel("Accuracy (%)")
    fig.suptitle("Matched Control vs Intervention")
    fig.tight_layout()
    fig.savefig(folder / "04_seen_macro.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        data = values[protocol]["branch"]
        for offset, name, color in ((-width/2, "control", "#457b9d"),
                                    (width/2, "intervention", "#e76f51")):
            ax.bar(np.arange(2) + offset,
                   [100*data[name]["old_to_new_rate"], 100*data[name]["new_to_old_rate"]],
                   width, label=name, color=color)
        ax.set_xticks([0, 1], ["Old→New", "New→Old"])
        ax.set_title(protocol)
        ax.legend()
    axes[0].set_ylabel("Directional error rate (%)")
    fig.suptitle("Old/new confusion trade-off")
    fig.tight_layout()
    fig.savefig(folder / "05_old_new_confusion.png", dpi=180)
    plt.close(fig)


def _write_audit(root: Path, manifest: dict, values: dict[str, dict]) -> None:
    lines = ["# Cross-Task CA Reweighting Pilot — audit", "",
             "- 새 backbone 학습, SCC/APR 변경, test 기반 coefficient tuning 없음.",
             "- Control/Intervention은 동일 T10 pre-CA head, 동일 epoch별 Gaussian bank, "
             "동일 label/permutation/minibatch 순서, 동일 SGD·scheduler·5 epochs·256/class를 사용.",
             "- Gaussian bank는 원본 pre-CA RNG 상태에서 생성하며 epoch마다 다시 샘플링. "
             "같은 bank를 두 head에 lockstep으로 공급하고 SHA-256만 저장한다.",
             "- Inference는 모든 seen class에 대한 argmax; task ID와 후보 필터는 사용하지 않음.",
             f"- 실행 physical GPU: `{manifest['physical_gpu']}`", ""]
    for protocol in PROTOCOLS:
        record = manifest["protocols"][protocol]
        result = values[protocol]
        audit = result["audit"]
        lines += [f"## {protocol}", "",
                  f"- Source experiment: `{record['source_experiment_id']}`; seed `{SEED}`",
                  f"- Training-recorded git commit: `{record['source_training_git_commit']}`; "
                  f"source snapshot hash: `{record['source_snapshot_hash_at_training']}`",
                  f"- Task partition: `{record['task_partition']}`",
                  f"- Fixed B1 ID count: `{result['b1_expected_n']}` (expected `{126 if protocol == 'IA10' else 473}`)",
                  f"- CE identity max abs: synthetic `{audit['ce_identity_synthetic_max_abs']:.8g}`, "
                  f"first real batch `{audit['ce_identity_first_real_max_abs']:.8g}`",
                  f"- Optimizer steps: `{audit['optimizer_steps']}`",
                  f"- Original CA task-size argument for Gaussian mean scaling: "
                  f"`{audit['original_ca_mean_scale_task_size_argument']}`; "
                  "loss G(y) follows the true task partition",
                  f"- Cached final-feature vs trajectory baseline prediction mismatches: "
                  f"`{audit['trajectory_baseline_vs_cached_final_head_prediction_mismatch_n']}`",
                  f"- Matched Control − cached vanilla accuracy: "
                  f"`{audit['matched_control_vs_vanilla_cached_accuracy_delta_pp']:+.4f} pp`",
                  f"- Original training T10 top-1: `{audit['original_training_task_summary_correct_n']}` "
                  f"correct, `{100*audit['original_training_task_summary_accuracy']:.4f}%`; "
                  f"cached vanilla − original: `{audit['cached_vanilla_vs_original_training_accuracy_delta_pp']:+.4f} pp`",
                  f"- Matched Control vs original final head max abs weight diff: "
                  f"`{audit['matched_control_vs_original_final_head_weight_max_abs']:.8g}`",
                  "", "| Source | SHA-256 |", "|---|---|"]
        for key, path in record["source_paths"].items():
            lines.append(f"| `{key}`: `{path}` | `{record['source_sha256'][key]}` |")
        lines += ["", "Epoch bank hashes are in `" + protocol + "/bank_manifest.json`.", ""]
    (root / "audit.md").write_text("\n".join(lines), encoding="utf-8")


def create_report(root: Path, manifest: dict) -> None:
    if any(manifest["protocols"][p]["status"] != "complete" for p in PROTOCOLS):
        raise ValueError("both protocols must complete before report generation")
    values = {protocol: _read_summary(root, protocol) for protocol in PROTOCOLS}
    decisions = [_decision(protocol, values[protocol]) for protocol in PROTOCOLS]
    promising = all(row["all_satisfied"] == "YES" for row in decisions)
    folder = root / "cross_protocol"
    folder.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for protocol in PROTOCOLS:
        v = values[protocol]
        c, i = v["branch"]["control"], v["branch"]["intervention"]
        o, b = v["overall"], v["b1"]
        summary_rows.append({
            "protocol": protocol, "n_test": o["n"],
            "control_accuracy": o["control_accuracy"],
            "intervention_accuracy": o["intervention_accuracy"],
            "repair_n": o["repair_n"], "harm_n": o["harm_n"],
            "net_gain_n": o["net_gain_n"], "delta_accuracy": o["delta_accuracy"],
            "control_macro_accuracy": c["class_macro_accuracy"],
            "intervention_macro_accuracy": i["class_macro_accuracy"],
            "b1_n": b["n"], "b1_control_accuracy": b["control_accuracy"],
            "b1_intervention_accuracy": b["intervention_accuracy"],
            "b1_repair_n": b["repair_n"], "b1_harm_n": b["harm_n"],
            "b1_net_gain_n": b["net_gain_n"],
            "control_same_task_error_n": c["same_task_error_n"],
            "intervention_same_task_error_n": i["same_task_error_n"],
            "control_cross_task_error_n": c["cross_task_error_n"],
            "intervention_cross_task_error_n": i["cross_task_error_n"],
            "control_old_accuracy": c["old_accuracy"],
            "intervention_old_accuracy": i["old_accuracy"],
            "control_new_accuracy": c["new_accuracy"],
            "intervention_new_accuracy": i["new_accuracy"],
            "control_old_to_new_rate": c["old_to_new_rate"],
            "intervention_old_to_new_rate": i["old_to_new_rate"],
            "control_new_to_old_rate": c["new_to_old_rate"],
            "intervention_new_to_old_rate": i["new_to_old_rate"],
        })
    write_csv(folder / "summary.csv", summary_rows)
    write_csv(folder / "decision_table.csv", decisions)
    _make_plots(root, values)
    _write_audit(root, manifest, values)
    lines = ["# Cross-Task CA Reweighting Pilot — IA10 / CAR10", "",
             "진단용 CA head-only pilot: vanilla Structure T10 최종 representation을 고정하고 "
             "원래 pre-CA head와 Gaussian replay를 재사용했다. Control은 원래 all-class CE이며, "
             "Intervention은 같은 logits에서 within-task + 2×cross-task loss다. "
             "계수 2는 실행 전 고정했고 sweep/선택은 하지 않았다.", "",
             "```text", f"CROSS-TASK CA PROTOTYPE PROMISING = {'YES' if promising else 'NO'}",
             "FULL METHOD READY = NO", "```", "",
             "위 YES는 두 protocol 모두에서 사전 고정한 Control parity S0 및 S1–S6가 성립할 때만 표시한다. "
             "단일 seed·T10 head-only 결과이므로 YES여도 정식 method 입증은 아니다.", "",
             "## 전체 real test: paired repair / harm", "",
             "| Protocol | N | Original vanilla T10 | Vanilla cached | Matched Control | Intervention | Repair | Harm | Net gain | ΔAcc |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        v = values[protocol]
        o = v["overall"]
        lines.append(f"| {protocol} | {o['n']} | "
                     f"{_pct(v['audit']['original_training_task_summary_accuracy'])} | "
                     f"{_pct(v['branch']['vanilla_cached']['accuracy'])} | "
                     f"{_pct(o['control_accuracy'])} | {_pct(o['intervention_accuracy'])} | "
                     f"{o['repair_n']} | {o['harm_n']} | {o['net_gain_n']:+d} | "
                     f"{_pp(o['delta_accuracy'])} |")
    lines += ["", "## 고정 baseline B1 subset", "",
              "B1 ID는 이전 trajectory diagnostic에서 고정했다. Intervention 결과로 다시 선정하지 않았다.", "",
              "| Protocol | B1 N | Control acc | Intervention acc | Repair | Harm | Net gain | B1 repair > overall harm? |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for protocol, decision in zip(PROTOCOLS, decisions):
        b1 = values[protocol]["b1"]
        lines.append(f"| {protocol} | {b1['n']} | {_pct(b1['control_accuracy'])} | "
                     f"{_pct(b1['intervention_accuracy'])} | {b1['repair_n']} | {b1['harm_n']} | "
                     f"{b1['net_gain_n']:+d} | {decision['b1_repair_gt_overall_harm']} |")
    lines += ["", "## Same-task / Cross-task 오류와 old/new trade-off", "",
              "오류 유형은 true class와 예측 class의 *최초 등장 task*가 같은지로 분류했다. "
              "Old/New은 T10 직전까지의 class와 T10 신규 class를 뜻한다.", "",
              "| Protocol | Branch | Same-task errors | Cross-task errors | Macro acc | Old acc | New acc | Old→New | New→Old |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for name in ("control", "intervention"):
            s = values[protocol]["branch"][name]
            lines.append(f"| {protocol} | {name} | {s['same_task_error_n']} | "
                         f"{s['cross_task_error_n']} | {_pct(s['class_macro_accuracy'])} | "
                         f"{_pct(s['old_accuracy'])} | {_pct(s['new_accuracy'])} | "
                         f"{s['old_to_new_n']} ({_pct(s['old_to_new_rate'])}) | "
                         f"{s['new_to_old_n']} ({_pct(s['new_to_old_rate'])}) |")
    lines += ["", "## 사전 고정 판정 S0–S6", "",
              "S0 Control parity는 원본 최종 head와 weight max abs ≤0.001 및 "
              "cached vanilla 대비 정확도 차이 ≤0.5 pp를 요구한다. "
              "S3 ‘명확한 B1 이득’은 net ≥5개이면서 B1 정확도 ≥5 pp 개선, "
              "S6 ‘큰 old/new 손실 없음’은 어느 쪽도 −1 pp 미만으로 악화하지 않는 것으로 "
              "실행 전에 정의했다. 세부 기준은 [implementation_notes.md](implementation_notes.md).", "",
              "| Protocol | S0 parity | S1 | S2 | S3 | S4 | S5 | S6 | 모두 충족 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in decisions:
        lines.append(f"| {row['protocol']} | {row['S0_control_parity']} | "
                     f"{row['S1_repair_gt_harm']} | "
                     f"{row['S2_seen_accuracy_up']} | {row['S3_clear_B1_gain']} | "
                     f"{row['S4_macro_accuracy_up']} | {row['S5_cross_reduction_not_offset']} | "
                     f"{row['S6_no_large_old_new_loss']} | {row['all_satisfied']} |")
    lines += ["", "조건 하나라도 실패하면 이 **고정 계수 2의 T10 CA reweighting pilot은 유망하지 않다**고 판정하고 "
              "계수 재조정이나 post-hoc candidate 제한을 하지 않는다.", "",
              "## 핵심 질문의 답", ""]
    for protocol, decision in zip(PROTOCOLS, decisions):
        v = values[protocol]
        o, b = v["overall"], v["b1"]
        c, i = v["branch"]["control"], v["branch"]["intervention"]
        cross_delta = i["cross_task_error_n"] - c["cross_task_error_n"]
        same_delta = i["same_task_error_n"] - c["same_task_error_n"]
        lines += [
            f"- {protocol}: 전체 정확도 변화 {_pp(o['delta_accuracy'])}; repair {o['repair_n']} vs "
            f"harm {o['harm_n']}. B1 복원 net {b['net_gain_n']:+d}/{b['n']}개; "
            f"B1 repair {b['repair_n']}개와 전체 harm {o['harm_n']}개. "
            f"Cross-task 오류 변화 {cross_delta:+d}개, same-task 오류 변화 {same_delta:+d}개. "
            f"Macro 변화 {_pp(i['class_macro_accuracy']-c['class_macro_accuracy'])}; "
            f"old/new 변화 {_pp(i['old_accuracy']-c['old_accuracy'])} / "
            f"{_pp(i['new_accuracy']-c['new_accuracy'])}.",
        ]
    lines += ["", "두 protocol의 방향이 같은지와 full incremental/independent seed 확대 여부는 "
              f"위 S0–S6의 공동 결과에 따라 판단한다: **{'제한적 후속 검증 후보' if promising else '이 고정 pilot은 확대하지 않음'}**. "
              "어느 경우에도 현재 단계에서 full method는 준비되지 않았다.", "",
              "## Control parity와 해석 한계", ""]
    for protocol in PROTOCOLS:
        a = values[protocol]["audit"]
        lines.append(f"- {protocol}: trajectory baseline과 최종 feature cache 기반 vanilla head의 "
                     f"예측 불일치 {a['trajectory_baseline_vs_cached_final_head_prediction_mismatch_n']}개; "
                     f"matched Control−cached vanilla 정확도 "
                     f"{a['matched_control_vs_vanilla_cached_accuracy_delta_pp']:+.4f} pp; "
                     f"head weight 최대 절댓값 차이 "
                     f"{a['matched_control_vs_original_final_head_weight_max_abs']:.8g}.")
    lines += ["", "B1의 birth-candidate restriction은 subset을 정의하는 과거 oracle 분석일 뿐, "
              "이번 inference에는 사용하지 않았다. B1 개선이 전체 accuracy 개선을 보장하지 않으며, "
              "Control이 vanilla와 다르면 핵심 인과 비교는 동일 bank를 공유한 Control vs Intervention이다. "
              "이번 prototype 자체를 새로운 논문 method로 부르지 않는다. "
              "LUCIR, SS-IL, CCIL/Essentials for Class Incremental Learning 등 기존 intra/inter-task "
              "학습 연구와의 중복 검토, independent seed/class order, full incremental run은 별도 후속 단계다.", "",
              "## Plots", "",
              "![Overall repair/harm](cross_protocol/plots/01_overall_repair_harm.png)", "",
              "![B1 repair/harm](cross_protocol/plots/02_b1_repair_harm.png)", "",
              "![Error group](cross_protocol/plots/03_error_group.png)", "",
              "![Seen/macro accuracy](cross_protocol/plots/04_seen_macro.png)", "",
              "![Old/new confusion](cross_protocol/plots/05_old_new_confusion.png)", "",
              "Reproducibility: [audit.md](audit.md), [run_manifest.json](run_manifest.json), "
              "branch별 sample CSV, [cross-protocol summary](cross_protocol/summary.csv), "
              "[decision table](cross_protocol/decision_table.csv).", ""]
    (root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
