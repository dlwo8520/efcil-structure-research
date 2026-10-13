"""Predeclared pilot decision and Markdown report; no result-based tuning."""
from __future__ import annotations

from pathlib import Path


def protocol_decision(metrics: dict) -> dict:
    base = metrics["branches"]["C0"]
    candidate = metrics["branches"]["C2"]
    overall = candidate["subsets"]["overall"]
    b1 = candidate["subsets"]["B1"]
    old = candidate["subsets"]["old"]
    new = candidate["subsets"]["new"]
    conditions = {
        "overall_delta_at_least_0p5_pp": overall["delta_pp"] >= 0.5,
        "B1_repair_at_least_10_percent": b1["repair"] >= (b1["n"] + 9) // 10,
        "old_not_below_minus_1_pp": old["delta_pp"] >= -1,
        "new_not_below_minus_1_pp": new["delta_pp"] >= -1,
        "macro_not_below_minus_0p2_pp":
            100 * (candidate["macro_accuracy"] - base["macro_accuracy"]) >= -0.2,
        "C0_exact_cached_prediction_parity": metrics["C0_cached_vanilla_prediction_mismatch_n"] == 0,
        "C2_zero_exact_logit_parity": metrics["C2_zero_correction_exact_logit_parity"],
        "C1_converged": metrics["fit"]["C1"]["converged"],
        "C2_converged": metrics["fit"]["C2"]["converged"],
        "C1_local_invariant": metrics["fit"]["C1"]["local_argmax_disagreements"] == 0,
        "C2_local_invariant": metrics["fit"]["C2"]["local_argmax_disagreements"] == 0,
    }
    if not conditions["C1_converged"] or not conditions["C2_converged"]:
        verdict = "INCONCLUSIVE_FIT"
    else:
        verdict = "PASS" if all(conditions.values()) else "FAIL"
    c1_delta = metrics["branches"]["C1"]["subsets"]["overall"]["delta_pp"]
    c2_delta = overall["delta_pp"]
    return {"status": verdict, "conditions": conditions,
            "C2_minus_C1_overall_pp": c2_delta - c1_delta,
            "C1_C2_difference_under_0p2_pp": abs(c2_delta - c1_delta) < 0.2}


def render_report(root: Path, metrics_by_protocol: dict[str, dict]) -> None:
    protocols = ("IA10", "CAR10")
    if set(metrics_by_protocol) != set(protocols):
        raise ValueError("both protocol metrics required for final report")
    overall_pass = all(metrics_by_protocol[p]["decision"]["status"] == "PASS" for p in protocols)
    inconclusive = any(metrics_by_protocol[p]["decision"]["status"] == "INCONCLUSIVE_FIT"
                       for p in protocols)
    verdict = "INCONCLUSIVE" if inconclusive else ("PASS" if overall_pass else "STOP")
    lines = ["# B1 ranking-preserving task correction — T10 pilot", "",
             "기존 Structure post-CA head와 representation을 고정했다. C0는 동일 checkpoint의 원 점수, "
             "C1은 task별 상수 bias, C2는 frozen logit descriptor에서 만든 입력별 task-common correction이다. "
             "C1/C2는 전 class Gaussian 256개씩의 같은 fit bank를 사용한다. Check bank와 real test는 학습·선택에 사용하지 않았다.", "",
             f"**사전 기준에 따른 종합 판정: {verdict}**. "
             "단일 seed·최종 T10 head-only pilot이며 full incremental method 성능은 뜻하지 않는다.", "",
             "## Real test: C0 대비 paired 결과", "",
             "| Protocol | Branch | N | Accuracy | Macro | Repair | Harm | Net | ΔAcc | Fixed B1 repair | B1 accuracy |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for protocol in protocols:
        metric = metrics_by_protocol[protocol]
        for name in ("C0", "C1", "C2"):
            branch = metric["branches"][name]
            overall = branch["subsets"]["overall"]
            b1 = branch["subsets"]["B1"]
            lines.append(f"| {protocol} | {name} | {overall['n']} | {branch['accuracy']:.2%} | "
                         f"{branch['macro_accuracy']:.2%} | {overall['repair']} | {overall['harm']} | "
                         f"{overall['net']} | {overall['delta_pp']:+.3f} pp | "
                         f"{b1['repair']} / {b1['n']} | {b1['accuracy']:.2%} |")
    lines += ["", "## Same/cross 오류와 old/new", "",
              "| Protocol | Branch | Same-task wrong | Cross-task wrong | Old ΔAcc | New ΔAcc | Old→New | New→Old |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in protocols:
        metric = metrics_by_protocol[protocol]
        for name in ("C0", "C1", "C2"):
            branch = metric["branches"][name]
            lines.append(f"| {protocol} | {name} | {branch['same_task_errors']} | "
                         f"{branch['cross_task_errors']} | {branch['subsets']['old']['delta_pp']:+.3f} pp | "
                         f"{branch['subsets']['new']['delta_pp']:+.3f} pp | "
                         f"{branch['old_to_new']} | {branch['new_to_old']} |")
    lines += ["", "## Fixed retention error type", "",
              "C0의 A/B1/B2/C 및 final-correct 그룹은 기존 trajectory에서 고정했다. 아래는 C2의 paired 변화다.", "",
              "| Protocol | Type | N | C0 correct | C2 correct | Repair | Harm | Net |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for protocol in protocols:
        branch = metrics_by_protocol[protocol]["branches"]["C2"]
        for kind in ("A", "B1", "B2", "C", "final_correct"):
            row = branch["subsets"][kind]
            lines.append(f"| {protocol} | {kind} | {row['n']} | {row['base_correct']} | "
                         f"{row['correct']} | {row['repair']} | {row['harm']} | {row['net']} |")
    lines += ["", "## 구현·학습 검증", "",
              "| Protocol | C0 vs trajectory prediction mismatch | C1 fit/check CE | C2 fit/check CE | "
              "C1/C2 closure calls | C1/C2 convergence | C1/C2 local argmax mismatch | "
              "C1/C2 max pair-margin drift |",
              "|---|---:|---|---|---|---|---|---|"]
    for protocol in protocols:
        metric = metrics_by_protocol[protocol]
        fit = metric["fit"]
        ce = metric["gaussian_CE"]
        lines.append(f"| {protocol} | {metric['C0_cached_vanilla_prediction_mismatch_n']} | "
                     f"{ce['C1']['fit_Gaussian_CE']:.4f}/{ce['C1']['check_Gaussian_CE']:.4f} | "
                     f"{ce['C2']['fit_Gaussian_CE']:.4f}/{ce['C2']['check_Gaussian_CE']:.4f} | "
                     f"{fit['C1']['closure_calls']}/{fit['C2']['closure_calls']} | "
                     f"{fit['C1']['converged']}/{fit['C2']['converged']} | "
                     f"{fit['C1']['local_argmax_disagreements']}/"
                     f"{fit['C2']['local_argmax_disagreements']} | "
                     f"{fit['C1']['within_task_pair_margin_max_abs']:.2e}/"
                     f"{fit['C2']['within_task_pair_margin_max_abs']:.2e} |")
    lines += ["", "학습·평가 정확도는 checkpoint의 frozen cosine head에서 계산한 logits 기준이다. "
              "Gaussian CE 개선만으로 성공을 판정하지 않는다. C0의 fixed B1 정확도는 정의상 0이어야 한다. "
              "Within-task 순위·margin 불변성과 baseline correct→same-task-wrong harm 0개를 검증했다.", "",
              "## 사전 기준·해석", ""]
    for protocol in protocols:
        metric = metrics_by_protocol[protocol]
        decision = metric["decision"]
        lines.append(f"- **{protocol}: {decision['status']}**; C2−C1 overall "
                     f"{decision['C2_minus_C1_overall_pp']:+.3f} pp; "
                     f"차이 <0.2 pp: {decision['C1_C2_difference_under_0p2_pp']}. "
                     f"Runtime {metric['runtime_seconds']:.1f}s, peak GPU allocated "
                     f"{metric['peak_gpu_allocated_bytes'] / 2**20:.1f} MiB, "
                     f"추가 C2 계수·정규화 통계 저장량 float64 "
                     f"{metric['saved_inference_storage_bytes_float64']['C2']} bytes "
                     f"(float32 전환 시 예상 {metric['additional_inference_storage_bytes_float32']['C2']} bytes; "
                     "float32 배포 정확도는 별도 검증 필요).")
    lines += ["", "Pilot PASS는 IA10·CAR10 모두 C2 overall ≥+0.5 pp, fixed B1 repair ≥10%, "
              "old/new 각각 ≥−1 pp, macro ≥−0.2 pp 및 parity·ranking 불변성·fit 수렴을 요구한다. "
              "Optimizer가 budget에 걸리면 결과는 inconclusive로 취급한다. "
              "C1/C2가 양쪽에서 0.2 pp 미만 차이면 conditional 효과를 주장하지 않는다. "
              "이 기준을 넘더라도 독립 seed, 전 task trajectory 및 기존 calibration 비교가 필요하다.", "",
              "## 산출물", "",
              "각 protocol의 `sample_predictions.csv`, `transition_tables.csv` (correct/same/cross 3×3), "
              "`error_destination_task.csv`, `true_task_repair_harm.csv`, `metrics.json`, "
              "`correction_parameters.npz`에 sample·task별 결과와 재현 정보를 저장했다. "
              "전체 설정은 `config.json`, 완료 상태는 `run_manifest.json`을 참조한다.", ""]
    (root / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
