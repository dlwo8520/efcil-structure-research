"""Saved predictions only: trajectory metrics, exact repair/harm, five plots."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .common import (ROOT, BASELINE, BRANCHES, read_csv, read_json, sha256, spec,
                     train_dir, write_csv, write_json)


def accuracy(rows, field="correct"):
    return 100 * sum(int(r[field]) for r in rows) / len(rows) if rows else None


def task_metric(rows, branch, task):
    current = [r for r in rows if int(r["birth_task"]) == task]
    old = [r for r in rows if int(r["birth_task"]) < task]
    labels = sorted({int(r["y"]) for r in rows})
    macro = sum(accuracy([r for r in rows if int(r["y"]) == y]) for y in labels) / len(labels)
    local = accuracy(current, "local_correct") if all(r.get("local_correct", "") != "" for r in current) else None
    return {"branch": branch, "task": task, "N_seen": len(rows), "seen_accuracy_pct": accuracy(rows),
            "macro_accuracy_pct": macro, "new_accuracy_pct": accuracy(current),
            "old_accuracy_pct": accuracy(old), "current_local_accuracy_pct": local,
            "N_old": len(old), "N_new": len(current), "evaluation_stage": "task_end_post_after_task"}


def paired(left, right):
    l = {r["sample_id"]: r for r in left}
    r = {r["sample_id"]: r for r in right}
    assert len(l) == len(left) and len(r) == len(right) and l.keys() == r.keys()
    repair = harm = 0
    for identity, a in l.items():
        b = r[identity]
        assert int(a["y"]) == int(b["y"]) and int(a["birth_task"]) == int(b["birth_task"])
        repair += int(a["correct"]) == 0 and int(b["correct"]) == 1
        harm += int(a["correct"]) == 1 and int(b["correct"]) == 0
    delta = 100 * (repair - harm) / len(left) if left else None
    if left:
        assert abs(delta - (accuracy(right) - accuracy(left))) < 1e-9
    return {"N": len(left), "repair": repair, "harm": harm, "net": repair - harm, "delta_pp": delta}


def vanilla_rows(task):
    path = Path(spec()["reference_trajectory"])
    rows = []
    for row in read_csv(path):
        if int(row["birth_task"]) <= task:
            pred = row[f"pred_T{task:02d}"]
            if pred == "":
                raise ValueError("existing vanilla trajectory incomplete; no silent retraining")
            y = int(row["true_label"])
            rows.append({"branch": "Vanilla", "task": task, "sample_id": row["sample_id"], "y": y,
                         "birth_task": int(row["birth_task"]), "pred": int(pred), "correct": int(int(pred) == y),
                         "local_correct": ""})
    return rows


def analyze():
    metrics, cohorts, pairs, all_predictions = [], [], [], []
    predictions = {}
    for branch in ("Vanilla", *BRANCHES):
        predictions[branch] = {}
        for task in range(1, 11):
            path = train_dir(branch) / f"predictions_T{task:02d}.csv" if branch != "Vanilla" else None
            if path is not None and not path.exists():
                continue
            rows = vanilla_rows(task) if branch == "Vanilla" else read_csv(path)
            assert len({r["sample_id"] for r in rows}) == len(rows)
            predictions[branch][task] = rows
            metrics.append(task_metric(rows, branch, task))
            all_predictions.extend(rows)
            for birth in range(1, task + 1):
                subset = [r for r in rows if int(r["birth_task"]) == birth]
                cohorts.append({"branch": branch, "birth_task": birth, "task": task, "N": len(subset),
                                "accuracy_pct": accuracy(subset)})
    common_tasks = sorted(set(predictions["S_full"]) & set(predictions["BOB_full"]))
    for task in common_tasks:
        left, right = predictions["S_full"][task], predictions["BOB_full"][task]
        reference = {r["sample_id"]: r for r in predictions["Vanilla"][task]}
        assert {r["sample_id"] for r in left} == reference.keys()
        for row in left:
            assert int(row["y"]) == int(reference[row["sample_id"]]["y"])
        for scope in ("whole_seen", "old", "current"):
            select = lambda row: scope == "whole_seen" or ((int(row["birth_task"]) < task) if scope == "old" else int(row["birth_task"]) == task)
            pairs.append({"task": task, "scope": scope, "comparison": "BOB_full-S_full",
                          **paired([r for r in left if select(r)], [r for r in right if select(r)])})
    write_csv(ROOT / "per_task_metrics.csv", metrics)
    write_csv(ROOT / "per_sample_predictions.csv", all_predictions)
    write_csv(ROOT / "cohort_retention.csv", cohorts)
    if pairs:
        write_csv(ROOT / "repair_harm_by_task.csv", pairs)
    generation = []
    resources = []
    for task in range(1, 11):
        path = ROOT / "bob_tasks" / f"T{task:02d}/generation_manifest.csv"
        if path.exists():
            generation.extend(read_csv(path))
        for path in (ROOT / "bob_tasks" / f"T{task:02d}").glob("resource_*.json"):
            resources.append(read_json(path))
    if generation:
        write_csv(ROOT / "generation_manifest.csv", generation)
    attempts = read_json(ROOT / "process_attempts.json") if (ROOT / "process_attempts.json").exists() else []
    write_json(ROOT / "resource_usage.json", {"process_attempts": attempts, "generator_completed_stage_resources": resources,
               "structure_task_resources": {b: [read_json(p) for p in sorted(train_dir(b).glob("resource_T*.json"))] for b in BRANCHES},
               "external_generation_seconds_all_recorded_attempts": sum(r["seconds"] for r in attempts if r["stage"] in ("caption", "lora", "generate")),
               "structure_training_process_seconds_all_recorded_attempts": {b: sum(r["seconds"] for r in attempts if r.get("branch") == b) for b in BRANCHES},
               "candidate_generated_count": len(generation), "accepted_generated_count": sum(int(r["accepted"]) for r in generation),
               "generated_disk_bytes": sum(Path(r["generated_path"]).stat().st_size for r in generation),
               "generator_artifact_disk_bytes": sum(p.stat().st_size for p in (ROOT / "bob_tasks").rglob("*") if p.is_file() and not p.is_symlink()),
               "generation_GPU": sorted({r.get("physical_gpu") for r in attempts if r["stage"] in ("caption", "lora", "generate")}),
               "repo_commit": spec()["bob"]["repo_commit"], "base_model": spec()["bob"]["base_model_id"],
               "note": "process timings include startup and failed recorded attempts; task resource timers exclude earlier interrupted epochs"})
    make_plots(metrics, cohorts, pairs, common_tasks)
    write_report(metrics, cohorts, pairs, common_tasks)
    return common_tasks == list(range(1, 11))


def make_plots(metrics, cohorts, pairs, tasks):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = ROOT / "figures"
    root.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    def save(name):
        plt.tight_layout()
        plt.savefig(root / (name + ".png"), dpi=180, bbox_inches="tight")
        plt.savefig(root / (name + ".svg"), bbox_inches="tight")
        plt.close()
    plt.figure(figsize=(8, 4))
    for branch in ("Vanilla", *BRANCHES):
        own = [r for r in metrics if r["branch"] == branch]
        if own:
            plt.plot([r["task"] for r in own], [r["seen_accuracy_pct"] for r in own], marker="o", label=branch)
    plt.xlabel("Task"); plt.ylabel("Seen accuracy (%)"); plt.xticks(range(1, 11)); plt.legend()
    save("seen_accuracy_trajectory")
    if not tasks:
        return
    lookup = {(r["branch"], r["task"]): r for r in metrics}
    for key, name, label in (("new_accuracy_pct", "new_delta_bob_minus_s", "New"),
                             ("seen_accuracy_pct", "seen_delta_bob_minus_s", "Seen")):
        delta = [lookup["BOB_full", t][key] - lookup["S_full", t][key] for t in tasks]
        plt.figure(figsize=(8, 3.6)); plt.bar(tasks, delta, color="#38789b")
        plt.axhline(0, color="black", linewidth=.8); plt.xticks(range(1, 11))
        plt.xlabel("Task"); plt.ylabel(f"BOB − S {label} (pp)")
        for t, d in zip(tasks, delta): plt.annotate(f"{d:+.2f}", (t, d), xytext=(0, 4 if d >= 0 else -14), textcoords="offset points", ha="center", fontsize=8)
        save(name)
    final = [r for r in pairs if r["task"] == 10]
    if final:
        plt.figure(figsize=(7, 4)); x = np.arange(len(final))
        plt.bar(x - .18, [r["repair"] for r in final], .36, label="Repair")
        plt.bar(x + .18, [r["harm"] for r in final], .36, label="Harm")
        plt.xticks(x, [r["scope"] for r in final]); plt.ylabel("T10 paired count"); plt.legend()
        for i, row in enumerate(final):
            for shift, field in ((-.18, "repair"), (.18, "harm")):
                plt.text(i + shift, row[field] + .3, str(row[field]), ha="center")
        save("final_repair_harm")
    matrix = np.full((10, 10), np.nan)
    values = {(r["branch"], r["birth_task"], r["task"]): r["accuracy_pct"] for r in cohorts}
    for t in tasks:
        for b in range(1, t + 1):
            matrix[b - 1, t - 1] = values["BOB_full", b, t] - values["S_full", b, t]
    limit = max(float(np.nanmax(np.abs(matrix))), .01)
    plt.figure(figsize=(9, 7)); plt.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit)
    plt.xticks(range(10), range(1, 11)); plt.yticks(range(10), range(1, 11))
    plt.xlabel("Evaluation task"); plt.ylabel("Birth cohort"); plt.colorbar(label="BOB − S cohort accuracy (pp)")
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value): plt.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=8)
    save("cohort_retention_delta_heatmap")


def write_report(metrics, cohorts, pairs, tasks):
    complete = tasks == list(range(1, 11))
    lookup = {(r["branch"], r["task"]): r for r in metrics}
    lines = ["# IA10 Full-Trajectory BOB vs Strong Augmentation", "",
             "상태: " + ("COMPLETE" if complete else f"PARTIAL — paired task {tasks}; 최종 판정 보류"), "",
             "두 branch는 독립 T1→T10 누적 trajectory다. 기존 vanilla는 재학습하지 않는다. 모든 저장 mean/cov는 current real train only이며 old synthetic replay는 없다.", "",
             "실제 acquisition은 current-only CosFace: 0.5×real + 0.5×aux + 5×SCC(real). 원 APR/Gaussian CA/optimizer 설정을 유지한다. Class order와 split은 vanilla manifest와 대조한다.", "",
             "$$A_{inc}=\\frac{1}{10}\\sum_{t=1}^{10}A_t^{seen},\\qquad \\Delta Acc_{pp}=100(Repair-Harm)/N.$$"]
    if not complete:
        lines += ["", "모든 조건이 완료되기 전 평균 incremental/final success를 만들지 않는다."]
    else:
        averages = {b: sum(lookup[b, t]["seen_accuracy_pct"] for t in range(1, 11)) / 10 for b in ("Vanilla", *BRANCHES)}
        delta = lookup["BOB_full", 10]["seen_accuracy_pct"] - lookup["S_full", 10]["seen_accuracy_pct"]
        inc_delta = averages["BOB_full"] - averages["S_full"]
        wins = sum(lookup["BOB_full", t]["new_accuracy_pct"] > lookup["S_full", t]["new_accuracy_pct"] for t in range(1, 11))
        old_delta = lookup["BOB_full", 10]["old_accuracy_pct"] - lookup["S_full", 10]["old_accuracy_pct"]
        mean_old_delta = sum(lookup["BOB_full", t]["old_accuracy_pct"] - lookup["S_full", t]["old_accuracy_pct"] for t in range(2, 11)) / 9
        final = next(r for r in pairs if r["task"] == 10 and r["scope"] == "whole_seen")
        positive = delta > 0 and inc_delta > 0 and wins >= 6 and final["repair"] > final["harm"]
        verdict = "BETTER" if positive and old_delta >= 0 and mean_old_delta >= 0 else ("WORSE" if delta <= 0 and inc_delta <= 0 else "MIXED")
        direction = "CONTINUE" if verdict == "BETTER" else ("STOP" if verdict == "WORSE" and wins <= 5 else "REVISE")
        lines += ["", "## 핵심 비교", "", "| Metric | Vanilla existing | S-full | BOB-full | BOB−S |", "|---|---:|---:|---:|---:|",
                  f"| Final seen (%) | {lookup['Vanilla',10]['seen_accuracy_pct']:.4f} | {lookup['S_full',10]['seen_accuracy_pct']:.4f} | {lookup['BOB_full',10]['seen_accuracy_pct']:.4f} | {delta:+.4f} pp |",
                  f"| Average incremental (%) | {averages['Vanilla']:.4f} | {averages['S_full']:.4f} | {averages['BOB_full']:.4f} | {inc_delta:+.4f} pp |",
                  f"| Final old (%) | {lookup['Vanilla',10]['old_accuracy_pct']:.4f} | {lookup['S_full',10]['old_accuracy_pct']:.4f} | {lookup['BOB_full',10]['old_accuracy_pct']:.4f} | {old_delta:+.4f} pp |", "",
                  f"New 승리 task는 {wins}/10이며, final whole seen repair/harm/net은 {final['repair']}/{final['harm']}/{final['net']:+d}다. Final old delta {old_delta:+.4f} pp, T2..T10 평균 old delta {mean_old_delta:+.4f} pp를 함께 본다.", "",
                  "큰 old 희생의 수치 threshold는 명세에 없으므로 결과를 본 뒤 새로 만들지 않는다. BETTER 자동 표시는 위 4가지 숫자 기준을 만족하고 final/평균 old 모두 감소하지 않은 보수적 경우에만 사용한다. 그 외 old trade-off는 MIXED로 남긴다. 단일 model seed이며 작은 +0.1~0.2 pp는 강한 우위가 아니다.", "",
                  "## Cohort 및 task별 해석", ""]
        for t in range(1, 11):
            dn = lookup["BOB_full", t]["new_accuracy_pct"] - lookup["S_full", t]["new_accuracy_pct"]
            ds = lookup["BOB_full", t]["seen_accuracy_pct"] - lookup["S_full", t]["seen_accuracy_pct"]
            local_delta = lookup["BOB_full", t]["current_local_accuracy_pct"] - lookup["S_full", t]["current_local_accuracy_pct"]
            lines.append(f"- T{t}: New {dn:+.4f} pp; Seen {ds:+.4f} pp; current local {local_delta:+.4f} pp (diagnostic only).")
        lines += ["", "Final cohort 차이:", ""]
        values = {(r["branch"], r["birth_task"], r["task"]): r["accuracy_pct"] for r in cohorts}
        for birth in range(1, 11):
            lines.append(f"- Birth T{birth}: T10 BOB−S {values['BOB_full',birth,10]-values['S_full',birth,10]:+.4f} pp")
        resource = read_json(ROOT / "resource_usage.json")
        lines += ["", "## 비용", "", f"기록된 generation/caption/LoRA process 시간 총 {resource['external_generation_seconds_all_recorded_attempts']:.1f} s. S/BOB Structure process 시간: {resource['structure_training_process_seconds_all_recorded_attempts']}. 생성 이미지 {resource['candidate_generated_count']}장, accepted {resource['accepted_generated_count']}장, generator artifact disk {resource['generator_artifact_disk_bytes']} bytes. 세부 task별 peak memory/시간은 resource_usage.json 참조.", "",
                  "## 최종 판정", "", "```text", "DATASET = ImageNet-A", "PROTOCOL = 200 classes / 10 tasks / 20 classes per task", "SEED = 1993", "VANILLA = existing reference", "PRIMARY_COMPARISON = BOB-full vs S-full",
                  f"BOB_FINAL_SEEN_DELTA_PP = {delta:+.6f}", f"BOB_AVG_INCREMENTAL_DELTA_PP = {inc_delta:+.6f}", f"BOB_NEW_WIN_TASKS = {wins} / 10",
                  f"FINAL_REPAIR = {final['repair']}", f"FINAL_HARM = {final['harm']}", f"FINAL_NET = {final['net']}",
                  f"BOB_FULL_TRAJECTORY = {verdict}", f"GENERATION_RESEARCH_DIRECTION = {direction}",
                  "MAIN_OBSERVATION = matched full trajectory and cohort deltas above; no test-driven selection",
                  "NEXT_STEP = independent seeds if promising; otherwise revise/stop this fixed recipe, not all generation", "```"]
    lines += ["", "## 그림", ""]
    for name in ("seen_accuracy_trajectory", "new_delta_bob_minus_s", "seen_delta_bob_minus_s", "final_repair_harm", "cohort_retention_delta_heatmap"):
        if (ROOT / "figures" / (name + ".png")).exists():
            lines += [f"![{name}](figures/{name}.png)", ""]
    lines += ["## 비교와 한계", "",
              "Vanilla는 저장 task-end 예측을 재사용하며 학습하지 않는다. 원 task_summary(평가 before after_task)와 저장 checkpoint의 post-after_task 진단은 IA10 최종 한 sample 차이가 알려져 있어 합치지 않는다. 이번 reference와 S/BOB는 post-after_task에 맞춘다.", "",
              "이전 SD img2img T1/T9 spot pilot의 수치를 BOB 효과로 합산하지 않는다. 이번은 text-to-image context-conditioned LoRA와 매 task 누적 acquisition이므로 generator와 intervention 위치가 모두 다르다. Full trajectory와 spot의 효과 차이는 위치와 recipe가 confounded된 비교다.", "",
              "BOB bank는 acquisition에만 쓰고 저장 moments는 real only다. 따라서 branch 차이는 generated-memory를 추가한 효과가 아니다. 다만 바뀐 representation의 real moments/APR/head가 다음 task로 이어지는 경로는 존재한다.", "",
              "기술 QC만 수행했다. 사람 검수, test-confidence filtering, Type A selector는 없으며 label fidelity를 보장하지 않는다. Qwen context의 class-agnostic prompting이 실제 class 정보 제거를 보증하지도 않는다. ImageNet-A는 BOB 공식 few-shot benchmark가 아니며 object descriptor/current-task bank 범위는 EF-CIL adaptation이다.", "",
              "원본 subtype 분석은 primary 판단에 사용하지 않는다. 한 model seed이며 독립 replicate/통계적 유의성은 없다. 세부 절차·고정 설정은 [BOB method notes](bob_method_notes.md), [experiment spec](experiment_spec.json), [resource usage](resource_usage.json)에 있다."]
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
