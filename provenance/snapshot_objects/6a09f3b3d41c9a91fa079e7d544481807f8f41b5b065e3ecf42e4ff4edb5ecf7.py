#!/usr/bin/env python3
"""Build the planned similarity-aware Markdown report from completed runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("B0", "B1", "S0", "S1", "S2", "S3", "S4")
COLORS = {
    "B0": "#222222",
    "B1": "#4c78a8",
    "S0": "#8c8c8c",
    "S1": "#59a14f",
    "S2": "#f28e2b",
    "S3": "#e15759",
    "S4": "#7b6fd0",
}


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument(
        "--report-dir", default=str(ROOT / "reports" / "similarity_aware_followup")
    )
    return parser.parse_args()


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return math.nan, math.nan
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else math.nan


def fmt(mean, sd, signed=False):
    prefix = "+" if signed and mean > 0 else ""
    if math.isnan(sd):
        return f"{prefix}{mean:.2f} (SD N/A)"
    return f"{prefix}{mean:.2f} ± {sd:.2f}"


def save(fig, report_dir, name):
    fig.savefig(report_dir / "figures" / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(report_dir / "figures" / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def common_metrics(dataset_root):
    rows = read_csv(dataset_root / "metrics_per_seed.csv")
    by_method = defaultdict(dict)
    for row in rows:
        if row.get("status") == "complete":
            by_method[row["method"]][int(row["seed"])] = row
    missing = [method for method in METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"completed methods missing: {missing}")
    seeds = sorted(set.intersection(*(set(by_method[method]) for method in METHODS)))
    if not seeds:
        raise RuntimeError("no common completed seed across B0/B1/S0-S4")
    return by_method, seeds


def main_table(by_method, seeds):
    keys = ("Inc-Acc", "Last-Acc", "Forgetting", "Old", "New")
    rows = []
    for method in METHODS:
        row = {"Method": method, "n": len(seeds)}
        for key in keys:
            values = [float(by_method[method][seed][key]) for seed in seeds]
            mean, sd = mean_sd(values)
            deltas = [
                float(by_method[method][seed][key]) - float(by_method["B0"][seed][key])
                for seed in seeds
            ]
            delta_mean, delta_sd = mean_sd(deltas)
            row[key] = fmt(mean, sd)
            row["Delta_" + key] = fmt(delta_mean, delta_sd, signed=True)
            row["_" + key] = mean
            row["_Delta_" + key] = delta_mean
        rows.append(row)
    return rows


def stability_plot(rows, report_dir):
    fig, axis = plt.subplots(figsize=(7, 6), layout="constrained")
    for row in rows[1:]:
        method = row["Method"]
        axis.scatter(
            row["_Delta_Old"], row["_Delta_New"], s=70, color=COLORS[method], label=method
        )
        axis.annotate(method, (row["_Delta_Old"], row["_Delta_New"]), xytext=(5, 5), textcoords="offset points")
    axis.axhline(0, color="#777777", linewidth=1)
    axis.axvline(0, color="#777777", linewidth=1)
    axis.set(xlabel="Δ Old accuracy (pp)", ylabel="Δ New accuracy (pp)", title="Stability–Plasticity vs B0")
    axis.grid(alpha=0.2)
    save(fig, report_dir, "stability_plasticity")


def run_rows(dataset_root, method, filename, seeds):
    rows = []
    for seed in seeds:
        for row in read_csv(dataset_root / method / f"seed_{seed}" / filename):
            row = dict(row)
            row["seed"] = int(row.get("seed", seed))
            rows.append(row)
    return rows


def s0_group_analysis(dataset_root, seeds, report_dir):
    pairs = run_rows(dataset_root, "S0", "pair_similarity.csv", seeds)
    forgetting = run_rows(dataset_root, "S0", "class_forgetting.csv", seeds)
    confusion = run_rows(dataset_root, "S0", "cross_task_confusion.csv", seeds)
    forget_lookup = {
        (int(row["seed"]), int(row["task"]), int(row["old_class_id"])): row
        for row in forgetting
    }
    old_to_new = defaultdict(float)
    new_to_old = defaultdict(float)
    for row in confusion:
        seed, task = int(row["seed"]), int(row["task"])
        true_class, pred_class = int(row["true_class"]), int(row["pred_class"])
        rate = float(row["normalized_rate"])
        if row["true_is_old_or_new"] == "old" and row["pred_is_old_or_new"] == "new":
            old_to_new[(seed, task, true_class)] += rate
        if row["true_is_old_or_new"] == "new" and row["pred_is_old_or_new"] == "old":
            new_to_old[(seed, task, pred_class)] += rate

    by_old = defaultdict(list)
    top1, top5 = set(), set()
    top_pair_examples = []
    for row in pairs:
        key = (int(row["seed"]), int(row["task"]), int(row["old_class_id"]))
        by_old[key].append(float(row["similarity"]))
        if row["is_top1"].lower() == "true":
            top1.add(key)
            top_pair_examples.append(row)
        if row["is_top5"].lower() == "true":
            top5.add(key)
    records = []
    for key, values in by_old.items():
        if key not in forget_lookup:
            continue
        frow = forget_lookup[key]
        records.append(
            {
                "seed": key[0],
                "task": key[1],
                "old_class_id": key[2],
                "max_similarity": max(values),
                "old_accuracy_drop": float(frow["forgetting_increment"]),
                "forgetting_increment": float(frow["forgetting_increment"]),
                "old_to_new_confusion": old_to_new[key],
                "new_to_old_confusion": new_to_old[key],
                "is_top1": key in top1,
                "is_top5": key in top5,
            }
        )
    similarities = np.asarray([row["max_similarity"] for row in records])
    q25, q75 = np.quantile(similarities, (0.25, 0.75))
    groups = {
        "Low": lambda row: row["max_similarity"] <= q25,
        "Mid": lambda row: q25 < row["max_similarity"] < q75,
        "High": lambda row: row["max_similarity"] >= q75,
        "Top1": lambda row: row["is_top1"],
        "Top5": lambda row: row["is_top5"],
    }
    summary = []
    for name, predicate in groups.items():
        selected = [row for row in records if predicate(row)]
        summary.append(
            {
                "Similarity group": name,
                "n": len(selected),
                "old accuracy drop": np.mean([row["old_accuracy_drop"] for row in selected]),
                "forgetting": np.mean([row["forgetting_increment"] for row in selected]),
                "old->new confusion": np.mean([row["old_to_new_confusion"] for row in selected]),
                "new->old confusion": np.mean([row["new_to_old_confusion"] for row in selected]),
            }
        )
    write_csv(report_dir / "tables" / "s0_similarity_groups.csv", summary)
    write_csv(report_dir / "tables" / "s0_old_class_records.csv", records)
    top_pair_examples.sort(key=lambda row: float(row["similarity"]), reverse=True)
    write_csv(report_dir / "tables" / "s0_top1_pair_examples.csv", top_pair_examples[:50])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    x = [row["max_similarity"] for row in records]
    axes[0].scatter(x, [row["old_accuracy_drop"] for row in records], s=10, alpha=0.35)
    axes[0].set(xlabel="max old-new similarity", ylabel="old accuracy drop (pp)", title="S0 similarity vs accuracy drop")
    axes[1].scatter(x, [row["old_to_new_confusion"] for row in records], s=10, alpha=0.35)
    axes[1].set(xlabel="max old-new similarity", ylabel="old→new confusion rate", title="S0 similarity vs confusion")
    for axis in axes:
        axis.grid(alpha=0.2)
    save(fig, report_dir, "s0_similarity_relationship")
    return summary, records


def diagnostic_plot(dataset_root, seeds, report_dir, method, filename, xkey, ykey, name, title):
    rows = run_rows(dataset_root, method, filename, seeds)
    points = []
    for row in rows:
        try:
            points.append((float(row[xkey]), float(row[ykey])))
        except (KeyError, TypeError, ValueError):
            continue
    if not points:
        return 0
    if len(points) > 30000:
        indices = np.linspace(0, len(points) - 1, 30000, dtype=int)
        points = [points[index] for index in indices]
    fig, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    axis.scatter([x for x, _ in points], [y for _, y in points], s=8, alpha=0.25, color=COLORS[method])
    axis.set(xlabel=xkey, ylabel=ykey, title=title)
    axis.grid(alpha=0.2)
    save(fig, report_dir, name)
    return len(rows)


def feature_gradient_plots(dataset_root, seeds, report_dir):
    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    for method in METHODS:
        rows = run_rows(dataset_root, method, "feature_norm.csv", seeds)
        final = defaultdict(list)
        for row in rows:
            final[int(row["task"])].append((int(row["epoch"]), float(row["mean"])))
        tasks, values = [], []
        for task, candidates in sorted(final.items()):
            max_epoch = max(epoch for epoch, _ in candidates)
            values_at_end = [value for epoch, value in candidates if epoch == max_epoch]
            tasks.append(task)
            values.append(float(np.mean(values_at_end)))
        axis.plot(tasks, values, marker="o", label=method, color=COLORS[method])
    axis.set(xlabel="Task", ylabel="Feature norm mean", title="Final-epoch feature norm")
    axis.grid(alpha=0.2)
    axis.legend(ncol=4)
    save(fig, report_dir, "feature_norm")

    gradient_rows = []
    for method in METHODS:
        rows = run_rows(dataset_root, method, "gradient_diagnostics.csv", seeds)
        for key in ("cos_cls_scc", "cos_cls_abs", "cos_cls_rel"):
            values = [float(row[key]) for row in rows if row.get(key) not in (None, "", "None")]
            if values:
                gradient_rows.append({"method": method, "component": key, "mean_cosine": np.mean(values)})
    write_csv(report_dir / "tables" / "gradient_cosine_summary.csv", gradient_rows)


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines)


def main():
    cli = args_parser()
    dataset_root = Path(cli.output_root).expanduser().resolve() / "imageneta"
    report_dir = Path(cli.report_dir).expanduser().resolve()
    (report_dir / "figures").mkdir(parents=True, exist_ok=True)
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    by_method, seeds = common_metrics(dataset_root)
    performance = main_table(by_method, seeds)
    write_csv(
        report_dir / "tables" / "main_performance.csv",
        [{key: value for key, value in row.items() if not key.startswith("_")} for row in performance],
    )
    stability_plot(performance, report_dir)
    s0_summary, s0_records = s0_group_analysis(dataset_root, seeds, report_dir)
    counts = {
        "S1": diagnostic_plot(dataset_root, seeds, report_dir, "S1", "similarity_tau_diagnostics.csv", "similarity", "similarity_scale", "s1_similarity_tau", "S1 similarity vs tau scale"),
        "S2": diagnostic_plot(dataset_root, seeds, report_dir, "S2", "relative_margin_diagnostics.csv", "class_similarity", "cosine_gap", "s2_similarity_gap", "S2 similarity vs cosine gap"),
        "S3": diagnostic_plot(dataset_root, seeds, report_dir, "S3", "scc_similarity_diagnostics.csv", "max_old_similarity", "scc_weight", "s3_similarity_scc", "S3 similarity vs SCC weight"),
        "S4": diagnostic_plot(dataset_root, seeds, report_dir, "S4", "soft_gate_diagnostics.csv", "similarity", "gate_weight", "s4_similarity_gate", "S4 similarity vs soft gate"),
    }
    feature_gradient_plots(dataset_root, seeds, report_dir)

    headers = ("Method", "n", "Inc-Acc", "Delta_Inc-Acc", "Last-Acc", "Delta_Last-Acc", "Forgetting", "Delta_Forgetting", "Old", "Delta_Old", "New", "Delta_New")
    perf_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in performance]
    s0_display = []
    for row in s0_summary:
        s0_display.append(
            {
                "Similarity group": row["Similarity group"],
                "n": row["n"],
                "old accuracy drop": f"{row['old accuracy drop']:.3f}",
                "forgetting": f"{row['forgetting']:.3f}",
                "old->new confusion": f"{row['old->new confusion']:.4f}",
                "new->old confusion": f"{row['new->old confusion']:.4f}",
            }
        )
    seed_text = ", ".join(map(str, seeds))
    report = f"""# Structure Similarity-Aware 후속 실험 보고서

공통 완료 seed: {seed_text} (n={len(seeds)}). 정확도와 forgetting은 %, 차이는 paired percentage point다. n=1이면 SD는 N/A이며 관측 pilot 결과로만 해석한다.

## 1. Main Performance

{markdown_table(headers, perf_rows)}

![stability-plasticity](figures/stability_plasticity.png)

오른쪽 위가 old/new를 동시에 개선한 영역이다. 점의 위치는 공통 seed에서 계산한 paired delta 평균이며 유의성을 뜻하지 않는다.

## 2. S0: Similar old-new가 진짜 더 문제인가?

{markdown_table(('Similarity group', 'n', 'old accuracy drop', 'forgetting', 'old->new confusion', 'new->old confusion'), s0_display)}

![S0 relationship](figures/s0_similarity_relationship.png)

각 old class와 현재 new classes의 최대 pre-training cosine을 사용했다. 상관은 원인으로 해석하지 않는다. 원자료는 [s0_old_class_records.csv](tables/s0_old_class_records.csv), top-1 사례는 [s0_top1_pair_examples.csv](tables/s0_top1_pair_examples.csv)다.

## 3. S1: Similarity-aware tau

![S1 tau](figures/s1_similarity_tau.png)

S1 진단 row 수는 {counts['S1']:,}개다. similarity가 높을수록 tau scale이 낮아지는지와 B1 대비 New/Old/Forgetting을 함께 판단한다.

## 4. S2: Relative margin

![S2 gap](figures/s2_similarity_gap.png)

S2 진단 row 수는 {counts['S2']:,}개다. 높은 similarity에서 cosine gap과 margin violation, new→hard-old confusion을 함께 확인한다.

## 5. S3: Similarity-aware SCC

![S3 SCC](figures/s3_similarity_scc.png)

S3 진단 row 수는 {counts['S3']:,}개다. SCC 완화로 New가 회복되더라도 Old/Forgetting이 크게 악화되면 성공으로 보지 않는다.

## 6. S4: Soft hybrid

![S4 gate](figures/s4_similarity_gate.png)

S4 진단 row 수는 {counts['S4']:,}개다. hard threshold S2 대비 trade-off와 seed 방향 반복을 확인한다.

## 7. Stability-Plasticity

위 stability-plasticity 그림의 x는 ΔOld, y는 ΔNew다. Inc/Last/Forgetting과 함께 판단하며 단일 headline metric만으로 성공을 선언하지 않는다.

## 8. Feature norm / Gradient

![feature norm](figures/feature_norm.png)

E3처럼 norm inflation이 재발하는지 확인한다. gradient cosine 요약은 [gradient_cosine_summary.csv](tables/gradient_cosine_summary.csv)에 저장했다.

## 9. 의미 있는 class pair 사례

S0의 top-1 pair와 method별 `relative_margin_diagnostics.csv`, `cross_task_confusion.csv`, `exports/class_accuracy.csv`를 class ID로 결합할 수 있다. 최소 5개 사례 선정은 공통 3-seed에서 개선/악화 방향이 반복되는 pair만 대상으로 한다.

## 10. 연구 질문 결론

- RQ1 Similar pair가 더 충돌하는가: S0 similarity group의 forgetting/confusion 차이로 판단한다.
- RQ2 similarity-aware tau가 new 손실을 줄이는가: S1 vs B1의 paired New와 Old/F를 함께 본다.
- RQ3 relative margin이 나은가: S2 vs B1/S1의 gap, confusion, main metric을 본다.
- RQ4 SCC 완화가 plasticity를 회복하는가: S3 vs S2 paired New/Old/F와 gradient를 본다.
- RQ5 soft hybrid가 안정적인가: S4 vs S1/S2의 seed별 paired delta 분산을 본다.

각 결론은 저장된 공통 seed 수가 3보다 작으면 pilot 관측으로 제한한다.

## 재생성

```bash
source ${{RESEARCH_WORKSPACE}}/miniconda3/etc/profile.d/conda.sh
conda activate cds
python {Path(__file__).resolve()} \\
  --output-root {Path(cli.output_root).expanduser().resolve()} \\
  --report-dir {report_dir}
```
"""
    (report_dir / "report.md").write_text(report, encoding="utf-8")
    print(report_dir / "report.md")


if __name__ == "__main__":
    main()

