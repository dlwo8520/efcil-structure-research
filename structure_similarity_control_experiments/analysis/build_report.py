#!/usr/bin/env python3
"""Build the Additional Control report after completed C0--C4 runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("C0", "C1", "C2", "C3", "C4")
LABELS = {
    "C0": "AM-Drift",
    "C1": "High-Off",
    "C2": "Low-Off",
    "C3": "Random-Off",
    "C4": "High-Weak",
}
COLORS = {"C0": "#333333", "C1": "#e15759", "C2": "#4e79a7", "C3": "#9c755f", "C4": "#59a14f"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--report-dir", default=str(ROOT / "reports" / "additional_control"))
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
    array = np.asarray(values, dtype=float)
    if not array.size:
        return math.nan, math.nan
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else math.nan


def fmt(values):
    mean, sd = mean_sd(values)
    return f"{mean:.2f} (SD N/A)" if math.isnan(sd) else f"{mean:.2f} ± {sd:.2f}"


def save(fig, report_dir, name):
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def completed_metrics(dataset_root):
    rows = read_csv(dataset_root / "metrics_per_seed.csv")
    lookup = defaultdict(dict)
    for row in rows:
        if row.get("status") == "complete":
            lookup[row["method"]][int(row["seed"])] = row
    missing = [method for method in METHODS if not lookup[method]]
    if missing:
        raise RuntimeError(f"completed control methods missing: {missing}")
    seeds = sorted(set.intersection(*(set(lookup[method]) for method in METHODS)))
    if not seeds:
        raise RuntimeError("C0--C4 have no common completed seed")
    return lookup, seeds


def run_rows(dataset_root, method, filename, seeds):
    rows = []
    for seed in seeds:
        for row in read_csv(dataset_root / method / f"seed_{seed}" / filename):
            item = dict(row)
            item["seed"] = int(item.get("seed", seed))
            rows.append(item)
    return rows


def main_table(lookup, seeds, dataset_root):
    rows = []
    for method in METHODS:
        row = {"Method": f"{method} {LABELS[method]}", "n": len(seeds)}
        for metric in ("Inc-Acc", "Last-Acc", "Forgetting", "Old", "New"):
            values = [float(lookup[method][seed][metric]) for seed in seeds]
            deltas = [
                float(lookup[method][seed][metric]) - float(lookup["C0"][seed][metric])
                for seed in seeds
            ]
            row[metric] = fmt(values)
            row[f"Δ{metric}"] = fmt(deltas)
        high_confusion = []
        for seed in seeds:
            for item in read_csv(
                dataset_root / method / f"seed_{seed}" / "control_group_metrics.csv"
            ):
                high_confusion.append(
                    float(item["high_sim_new_to_old_confusion"])
                    + float(item["high_sim_old_to_new_confusion"])
                )
        row["High-sim confusion"] = fmt(high_confusion)
        rows.append(row)
    return rows


def final_by_run(rows, value):
    candidates = defaultdict(list)
    for row in rows:
        try:
            candidates[(int(row["seed"]), int(row["task"]))].append(
                (int(row.get("epoch", 0)), float(row[value]))
            )
        except (KeyError, TypeError, ValueError):
            continue
    return {
        key: max(values, key=lambda item: item[0])[1] for key, values in candidates.items()
    }


def diagnostic_figures(dataset_root, seeds, report_dir):
    group_data = {}
    for method in METHODS:
        rows = run_rows(dataset_root, method, "control_group_metrics.csv", seeds)
        group_data[method] = {
            (int(row["seed"]), int(row["task"])): row for row in rows
        }

    fig, axis = plt.subplots(figsize=(9, 5), layout="constrained")
    x = np.arange(3)
    width = 0.16
    groups = ("high_sim", "low_sim", "random")
    for index, method in enumerate(METHODS):
        values = []
        for group in groups:
            deltas = []
            for key, row in group_data[method].items():
                baseline = group_data["C0"].get(key)
                if baseline:
                    deltas.append(
                        float(row[f"{group}_old_accuracy"])
                        - float(baseline[f"{group}_old_accuracy"])
                    )
            values.append(float(np.mean(deltas)))
        axis.bar(x + (index - 2) * width, values, width, color=COLORS[method], label=method)
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set_xticks(x, ("High", "Low", "Random"))
    axis.set(ylabel="Old-class accuracy delta vs C0 (pp)", title="Group-wise accuracy delta")
    axis.legend(ncol=5)
    save(fig, report_dir, "group_accuracy_delta")

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    positions = np.arange(len(METHODS))
    confusion = []
    for method in METHODS:
        values = [
            float(row["high_sim_new_to_old_confusion"])
            + float(row["high_sim_old_to_new_confusion"])
            for row in group_data[method].values()
        ]
        confusion.append(float(np.mean(values)))
    axis.bar(positions, confusion, color=[COLORS[m] for m in METHODS])
    axis.set_xticks(positions, METHODS)
    axis.set(ylabel="Bidirectional confusion rate (sum)", title="High-similarity pair confusion")
    save(fig, report_dir, "high_similarity_confusion")

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    old_values, new_values = [], []
    for method in METHODS:
        old_values.append(np.mean([float(row["high_sim_old_accuracy"]) for row in group_data[method].values()]))
        new_values.append(np.mean([float(row["high_sim_new_accuracy"]) for row in group_data[method].values()]))
    axis.bar(positions - 0.18, old_values, 0.36, label="Old")
    axis.bar(positions + 0.18, new_values, 0.36, label="New")
    axis.set_xticks(positions, METHODS)
    axis.set(ylabel="Accuracy (%)", title="High-similarity group old/new accuracy")
    axis.legend()
    save(fig, report_dir, "high_similarity_old_new_accuracy")

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for method in METHODS:
        final = final_by_run(run_rows(dataset_root, method, "feature_norm.csv", seeds), "mean")
        by_task = defaultdict(list)
        for (_, task), value in final.items():
            by_task[task].append(value)
        axis.plot(sorted(by_task), [np.mean(by_task[t]) for t in sorted(by_task)], marker="o", label=method, color=COLORS[method])
    axis.set(xlabel="Task", ylabel="Feature norm mean", title="Feature norm")
    axis.legend(ncol=5)
    save(fig, report_dir, "feature_norm")

    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for method in METHODS:
        final = final_by_run(
            run_rows(dataset_root, method, "control_intrusion.csv", seeds),
            "overall_intrusion",
        )
        by_task = defaultdict(list)
        for (_, task), value in final.items():
            by_task[task].append(value)
        axis.plot(sorted(by_task), [np.mean(by_task[t]) for t in sorted(by_task)], marker="o", label=method, color=COLORS[method])
    axis.set(xlabel="Task", ylabel="Raw pair intrusion rate", title="AM-Drift intrusion")
    axis.legend(ncol=5)
    save(fig, report_dir, "intrusion")


def similarity_accuracy_delta(dataset_root, seeds, report_dir):
    records = []
    for method in METHODS[1:]:
        for seed in seeds:
            selection = read_csv(
                dataset_root / method / f"seed_{seed}" / "control_pair_selection.csv"
            )
            current = read_csv(
                dataset_root / method / f"seed_{seed}" / "exports" / "class_accuracy.csv"
            )
            baseline = read_csv(
                dataset_root / "C0" / f"seed_{seed}" / "exports" / "class_accuracy.csv"
            )
            current_lookup = {
                (int(row["train_task"]), int(row["class_id"])): float(row["accuracy"])
                for row in current
            }
            baseline_lookup = {
                (int(row["train_task"]), int(row["class_id"])): float(row["accuracy"])
                for row in baseline
            }
            for pair in selection:
                task = int(pair["task"])
                old_id, new_id = int(pair["old_class_id"]), int(pair["new_class_id"])
                keys = ((task, old_id), (task, new_id))
                if not all(key in current_lookup and key in baseline_lookup for key in keys):
                    continue
                delta = np.mean(
                    [current_lookup[key] - baseline_lookup[key] for key in keys]
                )
                records.append(
                    {
                        "method": method,
                        "seed": seed,
                        "task": task,
                        "new_class_id": new_id,
                        "old_class_id": old_id,
                        "similarity": float(pair["similarity"]),
                        "pair_accuracy_delta_vs_C0": float(delta),
                    }
                )
    write_csv(report_dir / "tables" / "similarity_pair_accuracy_delta.csv", records)
    fig, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    for method in METHODS[1:]:
        selected = [row for row in records if row["method"] == method]
        if len(selected) > 10000:
            selected = selected[:: max(1, len(selected) // 10000)]
        axis.scatter(
            [row["similarity"] for row in selected],
            [row["pair_accuracy_delta_vs_C0"] for row in selected],
            s=7,
            alpha=0.12,
            label=method,
            color=COLORS[method],
        )
    axis.axhline(0, color="#777777", linewidth=1)
    axis.set(xlabel="Frozen old-model pair similarity", ylabel="Pair accuracy delta vs C0 (pp)", title="Similarity vs pair accuracy delta")
    axis.legend()
    save(fig, report_dir, "similarity_pair_accuracy_delta")


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |")
    return "\n".join(lines)


def main():
    cli = parse_args()
    dataset_root = Path(cli.output_root).expanduser().resolve() / "imageneta"
    report_dir = Path(cli.report_dir).expanduser().resolve()
    (report_dir / "tables").mkdir(parents=True, exist_ok=True)
    lookup, seeds = completed_metrics(dataset_root)
    table = main_table(lookup, seeds, dataset_root)
    write_csv(report_dir / "tables" / "main_performance.csv", table)
    diagnostic_figures(dataset_root, seeds, report_dir)
    similarity_accuracy_delta(dataset_root, seeds, report_dir)
    headers = (
        "Method",
        "n",
        "Inc-Acc",
        "Last-Acc",
        "Forgetting",
        "Old",
        "New",
        "High-sim confusion",
    )
    report = f"""# Additional Control: Does Similarity Matter?

공통 완료 seed: {', '.join(map(str, seeds))}. 모든 차이는 동일 seed의 C0와 paired 비교한다.

## Main Table

{markdown_table(headers, table)}

![group delta](figures/group_accuracy_delta.png)

![high confusion](figures/high_similarity_confusion.png)

![high old/new](figures/high_similarity_old_new_accuracy.png)

![feature norm](figures/feature_norm.png)

![intrusion](figures/intrusion.png)

![similarity delta](figures/similarity_pair_accuracy_delta.png)

## Paired contrasts

- C1−C0: high-similarity separation 제거 효과
- C1−C2: similarity 위치 선택 효과
- C1−C3: similarity selection 대 random selection
- C4−C1: weak constraint 대 complete-off

## 해석 질문

1. High-similarity separation 제거가 new learning을 개선하는가?
2. 단순 constraint 수 감소가 아니라 similarity selection의 효과인가?
3. 완전 제거와 weak constraint 중 어느 쪽이 나은가?
4. 제거 시 similar-pair confusion이 증가하는가?
5. 결과가 similarity-aware tau 또는 relative margin을 지지하는가?

위 질문의 확정 결론은 표와 그래프의 paired delta 및 seed 간 변동을 함께 보고 작성해야 한다.
"""
    # Keep the manually audited, interpretation-rich report.md intact.
    # This builder refreshes figures/tables and writes its compact summary separately.
    (report_dir / "report_auto.md").write_text(report, encoding="utf-8")
    print(report_dir / "report_auto.md")


if __name__ == "__main__":
    main()
