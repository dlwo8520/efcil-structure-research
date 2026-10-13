#!/usr/bin/env python3
"""Generate reproducible PTM-guided comparison figures for the Markdown report."""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SEEDS = [1993, 1996, 1997]
METHOD_LABELS = {
    "adapter_baseline": "Adapter\nBaseline",
    "ptm_place_only": "PTM\nPlace Only",
    "ptm_place_move": "PTM\nPlace+Move",
}
DATASET_LABELS = {"imageneta": "ImageNet-A", "cars196": "Cars-196"}
BLUE = "#8EC5F4"
GREEN = "#5BD477"
RED = "#F26666"
NEUTRAL = "#EEEEEE"
GRID = "#FFFFFF"
TEXT = "#242424"


def load_complete_runs(result_root, dataset, method, tasks=10):
    runs = {}
    base = result_root / dataset / "{}_tasks".format(tasks) / method
    for seed in SEEDS:
        path = base / "seed_{}".format(seed) / "metrics.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") == "COMPLETE" and len(data.get("task_metrics", [])) == tasks:
            runs[seed] = data
    return runs


def mean_matrix(runs, tasks=10):
    if not runs:
        raise ValueError("No complete runs supplied")
    matrix = np.full((tasks, tasks), np.nan, dtype=np.float64)
    for row in range(tasks):
        for column in range(row + 1):
            values = [
                run["task_accuracy_matrix"][row][column]
                for run in runs.values()
            ]
            matrix[row, column] = statistics.mean(values)
    return matrix


def _hex_rgb(value):
    value = value.lstrip("#")
    return np.asarray([int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _blend_with_white(color, strength):
    strength = min(max(float(strength), 0.0), 1.0)
    return np.ones(3) * (1.0 - strength) + _hex_rgb(color) * strength


def plot_task_delta_heatmap(
    result_root, output_dir, dataset, method, baseline_method="adapter_baseline", tasks=10
):
    baseline_runs = load_complete_runs(result_root, dataset, baseline_method, tasks)
    method_runs = load_complete_runs(result_root, dataset, method, tasks)
    if len(baseline_runs) != len(SEEDS) or len(method_runs) != len(SEEDS):
        return None

    baseline = mean_matrix(baseline_runs, tasks)
    current = mean_matrix(method_runs, tasks)
    delta = current - baseline
    off_diagonal = [
        abs(delta[row, column])
        for row in range(tasks) for column in range(row)
    ]
    scale = max(max(off_diagonal, default=0.0), 1.0)

    colors = np.ones((tasks, tasks, 3), dtype=np.float64)
    for row in range(tasks):
        for column in range(row + 1):
            if row == column:
                colors[row, column] = _hex_rgb(BLUE)
            else:
                value = round(float(delta[row, column]), 1)
                if value == 0.0:
                    colors[row, column] = _hex_rgb(NEUTRAL)
                else:
                    strength = 0.38 + 0.57 * min(abs(value) / scale, 1.0)
                    colors[row, column] = _blend_with_white(
                        GREEN if value > 0 else RED, strength
                    )

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.imshow(colors, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(tasks), ["T{}".format(i) for i in range(1, tasks + 1)])
    ax.set_yticks(range(tasks), ["T{}".format(i) for i in range(1, tasks + 1)])
    ax.set_xlabel("Task (Evaluation)", fontsize=13)
    ax.set_ylabel("After Training Task", fontsize=13)
    ax.set_title(
        "{} on {} — task accuracy change".format(
            METHOD_LABELS[method].replace("\n", " "), DATASET_LABELS[dataset]
        ),
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5, 1.015,
        "Diagonal: absolute accuracy   |   Lower triangle: Δ vs Adapter Baseline",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color="#555555",
    )

    ax.set_xticks(np.arange(-0.5, tasks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, tasks, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=1.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")

    for row in range(tasks):
        for column in range(row + 1):
            if row == column:
                label = "{:.1f}".format(current[row, column])
            else:
                rounded = round(float(delta[row, column]), 1)
                label = "{:+.1f}".format(rounded) if rounded != 0.0 else "0.0"
            ax.text(
                column, row, label, ha="center", va="center",
                fontsize=8.3, color=TEXT, fontweight="medium",
            )

    legend = [
        Patch(facecolor=BLUE, edgecolor="none", label="Accuracy (absolute)"),
        Patch(facecolor=GREEN, edgecolor="none", label="Improved vs Adapter"),
        Patch(facecolor=RED, edgecolor="none", label="Worse vs Adapter"),
        Patch(facecolor=NEUTRAL, edgecolor="none", label="No material change"),
    ]
    ax.legend(
        handles=legend, bbox_to_anchor=(1.03, 0.72), loc="center left",
        frameon=False, fontsize=11, labelspacing=1.3,
    )
    fig.subplots_adjust(right=0.80)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}_{}_vs_adapter_task_heatmap.png".format(dataset, method)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _mean_std(runs, key):
    values = [float(run[key]) for run in runs.values()]
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def plot_final_vs_structure(result_root, reference, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7), sharey=False)
    methods = ["structure", "adapter_baseline", "ptm_place_only", "ptm_place_move"]
    metrics = [("last_accuracy", "Last-Acc"), ("incremental_accuracy", "Inc-Acc")]
    offsets = [-0.18, 0.18]
    width = 0.33

    for ax, dataset in zip(axes, ("imageneta", "cars196")):
        ref = reference["results"][dataset]["10"]
        values = {"structure": {}}
        values["structure"]["last_accuracy"] = (
            float(ref["last_acc_mean"]), float(ref["last_acc_std"])
        )
        values["structure"]["incremental_accuracy"] = (
            float(ref["inc_acc_mean"]), float(ref["inc_acc_std"])
        )
        for method in methods[1:]:
            runs = load_complete_runs(result_root, dataset, method)
            if len(runs) != len(SEEDS):
                raise RuntimeError(
                    "{} {} does not have three complete seeds".format(dataset, method)
                )
            values[method] = {key: _mean_std(runs, key) for key, _ in metrics}

        x = np.arange(len(methods))
        for metric_index, (key, metric_label) in enumerate(metrics):
            heights = [values[method][key][0] for method in methods]
            errors = [values[method][key][1] for method in methods]
            reference_value = values["structure"][key][0]
            colors = [
                BLUE if method == "structure"
                else (GREEN if values[method][key][0] > reference_value else RED)
                for method in methods
            ]
            bars = ax.bar(
                x + offsets[metric_index], heights, width,
                yerr=errors, capsize=3, color=colors, edgecolor="#4A4A4A",
                linewidth=0.7, hatch="" if metric_index == 0 else "//",
                alpha=0.94,
            )
            for bar, value in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + 1.0,
                    "{:.1f}".format(value), ha="center", va="bottom",
                    fontsize=8.5, rotation=0,
                )

        ax.set_title("{} 10-task".format(DATASET_LABELS[dataset]), fontweight="bold")
        ax.set_xticks(
            x,
            ["Structure", "Adapter\nBaseline", "PTM\nPlace Only", "PTM\nPlace+Move"],
        )
        ax.set_ylabel("Accuracy (%)")
        minimum = min(values[method][key][0] for method in methods for key, _ in metrics)
        maximum = max(values[method][key][0] for method in methods for key, _ in metrics)
        ax.set_ylim(max(0, minimum - 12), min(100, maximum + 10))
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)

    color_legend = [
        Patch(facecolor=BLUE, edgecolor="#4A4A4A", label="Structure reference"),
        Patch(facecolor=GREEN, edgecolor="#4A4A4A", label="Higher than Structure"),
        Patch(facecolor=RED, edgecolor="#4A4A4A", label="Lower than Structure"),
        Patch(facecolor="white", edgecolor="#4A4A4A", label="Last-Acc"),
        Patch(facecolor="white", edgecolor="#4A4A4A", hatch="//", label="Inc-Acc"),
    ]
    fig.legend(
        handles=color_legend, loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Final performance vs Structure (CVPR 2026)",
        fontsize=17, fontweight="bold", y=1.01,
    )
    fig.text(
        0.5, 0.945,
        "Green = higher than Structure, Red = lower than Structure; error bars are reported standard deviations",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "phase1_final_vs_structure.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def generate_plots(result_root, reference_path, output_dir):
    result_root = Path(result_root)
    output_dir = Path(output_dir)
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    generated = []
    for dataset in ("imageneta", "cars196"):
        for method in ("ptm_place_only", "ptm_place_move"):
            path = plot_task_delta_heatmap(
                result_root, output_dir, dataset, method
            )
            if path is not None:
                generated.append(path)
    generated.append(plot_final_vs_structure(result_root, reference, output_dir))
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/ptm_guided")
    parser.add_argument("--reference", default="references/structure_cvpr2026_results.json")
    parser.add_argument("--output-dir", default="figures/ptm_guided")
    args = parser.parse_args()
    paths = generate_plots(args.result_root, args.reference, args.output_dir)
    for path in paths:
        print("Wrote {}".format(path))


if __name__ == "__main__":
    main()
