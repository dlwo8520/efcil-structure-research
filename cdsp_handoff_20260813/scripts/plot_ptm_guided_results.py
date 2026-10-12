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


def plot_task_retention_heatmap(
    result_root, output_dir, dataset, method="ptm_place_move", tasks=5
):
    """Plot forgetting relative to each task's accuracy when it was learned.

    Structure does not publish a task-accuracy matrix and the 5/20-task
    protocols do not have an Adapter Baseline run in this experiment.  The
    diagonal therefore shows the absolute accuracy immediately after learning
    each task, while every later cell shows current accuracy minus that
    diagonal value.  Positive values are retention gains; negative values are
    forgetting.
    """
    runs = load_complete_runs(result_root, dataset, method, tasks)
    if len(runs) != len(SEEDS):
        return None

    current = mean_matrix(runs, tasks)
    delta = np.full((tasks, tasks), np.nan, dtype=np.float64)
    for row in range(tasks):
        for column in range(row):
            delta[row, column] = current[row, column] - current[column, column]

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
                continue
            value = round(float(delta[row, column]), 1)
            if value == 0.0:
                colors[row, column] = _hex_rgb(NEUTRAL)
            else:
                strength = 0.38 + 0.57 * min(abs(value) / scale, 1.0)
                colors[row, column] = _blend_with_white(
                    GREEN if value > 0 else RED, strength
                )

    figure_size = (10.2, 7.3) if tasks <= 5 else (15.8, 12.8)
    annotation_size = 9.0 if tasks <= 5 else 5.8
    tick_size = 10 if tasks <= 5 else 7.5
    fig, ax = plt.subplots(figsize=figure_size)
    ax.imshow(colors, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(tasks), ["T{}".format(i) for i in range(1, tasks + 1)])
    ax.set_yticks(range(tasks), ["T{}".format(i) for i in range(1, tasks + 1)])
    ax.tick_params(axis="both", labelsize=tick_size)
    ax.set_xlabel("Task (Evaluation)", fontsize=13)
    ax.set_ylabel("After Training Task", fontsize=13)
    ax.set_title(
        "{} on {} — {}-task retention".format(
            METHOD_LABELS[method].replace("\n", " "),
            DATASET_LABELS[dataset], tasks,
        ),
        fontsize=16, fontweight="bold", pad=18,
    )
    ax.text(
        0.5, 1.012,
        "Diagonal: accuracy when learned   |   Lower triangle: current − initial task accuracy",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
        color="#555555",
    )

    ax.set_xticks(np.arange(-0.5, tasks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, tasks, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=1.4)
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
                fontsize=annotation_size, color=TEXT, fontweight="medium",
            )

    legend = [
        Patch(facecolor=BLUE, edgecolor="none", label="Accuracy when learned"),
        Patch(facecolor=GREEN, edgecolor="none", label="Improved since learned"),
        Patch(facecolor=RED, edgecolor="none", label="Forgot since learned"),
        Patch(facecolor=NEUTRAL, edgecolor="none", label="No material change"),
    ]
    ax.legend(
        handles=legend, bbox_to_anchor=(1.02, 0.72), loc="center left",
        frameon=False, fontsize=10.5, labelspacing=1.2,
    )
    fig.subplots_adjust(right=0.80)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "{}_{}task_{}_retention_heatmap.png".format(
        dataset, tasks, method
    )
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
        "10-task ablation vs Structure (CVPR 2026)",
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


def plot_all_tasks_final_vs_structure(result_root, reference, output_dir):
    """Compare completed 5/10/20-task Place+Move runs with Structure."""
    tasks_list = [5, 10, 20]
    metrics = [("last_accuracy", "Last-Acc"), ("incremental_accuracy", "Inc-Acc")]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), sharex="col")
    width = 0.34
    x = np.arange(len(tasks_list))

    for column, dataset in enumerate(("imageneta", "cars196")):
        for row, (key, metric_label) in enumerate(metrics):
            ax = axes[row, column]
            reference_values = []
            reference_errors = []
            current_values = []
            current_errors = []
            current_colors = []
            for tasks in tasks_list:
                ref = reference["results"][dataset][str(tasks)]
                ref_key = "last" if key == "last_accuracy" else "inc"
                reference_value = float(ref["{}_acc_mean".format(ref_key)])
                reference_error = float(ref["{}_acc_std".format(ref_key)])
                runs = load_complete_runs(
                    result_root, dataset, "ptm_place_move", tasks
                )
                if len(runs) != len(SEEDS):
                    raise RuntimeError(
                        "{} {}-task ptm_place_move does not have three complete seeds".format(
                            dataset, tasks
                        )
                    )
                current_value, current_error = _mean_std(runs, key)
                reference_values.append(reference_value)
                reference_errors.append(reference_error)
                current_values.append(current_value)
                current_errors.append(current_error)
                difference = current_value - reference_value
                current_colors.append(
                    GREEN if difference > 0 else RED if difference < 0 else NEUTRAL
                )

            reference_bars = ax.bar(
                x - width / 2, reference_values, width, yerr=reference_errors,
                capsize=3, color=BLUE, edgecolor="#4A4A4A", linewidth=0.7,
                label="Structure",
            )
            current_bars = ax.bar(
                x + width / 2, current_values, width, yerr=current_errors,
                capsize=3, color=current_colors, edgecolor="#4A4A4A",
                linewidth=0.7, label="PTM Place+Move",
            )
            for bar, value in zip(reference_bars, reference_values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + 0.8,
                    "{:.1f}".format(value), ha="center", va="bottom", fontsize=8.5,
                )
            for bar, value, reference_value in zip(
                current_bars, current_values, reference_values
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, value + 0.8,
                    "{:.1f}\n({:+.1f})".format(value, value - reference_value),
                    ha="center", va="bottom", fontsize=8.3,
                )

            all_values = reference_values + current_values
            ax.set_ylim(max(0, min(all_values) - 12), min(100, max(all_values) + 11))
            ax.set_title(
                "{} — {}".format(DATASET_LABELS[dataset], metric_label),
                fontsize=13, fontweight="bold",
            )
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(x, ["5 tasks", "10 tasks", "20 tasks"])
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.set_axisbelow(True)

    legend = [
        Patch(facecolor=BLUE, edgecolor="#4A4A4A", label="Structure reference"),
        Patch(facecolor=GREEN, edgecolor="#4A4A4A", label="Higher than Structure"),
        Patch(facecolor=RED, edgecolor="#4A4A4A", label="Lower than Structure"),
    ]
    fig.legend(
        handles=legend, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "PTM Place+Move across task granularities vs Structure",
        fontsize=17, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Values in parentheses are PTM − Structure; error bars are reported standard deviations",
        ha="center", fontsize=10, color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "all_tasks_final_vs_structure.png"
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
        for tasks in (5, 20):
            path = plot_task_retention_heatmap(
                result_root, output_dir, dataset, "ptm_place_move", tasks
            )
            if path is not None:
                generated.append(path)
    generated.append(plot_final_vs_structure(result_root, reference, output_dir))
    generated.append(
        plot_all_tasks_final_vs_structure(result_root, reference, output_dir)
    )
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
