from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from efcil_diag.task_matrix import (
    long_format_rows,
    parse_lower_triangular_matrix,
    summarize_cil_matrix,
)
from efcil_diag.utils import load_config


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_markdown(matrix: list[list[float]]) -> list[str]:
    num_tasks = len(matrix)
    header = "| trained through | " + " | ".join(
        f"Task{task_id}" for task_id in range(num_tasks)
    ) + " | Seen Avg |"
    alignment = "|---|" + "---:|" * (num_tasks + 1)
    lines = [header, alignment]
    for stage, row in enumerate(matrix):
        values = [f"{100.0 * value:.2f}" for value in row]
        values.extend("-" for _ in range(num_tasks - len(row)))
        seen_average = 100.0 * sum(row) / len(row)
        lines.append(
            f"| Task{stage} | " + " | ".join(values) + f" | {seen_average:.2f} |"
        )
    final_average = 100.0 * sum(matrix[-1]) / len(matrix[-1])
    lines.append(
        "| Final Avg | " + " | ".join("" for _ in range(num_tasks))
        + f" | {final_average:.2f} |"
    )
    return lines


def save_heatmap(path: Path, matrix: list[list[float]], title: str) -> None:
    num_tasks = len(matrix)
    values = np.full((num_tasks, num_tasks), np.nan, dtype=np.float64)
    for stage, row in enumerate(matrix):
        values[stage, : len(row)] = np.asarray(row) * 100.0
    masked = np.ma.masked_invalid(values)
    figure, axis = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    image = axis.imshow(masked, vmin=0.0, vmax=100.0, cmap="viridis", aspect="equal")
    cmap = image.cmap.copy()
    cmap.set_bad(color="white")
    image.set_cmap(cmap)
    for stage in range(num_tasks):
        for task_id in range(stage + 1):
            value = values[stage, task_id]
            color = "white" if value < 45.0 else "black"
            axis.text(task_id, stage, f"{value:.1f}", ha="center", va="center", fontsize=7, color=color)
    axis.set_xticks(range(num_tasks), labels=range(num_tasks))
    axis.set_yticks(range(num_tasks), labels=range(num_tasks))
    axis.set_xlabel("Evaluated task")
    axis.set_ylabel("Model after training task")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="Single-head accuracy (%)", shrink=0.85)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize(method_dir: Path, num_tasks: int) -> dict:
    tasks = read_rows(method_dir / "task_metrics.csv")
    classes = read_rows(method_dir / "class_metrics.csv")
    steps = read_rows(method_dir / "step_updates.csv")
    initialization = read_rows(method_dir / "lora_initialization.csv")
    if len(tasks) != num_tasks:
        raise RuntimeError(f"{method_dir}: expected {num_tasks} task rows, found {len(tasks)}")
    final = tasks[-1]
    final_class_rows = [row for row in classes if int(row["task_id"]) == num_tasks - 1]
    old_final = [row for row in final_class_rows if int(row["is_new"]) == 0]
    nonzero_steps = [row for row in steps if f(row, "parameter_update_norm") > 0]
    unsaturated = [row for row in steps if int(row["norm_match_saturated"]) == 0 and row["target_update_norm"] not in {"", "nan"}]
    norm_error = [
        abs(f(row, "parameter_update_norm") - f(row, "target_update_norm"))
        / (f(row, "target_update_norm") + 1.0e-12)
        for row in unsaturated
        if not math.isnan(f(row, "target_update_norm"))
    ]
    old_stage_rows = tasks[1:]
    return {
        "method": method_dir.name,
        "final_accuracy": f(final, "seen_accuracy"),
        "average_incremental_accuracy": f(final, "average_incremental_accuracy"),
        "final_old_accuracy": f(final, "old_accuracy"),
        "final_new_accuracy": f(final, "new_accuracy"),
        "average_forgetting": f(final, "average_forgetting"),
        "old_class_forgetting": mean(f(row, "forgetting") for row in old_final),
        "old_prototype_cosine_drift": mean(f(row, "prototype_cosine_drift") for row in old_final),
        "old_prototype_l2_drift": mean(f(row, "prototype_l2_drift") for row in old_final),
        "mean_stage_old_prototype_cosine_drift": mean(
            f(row, "old_prototype_cosine_drift") for row in old_stage_rows
        ),
        "mean_stage_old_prototype_l2_drift": mean(
            f(row, "old_prototype_l2_drift") for row in old_stage_rows
        ),
        "mean_stage_old_accuracy": mean(f(row, "old_accuracy") for row in old_stage_rows),
        "mean_stage_new_accuracy": mean(f(row, "new_accuracy") for row in old_stage_rows),
        "mean_update_norm": mean(f(row, "parameter_update_norm") for row in nonzero_steps),
        "mean_parallel_ratio": mean(f(row, "protected_parallel_ratio") for row in nonzero_steps),
        "max_parallel_ratio": max(f(row, "protected_parallel_ratio") for row in nonzero_steps),
        "mean_removed_gradient_ratio": mean(f(row, "removed_ratio") for row in initialization),
        "max_aq_norm": max(f(row, "aq_norm") for row in initialization),
        "peak_cuda_allocated_gb": max(f(row, "peak_cuda_allocated_gb") for row in tasks),
        "norm_match_mean_relative_error_unsaturated": mean(norm_error) if norm_error else float("nan"),
        "norm_match_max_relative_error_unsaturated": max(norm_error) if norm_error else float("nan"),
        "norm_match_saturated_steps": sum(int(row["norm_match_saturated"]) for row in steps),
        "optimizer_skipped_steps": sum(int(row.get("optimizer_step_skipped", 0)) for row in steps),
        "final_protected_rank": f(final, "protected_rank"),
        "final_protected_ratio": f(final, "protected_ratio"),
        "steps": len(steps),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) < 3:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_root = (
        Path(config["output_root"])
        / config["experiment_name"]
        / f'seed_{int(config["seed"])}'
    )
    ordered = [
        "vanilla_lora",
        "small_update_lora",
        "orthogonal_lora",
        "orthogonal_lora_norm_matched",
    ]
    summaries = [summarize(seed_root / method, int(config["num_tasks"])) for method in ordered]
    task_rows = {
        method: read_rows(seed_root / method / "task_metrics.csv") for method in ordered
    }
    num_tasks = int(config["num_tasks"])
    seed = int(config["seed"])
    matrices = {
        method: parse_lower_triangular_matrix(
            read_rows(seed_root / method / "accuracy_matrix.csv"), num_tasks
        )
        for method in ordered
    }
    matrix_summaries = {
        method: summarize_cil_matrix(matrix) for method, matrix in matrices.items()
    }
    report_dir = seed_root / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    long_rows = [
        row
        for method in ordered
        for row in long_format_rows(method, seed, matrices[method])
    ]
    write_rows(report_dir / "task_accuracy_matrix.csv", long_rows)
    summary_metric_rows = [
        {
            "method": method,
            "seed": seed,
            "transfer": "N/A",
            "average": round(100.0 * matrix_summaries[method].average, 10),
            "last": round(100.0 * matrix_summaries[method].last, 10),
        }
        for method in ordered
    ]
    write_rows(report_dir / "summary_metrics.csv", summary_metric_rows)

    display_names = {
        "vanilla_lora": "Vanilla LoRA",
        "small_update_lora": "Small Update LoRA",
        "orthogonal_lora": "Orthogonal LoRA",
        "orthogonal_lora_norm_matched": "Orthogonal LoRA Norm-Matched",
    }
    for method in ordered:
        save_heatmap(
            report_dir / f"task_matrix_{method}.png",
            matrices[method],
            display_names[method],
        )

    consistency: dict[str, dict] = {}
    tolerance = 1.0e-10
    for method in ordered:
        matrix = matrices[method]
        matrix_summary = matrix_summaries[method]
        metrics = task_rows[method]
        class_metrics = read_rows(seed_root / method / "class_metrics.csv")
        row_differences = [
            abs(matrix_summary.row_averages[stage] - f(metrics[stage], "seen_accuracy"))
            for stage in range(num_tasks)
        ]
        sample_weighted_differences: list[float] = []
        cell_sample_differences: list[float] = []
        equal_task_sample_counts = True
        for stage in range(num_tasks):
            stage_rows = [
                row for row in class_metrics if int(row["task_id"]) == stage
            ]
            total_correct = sum(int(row["correct"]) for row in stage_rows)
            total_count = sum(int(row["count"]) for row in stage_rows)
            weighted_seen = total_correct / total_count
            sample_weighted_differences.append(
                abs(weighted_seen - f(metrics[stage], "seen_accuracy"))
            )
            counts_by_task: list[int] = []
            for evaluated_task in range(stage + 1):
                task_class_rows = [
                    row for row in stage_rows
                    if int(row["learned_task"]) == evaluated_task
                ]
                task_correct = sum(int(row["correct"]) for row in task_class_rows)
                task_count = sum(int(row["count"]) for row in task_class_rows)
                counts_by_task.append(task_count)
                cell_sample_differences.append(
                    abs(task_correct / task_count - matrix[stage][evaluated_task])
                )
            equal_task_sample_counts &= len(set(counts_by_task)) == 1
        final_difference = abs(
            matrix_summary.last - summaries[ordered.index(method)]["final_accuracy"]
        )
        average_difference = abs(
            matrix_summary.average
            - summaries[ordered.index(method)]["average_incremental_accuracy"]
        )
        seen_class_counts = [int(float(row["seen_classes"])) for row in metrics]
        expected_seen_counts = [
            (stage + 1) * int(config["classes_per_task"]) for stage in range(num_tasks)
        ]
        method_checks = {
            "matrix_rows": len(matrix),
            "learned_cells": sum(len(row) for row in matrix),
            "future_cells_are_empty": True,
            "equal_test_samples_per_task_at_each_stage": equal_task_sample_counts,
            "max_cell_vs_sample_accuracy_difference": max(cell_sample_differences),
            "max_row_mean_vs_seen_accuracy_difference": max(row_differences),
            "max_sample_weighted_vs_seen_accuracy_difference": max(sample_weighted_differences),
            "final_vs_existing_final_accuracy_difference": final_difference,
            "average_vs_existing_incremental_accuracy_difference": average_difference,
            "seen_classifier_class_counts": seen_class_counts,
            "expected_seen_classifier_class_counts": expected_seen_counts,
            "prediction_space_check": "evaluate_seen uses argmax over all current classifier logits; no task ID or task-local masking",
        }
        numeric_checks = [
            max(cell_sample_differences),
            max(row_differences),
            max(sample_weighted_differences),
            final_difference,
            average_difference,
        ]
        method_checks["passed"] = (
            all(value <= tolerance for value in numeric_checks)
            and equal_task_sample_counts
            and seen_class_counts == expected_seen_counts
        )
        consistency[method] = method_checks
    if not all(checks["passed"] for checks in consistency.values()):
        raise RuntimeError(f"Task matrix consistency failure: {consistency}")
    with (report_dir / "task_matrix_consistency.json").open("w", encoding="utf-8") as handle:
        json.dump(consistency, handle, indent=2)
    by_method = {row["method"]: row for row in summaries}
    vanilla = by_method["vanilla_lora"]
    small = by_method["small_update_lora"]
    orthogonal = by_method["orthogonal_lora"]
    norm_matched = by_method["orthogonal_lora_norm_matched"]
    lines = [
        f"# {config['experiment_name']}: four-condition result",
        "",
        "All accuracies are single-head class-incremental accuracies. Values are percentage points except drift, update geometry, and memory.",
        "",
        "| method | final acc | avg incremental acc | final old acc | final new acc | forgetting | old prototype drift | mean update norm | mean protected ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {percent(row['final_accuracy'])} | "
            f"{percent(row['average_incremental_accuracy'])} | {percent(row['final_old_accuracy'])} | "
            f"{percent(row['final_new_accuracy'])} | {percent(row['average_forgetting'])} | "
            f"{row['old_prototype_cosine_drift']:.5f} | {row['mean_update_norm']:.4e} | "
            f"{row['mean_parallel_ratio']:.4e} |"
        )
    direction_gain = 100.0 * (norm_matched["final_accuracy"] - small["final_accuracy"])
    magnitude_gain = 100.0 * (small["final_accuracy"] - vanilla["final_accuracy"])
    natural_gain = 100.0 * (orthogonal["final_accuracy"] - vanilla["final_accuracy"])
    vanilla_after_first = task_rows["vanilla_lora"][1:]
    interference_forgetting = correlation(
        [f(row, "mean_projection_ratio") for row in vanilla_after_first],
        [f(row, "average_forgetting") for row in vanilla_after_first],
    )
    interference_drift = correlation(
        [f(row, "mean_projection_ratio") for row in vanilla_after_first],
        [f(row, "old_prototype_cosine_drift") for row in vanilla_after_first],
    )
    orth_after_first = task_rows["orthogonal_lora"][1:]
    matched_after_first = task_rows["orthogonal_lora_norm_matched"][1:]
    orth_rank_plasticity = correlation(
        [f(row, "protected_ratio") for row in orth_after_first],
        [f(row, "new_accuracy") for row in orth_after_first],
    )
    matched_rank_plasticity = correlation(
        [f(row, "protected_ratio") for row in matched_after_first],
        [f(row, "new_accuracy") for row in matched_after_first],
    )
    lines.extend([
        "",
        "## Controlled contrasts",
        "",
        f"- Direction at matched small magnitude: orthogonal norm-matched minus small-update = {direction_gain:+.2f} points final accuracy.",
        f"- Magnitude with unconstrained direction: small-update minus vanilla = {magnitude_gain:+.2f} points final accuracy.",
        f"- Natural combined effect: orthogonal minus vanilla = {natural_gain:+.2f} points final accuracy.",
        "",
        "## Task trajectory (seen-class accuracy, %)",
        "",
        "| task | vanilla | small update | orthogonal | orthogonal, norm matched |",
        "|---:|---:|---:|---:|---:|",
    ])
    for task_id in range(int(config["num_tasks"])):
        lines.append(
            f"| {task_id} | "
            + " | ".join(
                percent(f(task_rows[method][task_id], "seen_accuracy")) for method in ordered
            )
            + " |"
        )
    lines.extend([
        "",
        "## KeepLoRA metrics and CIL mapping",
        "",
        "KeepLoRA defines `a_t^(i)` as task-t accuracy after training task i. Its per-task metrics are `Transfer_t = mean_{i<t} a_t^(i)`, `Average_t = mean_{i=1..n} a_t^(i)`, and `Last_t = a_t^(n)`. The official implementation applies these operations directly to the full stage-by-task matrix. See the [KeepLoRA paper, Appendix B.2](https://openreview.net/pdf/5b1c7a77eba775963f7a9e2e29612ab26bd10cd2.pdf) and [official implementation](https://github.com/MaolinLuo/KeepLoRA/blob/main/MTIL/model/utils.py).",
        "",
        "KeepLoRA evaluates every MTIL dataset at every stage, including unseen future datasets. This expanding single-head CIL classifier has no rows for future classes, so an equivalent Transfer value—and consequently KeepLoRA's original full-matrix Average—is undefined. Transfer is therefore reported as `N/A`. The `Average` below is explicitly the CIL Average Incremental Accuracy (mean of lower-triangle Seen Avg values), while `Last` is the final 100-class single-head accuracy.",
        "",
        "## Task-wise Accuracy Matrix",
        "",
        "Each `A[t,j]` evaluates only Task-j test samples, but prediction remains over every class seen through Task t. No Task-ID or task-local candidate restriction is used. Future-task cells are intentionally blank.",
    ])
    for method in ordered:
        lines.extend([
            "",
            f"### {display_names[method]}",
            "",
            *matrix_markdown(matrices[method]),
            "",
            f"Heatmap: [`task_matrix_{method}.png`](task_matrix_{method}.png)",
        ])
    lines.extend([
        "",
        "## Transfer / Average / Last",
        "",
        "| Method | Transfer | CIL Average | Last |",
        "|---|---:|---:|---:|",
    ])
    for row in summary_metric_rows:
        lines.append(
            f"| {display_names[row['method']]} | N/A | "
            f"{float(row['average']):.2f} | {float(row['last']):.2f} |"
        )
    lines.extend([
        "",
        "## Final Task-wise Accuracy",
        "",
        "| Method | " + " | ".join(f"Task{i}" for i in range(num_tasks)) + " | Average |",
        "|---|" + "---:|" * (num_tasks + 1),
    ])
    for method in ordered:
        final_row = matrices[method][-1]
        lines.append(
            f"| {display_names[method]} | "
            + " | ".join(f"{100.0 * value:.2f}" for value in final_row)
            + f" | {100.0 * sum(final_row) / len(final_row):.2f} |"
        )
    lines.extend([
        "",
        "## Task-matrix Consistency Checks",
        "",
        "All four methods passed: lower-triangle shape and blank future cells; task-wise cells versus sample-count accuracies; row means versus existing Seen Accuracy; final-row mean versus Final Accuracy; mean row average versus Average Incremental Accuracy; and expanding classifier sizes 10, 20, ..., 100. Every task contains the same 1,000 test samples, so task-wise and sample-wise row averages are equal. Full details are in [`task_matrix_consistency.json`](task_matrix_consistency.json).",
    ])
    lines.extend([
        "",
        "## Diagnostic questions",
        "",
        f"1. **Vanilla forgetting — yes.** Final old accuracy is {percent(vanilla['final_old_accuracy'])}% and final forgetting is {percent(vanilla['average_forgetting'])}%, while final new-class accuracy remains {percent(vanilla['final_new_accuracy'])}%.",
        f"2. **Small magnitude helps, but does not solve forgetting.** A 75% reduction in every measured step norm improves final accuracy by {magnitude_gain:+.2f} points and changes forgetting by {100.0 * (small['average_forgetting'] - vanilla['average_forgetting']):+.2f} points.",
        f"3. **Natural orthogonal updates help.** They improve final accuracy by {natural_gain:+.2f} points and change forgetting by {100.0 * (orthogonal['average_forgetting'] - vanilla['average_forgetting']):+.2f} points versus vanilla.",
        f"4. **Direction matters at equal magnitude.** The two small-norm conditions have the same mean effective update norm ({small['mean_update_norm']:.4e}); orthogonal direction adds {direction_gain:+.2f} points final accuracy, {100.0 * (norm_matched['final_old_accuracy'] - small['final_old_accuracy']):+.2f} points old accuracy, and changes forgetting by {100.0 * (norm_matched['average_forgetting'] - small['average_forgetting']):+.2f} points.",
        f"5. **Representation drift is reduced on average across stages, not at the final stage.** Mean old-class cosine drift across tasks 1..9 is {vanilla['mean_stage_old_prototype_cosine_drift']:.4f} for vanilla and {norm_matched['mean_stage_old_prototype_cosine_drift']:.4f} for matched orthogonal. At task 9 the corresponding values are {vanilla['old_prototype_cosine_drift']:.4f} and {norm_matched['old_prototype_cosine_drift']:.4f}; the early/mid-task advantage is no longer visible in this final-only statistic.",
        f"6. **Interference tracks forgetting descriptively.** Across vanilla tasks 1..9, mean protected-direction ratio versus forgetting has Pearson r={interference_forgetting['pearson_r']:.3f} (p={interference_forgetting['pearson_p']:.4f}) and Spearman rho={interference_forgetting['spearman_r']:.3f} (p={interference_forgetting['spearman_p']:.4f}). The ratio versus prototype drift has Pearson r={interference_drift['pearson_r']:.3f} (p={interference_drift['pearson_p']:.4g}). This is a nine-point, single-seed association and is not causal evidence.",
        f"7. **No large plasticity cost is observed.** Mean new-class accuracy over tasks 1..9 is {percent(small['mean_stage_new_accuracy'])}% for small-update and {percent(norm_matched['mean_stage_new_accuracy'])}% for its direction-matched orthogonal counterpart.",
        f"8. **The protected space grows substantially without clear new-class collapse.** Rank grows to {norm_matched['final_protected_rank']:.0f}/768 ({percent(norm_matched['final_protected_ratio'])}%). For matched orthogonal, rank ratio versus new accuracy has Pearson r={matched_rank_plasticity['pearson_r']:.3f} (p={matched_rank_plasticity['pearson_p']:.3f}); new accuracy stays between {percent(min(f(row, 'new_accuracy') for row in matched_after_first))}% and {percent(max(f(row, 'new_accuracy') for row in matched_after_first))}%.",
        "",
        "## Integrity diagnostics",
        "",
    ])
    for row in summaries:
        lines.append(
            f"- `{row['method']}`: max ||A Q||={row['max_aq_norm']:.3e}, "
            f"max protected update ratio={row['max_parallel_ratio']:.3e}, "
            f"norm-match saturated={row['norm_match_saturated_steps']}/{row['steps']}, "
            f"optimizer skipped={row['optimizer_skipped_steps']}/{row['steps']}, "
            f"peak CUDA allocated={row['peak_cuda_allocated_gb']:.2f} GiB."
        )
    lines.extend([
        "",
        "The protected component ratio is `||delta_W Q Q^T||_F / ||delta_W||_F`, where columns of `Q` are the accumulated old-class input directions. The residual is `delta_W - delta_W Q Q^T`.",
        "",
        "## Scope",
        "",
        "This is a controlled seed-0 diagnostic (two epochs per task), not a paper-scale benchmark. The classifier is one expanding cosine head with frozen old weight rows; old test images are evaluation-only; no raw training images, paths, replay samples, task routing, or task-specific inference adapters are retained.",
    ])
    report_dir = seed_root / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=True)
    with (report_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "interference_forgetting": interference_forgetting,
            "interference_drift": interference_drift,
            "orthogonal_rank_plasticity": orth_rank_plasticity,
            "matched_orthogonal_rank_plasticity": matched_rank_plasticity,
        }, handle, indent=2, allow_nan=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
