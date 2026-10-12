#!/usr/bin/env python3
"""Rebuild E0-E7 comparisons and figures from saved CSV/JSON records only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EXPERIMENTS = [f"E{index}" for index in range(8)]
TASK_COUNT = 5
METHODS = {
    "E0": (False, "none"), "E1": (True, "none"),
    "E2": (False, "isotropic"), "E3": (False, "diagonal"),
    "E4": (False, "full"), "E5": (True, "isotropic"),
    "E6": (True, "diagonal"), "E7": (True, "full"),
}


def load_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def mean_std(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def protocol_signature(run_dir):
    metadata = load_json(run_dir / "run_metadata.json", {}) or {}
    protocol = metadata.get("protocol") or load_json(run_dir / "protocol_manifest.json", {}) or {}
    splits = protocol.get("splits", {})
    identity = {
        "dataset": protocol.get("dataset"),
        "num_classes": protocol.get("num_classes"),
        "increments": protocol.get("increments"),
        "class_order_original_ids": protocol.get("class_order_original_ids"),
        "train_sha256": splits.get("train", {}).get("sha256"),
        "test_sha256": splits.get("test", {}).get("sha256"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def collect_runs(root):
    runs = {experiment: {} for experiment in EXPERIMENTS}
    exclusions = []
    for experiment in EXPERIMENTS:
        for run_dir in sorted((root / experiment).glob("seed_*")):
            try:
                seed = int(run_dir.name.removeprefix("seed_"))
            except ValueError:
                exclusions.append({
                    "experiment": experiment, "seed": None, "run_dir": str(run_dir),
                    "reason": "invalid seed directory", "failure_type": None, "failure_reason": None,
                })
                continue
            summary = load_json(run_dir / "summary.json", {}) or {}
            metadata = load_json(run_dir / "run_metadata.json", {}) or {}
            config = load_json(run_dir / "resolved_config.json", {}) or {}
            reason = None
            if not summary:
                reason = "missing summary.json"
            elif summary.get("status") != "complete" or int(summary.get("completed_tasks", 0)) != TASK_COUNT:
                reason = (
                    f"run status={summary.get('status', metadata.get('status', 'unknown'))}, "
                    f"completed_tasks={summary.get('completed_tasks', 0)}"
                )
            elif summary.get("recording_status") != "complete":
                reason = (
                    f"recording_status={summary.get('recording_status', 'legacy_unknown')}; "
                    f"gaps={summary.get('recording_gaps', [])}"
                )
            run = {
                "experiment": experiment, "seed": seed, "run_dir": run_dir,
                "summary": summary, "metadata": metadata, "config": config,
                "protocol_signature": protocol_signature(run_dir), "complete": reason is None,
            }
            runs[experiment][seed] = run
            if reason:
                exclusions.append({
                    "experiment": experiment, "seed": seed, "run_dir": str(run_dir), "reason": reason,
                    "failure_type": metadata.get("failure_type"),
                    "failure_reason": metadata.get("failure_reason"),
                })
    return runs, exclusions


def common_comparable_seeds(runs, exclusions):
    seed_sets = [
        {seed for seed, run in runs[experiment].items() if run["complete"]}
        for experiment in EXPERIMENTS
    ]
    common = set.intersection(*seed_sets) if seed_sets else set()
    comparable = []
    for seed in sorted(common):
        signatures = {runs[experiment][seed]["protocol_signature"] for experiment in EXPERIMENTS}
        if len(signatures) == 1:
            comparable.append(seed)
        else:
            exclusions.append({
                "experiment": "E0-E7", "seed": seed, "run_dir": None,
                "reason": "protocol signatures differ across experiments",
                "failure_type": None, "failure_reason": None,
            })
    return comparable


def per_seed_rows(runs, common_seeds):
    rows = []
    for experiment in EXPERIMENTS:
        for seed in common_seeds:
            run = runs[experiment][seed]
            summary, config = run["summary"], run["config"]
            accuracies = summary.get("A_by_task", [])
            rows.append({
                "experiment": experiment,
                "multi_center": config.get("multi_center"),
                "separation_covariance": (
                    config.get("separation_covariance") if config.get("old_new_separation") else "none"
                ),
                "seed": seed,
                **{f"A_{task + 1}": accuracies[task] if len(accuracies) > task else None for task in range(TASK_COUNT)},
                "Inc-Acc": summary.get("Inc-Acc"), "Last-Acc": summary.get("Last-Acc"),
                "forgetting": summary.get("Forgetting"),
                "total_training_seconds": summary.get("total_training_seconds"),
                "max_peak_cuda_allocated_bytes": summary.get("max_peak_cuda_allocated_bytes"),
                "max_peak_cuda_reserved_bytes": summary.get("max_peak_cuda_reserved_bytes"),
                "max_peak_main_process_rss_bytes": summary.get("max_peak_main_process_rss_bytes"),
                "final_total_statistics_bytes": summary.get("final_total_statistics_bytes"),
                "max_separation_cache_bytes": summary.get("max_separation_cache_bytes"),
                "protocol_signature": run["protocol_signature"], "run_dir": str(run["run_dir"]),
            })
    return rows


def mean_std_rows(per_seed, common_seeds):
    metrics = [
        *[f"A_{task}" for task in range(1, TASK_COUNT + 1)], "Inc-Acc", "Last-Acc", "forgetting",
        "total_training_seconds", "max_peak_cuda_allocated_bytes", "max_peak_cuda_reserved_bytes",
        "max_peak_main_process_rss_bytes", "final_total_statistics_bytes", "max_separation_cache_bytes",
    ]
    rows = []
    for experiment in EXPERIMENTS:
        selected = [row for row in per_seed if row["experiment"] == experiment]
        config_row = selected[0] if selected else {}
        default_multi_center, default_covariance = METHODS[experiment]
        row = {
            "experiment": experiment,
            "multi_center": config_row.get("multi_center", default_multi_center),
            "separation_covariance": config_row.get("separation_covariance", default_covariance),
            "valid_seed_count": len(selected), "common_seeds": ",".join(map(str, common_seeds)),
        }
        for metric in metrics:
            mean, std = mean_std([as_float(item.get(metric)) for item in selected])
            row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
        rows.append(row)
    return rows


def collect_loss_rows(runs, common_seeds):
    rows = []
    for experiment in EXPERIMENTS:
        for seed in common_seeds:
            rows.extend(read_csv(runs[experiment][seed]["run_dir"] / "epoch_metrics.csv"))
    return rows


def aggregate_losses(loss_rows):
    metric_names = [
        "classification_raw", "scc_raw", "feature_distillation_raw", "logit_distillation_raw",
        "separation_raw", "original_structure_loss", "total_loss", "intrusion_ratio",
        "nearest_old_rho_mean", "epoch_seconds",
    ]
    grouped = defaultdict(list)
    for row in loss_rows:
        grouped[(row.get("experiment_id"), row.get("task_id"), row.get("stage"), row.get("epoch"))].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: (item[0], int(item[1]), item[2], int(item[3]))):
        experiment, task_id, stage, epoch = key
        values = grouped[key]
        record = {
            "experiment": experiment, "task_id": task_id, "stage": stage, "epoch": epoch,
            "valid_seed_count": len({row.get("seed") for row in values}),
        }
        for metric in metric_names:
            mean, std = mean_std([as_float(row.get(metric)) for row in values])
            record[f"{metric}_mean"], record[f"{metric}_std"] = mean, std
        output.append(record)
    return output


def collect_accuracy_rows(runs, common_seeds):
    rows = []
    for experiment in EXPERIMENTS:
        for seed in common_seeds:
            rows.extend(read_csv(runs[experiment][seed]["run_dir"] / "task_accuracy.csv"))
    return rows


def collect_task_summary_rows(runs, common_seeds):
    rows = []
    for experiment in EXPERIMENTS:
        for seed in common_seeds:
            rows.extend(read_csv(runs[experiment][seed]["run_dir"] / "task_summary.csv"))
    return rows


def aggregate_task_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("experiment_id"), row.get("task_id"), row.get("eval_stage"))].append(row)
    output = []
    for key in sorted(grouped, key=lambda item: (item[0], int(item[1]), item[2])):
        experiment, task_id, eval_stage = key
        selected = grouped[key]
        record = {
            "experiment": experiment, "task_id": task_id, "eval_stage": eval_stage,
            "valid_seed_count": len({row.get("seed") for row in selected}),
        }
        for metric in ("accuracy", "old_accuracy", "new_accuracy"):
            mean, std = mean_std([as_float(row.get(metric)) for row in selected])
            record[f"{metric}_mean"], record[f"{metric}_std"] = mean, std
        output.append(record)
    return output


def plot_results(destination, mean_rows, loss_mean_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for row in mean_rows:
        y = [as_float(row.get(f"A_{task}_mean")) for task in range(1, TASK_COUNT + 1)]
        if any(value is not None for value in y):
            axis.plot(range(1, TASK_COUNT + 1), y, marker="o", label=row["experiment"])
    axis.set(xlabel="Completed task", ylabel="Seen-class accuracy (%)", xticks=range(1, TASK_COUNT + 1))
    axis.grid(alpha=0.25)
    if axis.lines:
        axis.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(figures / "task_accuracy_curves.png", dpi=160)
    plt.close(fig)

    loss_metrics = ["classification_raw", "scc_raw", "separation_raw", "total_loss"]
    for stage in ("feature_training", "classifier_retraining"):
        selected = [row for row in loss_mean_rows if row["stage"] == stage]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for axis, metric in zip(axes.flat, loss_metrics):
            for experiment in EXPERIMENTS:
                points = []
                for row in selected:
                    if row["experiment"] != experiment:
                        continue
                    value = as_float(row.get(f"{metric}_mean"))
                    if value is not None:
                        points.append(((int(row["task_id"]) - 1) * 100 + int(row["epoch"]), value))
                if points:
                    axis.plot([x for x, _ in points], [y for _, y in points], label=experiment)
            axis.set_title(metric)
            axis.grid(alpha=0.25)
        axes[-1, 0].set_xlabel("(task-1)*100 + epoch")
        axes[-1, 1].set_xlabel("(task-1)*100 + epoch")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=8)
        fig.suptitle(stage)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(figures / f"loss_{stage}.png", dpi=160)
        plt.close(fig)


def markdown_report(
    destination, per_seed, mean_rows, accuracy_rows, task_summary_means,
    loss_mean_rows, common_seeds, exclusions,
):
    def display(value, digits=3):
        value = as_float(value)
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [
        "# E0–E7 comparison report", "",
        "이 보고서는 저장된 CSV/JSON만으로 생성되며 accuracy 단위는 0–100이다.",
        "평균 비교에는 모든 E0–E7에서 완료되고 protocol signature가 같은 seed 교집합만 사용한다.", "",
        f"- Common comparable seeds: {', '.join(map(str, common_seeds)) if common_seeds else '없음'}",
        f"- Excluded/incomplete records: {len(exclusions)}", "", "## Overall mean ± std", "",
        "| Experiment | Multi-center | Separation | Seeds | Inc-Acc | Last-Acc | Forgetting | Time (s) | GPU allocated (bytes) | Statistics (bytes) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_rows:
        pair = lambda metric: f"{display(row.get(metric + '_mean'))} ± {display(row.get(metric + '_std'))}"
        lines.append(
            f"| {row['experiment']} | {row.get('multi_center', 'N/A')} | {row.get('separation_covariance', 'N/A')} | "
            f"{row['valid_seed_count']} | {pair('Inc-Acc')} | {pair('Last-Acc')} | {pair('forgetting')} | "
            f"{pair('total_training_seconds')} | {pair('max_peak_cuda_allocated_bytes')} | "
            f"{pair('final_total_statistics_bytes')} |"
        )
    lines.extend(["", "## Per-seed A1–A5", ""])
    lines.append("| Experiment | Seed | A1 | A2 | A3 | A4 | A5 | Inc-Acc | Last-Acc | Forgetting | Time (s) | GPU bytes | Statistics bytes |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in per_seed:
        lines.append(
            "| {experiment} | {seed} | {a1} | {a2} | {a3} | {a4} | {a5} | {inc} | {last} | {forget} | {time} | {gpu} | {stats} |".format(
                experiment=row["experiment"], seed=row["seed"],
                a1=display(row["A_1"]), a2=display(row["A_2"]), a3=display(row["A_3"]),
                a4=display(row["A_4"]), a5=display(row["A_5"]), inc=display(row["Inc-Acc"]),
                last=display(row["Last-Acc"]), forget=display(row["forgetting"]),
                time=display(row["total_training_seconds"]),
                gpu=display(row["max_peak_cuda_allocated_bytes"], 0),
                stats=display(row["final_total_statistics_bytes"], 0),
            )
        )

    lines.extend(["", "## Mean task-accuracy matrices", ""])
    grouped_accuracy = defaultdict(list)
    for row in accuracy_rows:
        grouped_accuracy[(row["experiment_id"], int(row["trained_task"]), int(row["evaluated_task"]))].append(as_float(row["accuracy"]))
    for experiment in EXPERIMENTS:
        lines.extend([
            f"### {experiment}", "", "| Trained | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for trained in range(1, TASK_COUNT + 1):
            cells = []
            for evaluated in range(1, TASK_COUNT + 1):
                values = grouped_accuracy.get((experiment, trained, evaluated), [])
                cells.append(display(mean_std(values)[0]) if values else "N/A")
            lines.append(f"| Task {trained} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## Seen/old/new task accuracy", ""])
    lines.extend([
        "| Experiment | Task | Stage | Seeds | Seen accuracy | Old accuracy | New accuracy |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for row in task_summary_means:
        pair = lambda metric: f"{display(row.get(metric + '_mean'))} ± {display(row.get(metric + '_std'))}"
        lines.append(
            f"| {row['experiment']} | {row['task_id']} | {row['eval_stage']} | {row['valid_seed_count']} | "
            f"{pair('accuracy')} | {pair('old_accuracy')} | {pair('new_accuracy')} |"
        )

    lines.extend(["", "## Loss records", ""])
    lines.extend([
        "Loss는 stage와 이름별 별도 열로 유지한다. 비활성 또는 과거 누락 값은 N/A이며 0으로 대체하지 않는다.",
        f"집계 가능한 epoch rows: {len(loss_mean_rows)}", "",
        "아래 표는 각 experiment/task/stage에서 마지막으로 기록된 epoch이며 서로 다른 loss는 별도 열이다.", "",
        "| Experiment | Task | Stage | Epoch | Classification | SCC | Separation | Total |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    final_loss_rows = {}
    for row in loss_mean_rows:
        key = (row["experiment"], int(row["task_id"]), row["stage"])
        if key not in final_loss_rows or int(row["epoch"]) > int(final_loss_rows[key]["epoch"]):
            final_loss_rows[key] = row
    for key in sorted(final_loss_rows):
        row = final_loss_rows[key]
        lines.append(
            f"| {row['experiment']} | {row['task_id']} | {row['stage']} | {row['epoch']} | "
            f"{display(row.get('classification_raw_mean'))} | {display(row.get('scc_raw_mean'))} | "
            f"{display(row.get('separation_raw_mean'))} | {display(row.get('total_loss_mean'))} |"
        )
    lines.extend([
        "",
        "![Task accuracy curves](figures/task_accuracy_curves.png)", "",
        "![Feature-stage losses](figures/loss_feature_training.png)", "",
        "![Classifier-retraining losses](figures/loss_classifier_retraining.png)", "",
        "## Excluded and incomplete runs", "",
    ])
    if exclusions:
        lines.extend(["| Experiment | Seed | Reason | Failure |", "|---|---:|---|---|"])
        for row in exclusions:
            lines.append(
                f"| {row.get('experiment')} | {row.get('seed', 'N/A')} | {row.get('reason')} | "
                f"{row.get('failure_type') or ''}: {row.get('failure_reason') or ''} |"
            )
    else:
        lines.append("없음")
    (destination / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="imageneta")
    args = parser.parse_args()
    root = Path(args.output_root).resolve() / args.dataset
    destination = root / "aggregate"
    destination.mkdir(parents=True, exist_ok=True)

    runs, exclusions = collect_runs(root)
    common_seeds = common_comparable_seeds(runs, exclusions)
    per_seed = per_seed_rows(runs, common_seeds)
    mean_rows = mean_std_rows(per_seed, common_seeds)
    loss_rows = collect_loss_rows(runs, common_seeds)
    loss_mean_rows = aggregate_losses(loss_rows)
    accuracy_rows = collect_accuracy_rows(runs, common_seeds)
    task_summary_rows = collect_task_summary_rows(runs, common_seeds)
    task_summary_means = aggregate_task_summaries(task_summary_rows)

    per_seed_fields = [
        "experiment", "multi_center", "separation_covariance", "seed",
        *[f"A_{task}" for task in range(1, TASK_COUNT + 1)],
        "Inc-Acc", "Last-Acc", "forgetting", "total_training_seconds",
        "max_peak_cuda_allocated_bytes", "max_peak_cuda_reserved_bytes",
        "max_peak_main_process_rss_bytes", "final_total_statistics_bytes",
        "max_separation_cache_bytes", "protocol_signature", "run_dir",
    ]
    mean_fields = list(mean_rows[0])
    loss_fields = list(loss_mean_rows[0]) if loss_mean_rows else [
        "experiment", "task_id", "stage", "epoch", "valid_seed_count"
    ]
    task_summary_fields = list(task_summary_means[0]) if task_summary_means else [
        "experiment", "task_id", "eval_stage", "valid_seed_count"
    ]
    exclusion_fields = ["experiment", "seed", "run_dir", "reason", "failure_type", "failure_reason"]
    write_rows(destination / "comparison_per_seed.csv", per_seed, per_seed_fields)
    write_rows(destination / "comparison_mean_std.csv", mean_rows, mean_fields)
    write_rows(
        destination / "loss_per_epoch.csv", loss_rows,
        list(loss_rows[0]) if loss_rows else ["experiment_id", "seed", "task_id", "stage", "epoch"],
    )
    write_rows(destination / "loss_mean_std.csv", loss_mean_rows, loss_fields)
    write_rows(
        destination / "task_summary_per_seed.csv", task_summary_rows,
        list(task_summary_rows[0]) if task_summary_rows else [
            "experiment_id", "seed", "task_id", "eval_stage", "accuracy", "old_accuracy", "new_accuracy"
        ],
    )
    write_rows(destination / "task_summary_mean_std.csv", task_summary_means, task_summary_fields)
    write_rows(destination / "excluded_runs.csv", exclusions, exclusion_fields)
    write_rows(destination / "summary.csv", mean_rows, mean_fields)
    (destination / "raw_runs.json").write_text(
        json.dumps({
            "common_comparable_seeds": common_seeds, "excluded": exclusions,
            "runs": {
                experiment: {
                    str(seed): {
                        "complete": run["complete"], "run_dir": str(run["run_dir"]),
                        "protocol_signature": run["protocol_signature"],
                    }
                    for seed, run in experiment_runs.items()
                }
                for experiment, experiment_runs in runs.items()
            },
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_results(destination, mean_rows, loss_mean_rows)
    markdown_report(
        destination, per_seed, mean_rows, accuracy_rows, task_summary_means,
        loss_mean_rows, common_seeds, exclusions,
    )
    print(destination / "comparison_report.md")


if __name__ == "__main__":
    main()
