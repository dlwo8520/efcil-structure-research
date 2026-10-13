#!/usr/bin/env python3
"""Rebuild the seed-1993 Structure E3 follow-up report from saved records.

No training, checkpoint evaluation, dataset loading, or GPU operation is
performed. Requires Python >=3.9, numpy, and matplotlib; pandas is not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np


METHODS = [
    "B0_STRUCTURE",
    "L01",
    "L03",
    "L05",
    "L10",
    "FCS_PIC",
    "E3_NORM",
    "E3_AHN",
    "CEOS50",
    "CEOS100",
    "AM_VARIANCE",
    "AM_AGE",
    "AM_DRIFT",
]
BASE = "B0_STRUCTURE"
SEED = 1993
EPS = 1e-10
LABELS = {
    "B0_STRUCTURE": "Structure",
    "L01": "E3 λ=.1",
    "L03": "E3 λ=.3",
    "L05": "E3 λ=.5",
    "L10": "E3 λ=1.0",
    "FCS_PIC": "FCS PIC",
    "E3_NORM": "E3-Norm",
    "E3_AHN": "E3 + AHN",
    "CEOS50": "CEOS50",
    "CEOS100": "CEOS100",
    "AM_VARIANCE": "AM-Variance",
    "AM_AGE": "AM-Age",
    "AM_DRIFT": "AM-Drift",
}
IDEAS = {
    "B0_STRUCTURE": "Structure baseline",
    "L01": "diagonal separation, λ=.1",
    "L03": "diagonal separation, λ=.3",
    "L05": "diagonal separation, λ=.5",
    "L10": "diagonal separation, λ=1.0",
    "FCS_PIC": "adapted point-prototype contrastive",
    "E3_NORM": "normalized diagonal approximation",
    "E3_AHN": "λ=.3 + adapted angular hard negative",
    "CEOS50": "50% Gaussian + 50% adapted CEOS",
    "CEOS100": "100% adapted CEOS for old classes",
    "AM_VARIANCE": "λ=.3 + variance-aware tau",
    "AM_AGE": "λ=.3 + age-aware tau",
    "AM_DRIFT": "λ=.3 + drift-aware tau",
}
COLORS = {
    "B0_STRUCTURE": "#222222",
    "L01": "#4c78a8",
    "L03": "#2f5597",
    "L05": "#6f4e9c",
    "L10": "#9c4e97",
    "FCS_PIC": "#e17c05",
    "E3_NORM": "#59a14f",
    "E3_AHN": "#00a6a6",
    "CEOS50": "#f1ce63",
    "CEOS100": "#d55e00",
    "AM_VARIANCE": "#af7aa1",
    "AM_AGE": "#e15759",
    "AM_DRIFT": "#00876c",
}
CMAP = LinearSegmentedColormap.from_list("delta", ["#c64242", "#eeeeee", "#248b53"])
ABS_CMAP = plt.get_cmap("Blues")
UNIT = "ImageNet-A · seed 1993 · n=1 · SD=N/A"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
ROOT = args.project_root.resolve()
OUT = args.output_dir.resolve()
DATA = ROOT / "outputs" / "imageneta"
for subdirectory in ("figures", "tables"):
    (OUT / subdirectory).mkdir(parents=True, exist_ok=True)

font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if font_path.is_file():
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "figure.dpi": 120,
        "savefig.dpi": 170,
    }
)

sources = {}
checks = []


def track(path):
    path = Path(path)
    if path.is_file() and str(path) not in sources:
        payload = path.read_bytes()
        sources[str(path)] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return path


def load_json(path, default=None):
    path = track(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_rows(path):
    path = track(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path):
    path = track(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def number(value):
    if value in (None, "", "N/A"):
        return float("nan")
    return float(value)


def truth(value):
    return value is True or str(value).lower() == "true"


def validate(name, condition, detail=""):
    checks.append({"check": name, "passed": bool(condition), "detail": str(detail)})
    if not condition:
        raise ValueError(f"{name}: {detail}")


def write_csv(name, records, fields=None):
    records = list(records)
    fields = fields or list(dict.fromkeys(key for row in records for key in row))
    with (OUT / "tables" / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: "N/A"
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                    for key, value in row.items()
                }
            )


def format_value(value, signed=False, digits=2):
    if not np.isfinite(value):
        return "N/A"
    if abs(value) <= EPS:
        return f"{0:.{digits}f}"
    if abs(value) < 0.5 * 10 ** (-digits):
        return f"{value:+.2e}" if signed else f"{value:.2e}"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def hms(seconds):
    seconds = round(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02}:{seconds % 60:02}"


def markdown_table(headers, records):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(escape(value) for value in row) + " |" for row in records]
    )


def save_figure(figure, name):
    for extension in ("png", "svg"):
        figure.savefig(
            OUT / "figures" / f"{name}.{extension}", bbox_inches="tight", facecolor="white"
        )
    plt.close(figure)


def delta_color(value, lower_is_better=False):
    if abs(value) <= EPS:
        return "#999999"
    good = value < 0 if lower_is_better else value > 0
    return "#248b53" if good else "#c64242"


def mean_finite(values):
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


R = {}
for method in METHODS:
    run_dir = DATA / method / f"seed_{SEED}"
    record = {
        "run_dir": run_dir,
        "config": load_json(run_dir / "resolved_config.json"),
        "metadata": load_json(run_dir / "run_metadata.json"),
        "summary": load_json(run_dir / "summary.json"),
        "protocol": load_json(run_dir / "protocol_manifest.json"),
        "checkpoint": load_json(run_dir / "checkpoint_manifest.json"),
        "source_state": load_json(run_dir / "source_state.json"),
        "snapshot_manifest": load_json(run_dir / "source_snapshot" / "manifest.json"),
    }
    for key in (
        "task_accuracy",
        "task_summary",
        "class_accuracy",
        "task_metrics",
        "epoch_metrics",
        "gradient_metrics",
        "adaptive_margin",
        "adaptive_class_diagnostics",
        "ceos_summary",
        "apr_diagnostics",
    ):
        record[key] = load_rows(run_dir / f"{key}.csv")
    record["followup_task_end"] = load_jsonl(run_dir / "followup_task_end_diagnostics.jsonl")
    track(run_dir / "training.log")

    validate(method + " complete", record["summary"]["status"] == "complete")
    validate(method + " metadata complete", record["metadata"]["status"] == "complete")
    tasks = int(record["protocol"]["num_tasks"])
    validate(method + " task count", record["summary"]["completed_tasks"] == tasks == 5)
    validate(method + " no recording gaps", not record["metadata"].get("recording_gaps"))
    checkpoint_path = run_dir / "checkpoint_latest.pt"
    validate(
        method + " final checkpoint",
        checkpoint_path.is_file()
        and record["checkpoint"]["checkpoint_state_id"] == f"task_end:task_{tasks}",
    )
    record["checkpoint_bytes"] = checkpoint_path.stat().st_size

    matrix = np.full((tasks, tasks), np.nan)
    cells = [row for row in record["task_accuracy"] if row["eval_stage"] == "task_final"]
    validate(method + " triangle cell count", len(cells) == tasks * (tasks + 1) // 2)
    for row in cells:
        trained = int(row["trained_task"]) - 1
        evaluated = int(row["evaluated_task"]) - 1
        accuracy = number(row["accuracy"])
        recomputed = 100.0 * int(row["num_correct"]) / int(row["num_samples"])
        validate(method + " task cell counts", abs(accuracy - recomputed) < EPS)
        matrix[trained, evaluated] = accuracy
    record["a"] = matrix

    task_rows = sorted(
        [row for row in record["task_summary"] if row["eval_stage"] == "task_final"],
        key=lambda row: int(row["task_id"]),
    )
    validate(method + " task summary rows", len(task_rows) == tasks)
    record["A"] = []
    record["old"] = []
    record["new"] = []
    for task_index, row in enumerate(task_rows):
        accuracy = 100.0 * int(row["num_correct"]) / int(row["num_samples"])
        validate(method + " weighted A", abs(accuracy - number(row["accuracy"])) < EPS)
        record["A"].append(accuracy)
        record["old"].append(number(row["old_accuracy"]))
        record["new"].append(number(row["new_accuracy"]))
    record["forgetting_by_task"] = [
        float(np.max(matrix[index : tasks - 1, index]) - matrix[tasks - 1, index])
        for index in range(tasks - 1)
    ]
    record["Inc"] = float(np.mean(record["A"]))
    record["Last"] = float(record["A"][-1])
    record["F"] = float(np.mean(record["forgetting_by_task"]))
    for key, stored_key in (("Inc", "Inc-Acc"), ("Last", "Last-Acc"), ("F", "Forgetting")):
        validate(method + " recomputed " + key, abs(record[key] - record["summary"][stored_key]) < EPS)
    record["wall_seconds"] = (
        datetime.fromisoformat(record["metadata"]["ended_at_utc"])
        - datetime.fromisoformat(record["metadata"]["started_at_utc"])
    ).total_seconds()
    record["final_classes"] = {
        int(row["mapped_class_id"]): row
        for row in record["class_accuracy"]
        if int(row["trained_task"]) == tasks and row["eval_stage"] == "task_final"
    }
    R[method] = record

T = 5
protocol = R[BASE]["protocol"]
validate("ImageNet-A protocol", protocol["dataset"] == "imageneta")
validate("200 classes", protocol["num_classes"] == 200)
validate("40x5 increments", protocol["increments"] == [40] * 5)
for method in METHODS:
    validate(method + " protocol identity", R[method]["protocol"] == protocol)
    validate(method + " snapshot present", R[method]["snapshot_manifest"] is not None)

snapshot_hashes = {
    hashlib.sha256((R[method]["run_dir"] / "source_snapshot" / "manifest.json").read_bytes()).hexdigest()
    for method in METHODS
}
source_hashes = {R[method]["source_state"].get("working_tree_sha256") for method in METHODS}
validate("same source snapshot", len(snapshot_hashes) == 1)
validate("same base working tree digest", len(source_hashes) == 1)

reference_workload = [
    (
        row["task_id"],
        row["stage"],
        row["epoch"],
        row["processed_samples"],
        row["optimizer_steps"],
        row["learning_rates"],
    )
    for row in R[BASE]["epoch_metrics"]
]
for method in METHODS:
    workload = [
        (
            row["task_id"],
            row["stage"],
            row["epoch"],
            row["processed_samples"],
            row["optimizer_steps"],
            row["learning_rates"],
        )
        for row in R[method]["epoch_metrics"]
    ]
    validate(method + " workload identity", workload == reference_workload)
    for row in R[method]["epoch_metrics"]:
        weighted = []
        for component in (
            "classification",
            "scc",
            "feature_distillation",
            "logit_distillation",
            "separation",
            "pic",
            "ahn",
        ):
            if truth(row.get(component + "_active")):
                raw = number(row.get(component + "_raw"))
                coefficient = number(row.get(component + "_coefficient"))
                weighted_value = number(row.get(component + "_weighted"))
                validate(
                    method + " weighted loss " + component,
                    abs(raw * coefficient - weighted_value) < 2e-5,
                )
                weighted.append(weighted_value)
        validate(
            method + " total loss decomposition",
            abs(sum(weighted) - number(row["total_loss"])) < 3e-5,
        )
        validate(method + " finite flag", not truth(row.get("nan_or_inf_detected")))


def last_feature_row(method, task):
    candidates = [
        row
        for row in R[method]["epoch_metrics"]
        if row["stage"] == "feature_training" and int(row["task_id"]) == task
    ]
    return max(candidates, key=lambda row: int(row["epoch"]))


config_differences = []
identity_keys = {
    "experiment_id",
    "prefix",
    "config",
    "run_dir",
    "launch_command",
}
method_keys = {
    "old_new_separation",
    "sep_lambda",
    "pic_enabled",
    "pic_lambda",
    "ahn_enabled",
    "ahn_lambda",
    "normalized_diagonal",
    "ceos_replay_fraction",
    "adaptive_margin_mode",
    "adaptive_margin_strength",
}
for method in METHODS:
    config = R[method]["config"]
    baseline_config = R[BASE]["config"]
    for key in sorted(set(config) | set(baseline_config)):
        if config.get(key) == baseline_config.get(key):
            continue
        config_differences.append(
            {
                "method": method,
                "key": key,
                "baseline": json.dumps(baseline_config.get(key), ensure_ascii=False),
                "value": json.dumps(config.get(key), ensure_ascii=False),
                "category": "identity"
                if key in identity_keys
                else "method"
                if key in method_keys
                else "other_control",
            }
        )
validate(
    "no uncontrolled config differences",
    not [row for row in config_differences if row["category"] == "other_control"],
)

metrics = []
task_metrics = []
cell_metrics = []
forgetting_metrics = []
base = R[BASE]
for method in METHODS:
    record = R[method]
    summary = record["summary"]
    final_old = record["old"][-1]
    final_new = record["new"][-1]
    metrics.append(
        {
            "method": method,
            "label": LABELS[method],
            "seed": SEED,
            "n": 1,
            "sample_sd": "N/A",
            "Inc_Acc": record["Inc"],
            "delta_Inc_pp": record["Inc"] - base["Inc"],
            "Last_Acc": record["Last"],
            "delta_Last_pp": record["Last"] - base["Last"],
            "Forgetting_pp": record["F"],
            "delta_Forgetting_pp": record["F"] - base["F"],
            "final_old_accuracy": final_old,
            "delta_final_old_pp": final_old - base["old"][-1],
            "final_new_accuracy": final_new,
            "delta_final_new_pp": final_new - base["new"][-1],
            "SPGap_pp": (final_old - base["old"][-1]) - (final_new - base["new"][-1]),
            "incremental_train_seconds": summary["total_training_seconds"],
            "wall_seconds": record["wall_seconds"],
            "peak_cuda_allocated_GiB": summary["max_peak_cuda_allocated_bytes"] / 2**30,
            "peak_cuda_reserved_GiB": summary["max_peak_cuda_reserved_bytes"] / 2**30,
            "final_statistics_MiB": summary["final_total_statistics_bytes"] / 2**20,
            "max_separation_cache_MiB": summary["max_separation_cache_bytes"] / 2**20,
        }
    )
    for task in range(1, T + 1):
        task_metrics.append(
            {
                "method": method,
                "seed": SEED,
                "task": task,
                "A": record["A"][task - 1],
                "delta_A_pp": record["A"][task - 1] - base["A"][task - 1],
                "old_accuracy": record["old"][task - 1],
                "delta_old_pp": record["old"][task - 1] - base["old"][task - 1],
                "new_accuracy": record["new"][task - 1],
                "delta_new_pp": record["new"][task - 1] - base["new"][task - 1],
            }
        )
        for evaluated in range(1, task + 1):
            cell_metrics.append(
                {
                    "method": method,
                    "seed": SEED,
                    "trained_task": task,
                    "evaluated_task": evaluated,
                    "accuracy": record["a"][task - 1, evaluated - 1],
                    "paired_delta_E0_pp": record["a"][task - 1, evaluated - 1]
                    - base["a"][task - 1, evaluated - 1],
                    "paired_delta_sample_sd": "N/A",
                }
            )
    for index, value in enumerate(record["forgetting_by_task"], 1):
        forgetting_metrics.append(
            {
                "method": method,
                "seed": SEED,
                "evaluated_task": index,
                "forgetting_pp": value,
                "delta_E0_pp": value - base["forgetting_by_task"][index - 1],
            }
        )

class_deltas = []
for method in METHODS:
    for class_id, row in R[method]["final_classes"].items():
        baseline_row = base["final_classes"][class_id]
        class_deltas.append(
            {
                "method": method,
                "seed": SEED,
                "mapped_class_id": class_id,
                "original_class_id": row["original_class_id"],
                "class_name": row["class_name"],
                "test_samples": int(row["num_samples"]),
                "accuracy": number(row["accuracy"]),
                "baseline_accuracy": number(baseline_row["accuracy"]),
                "delta_pp": number(row["accuracy"]) - number(baseline_row["accuracy"]),
                "correct_difference": int(row["num_correct"]) - int(baseline_row["num_correct"]),
            }
        )

write_csv("metrics_per_seed.csv", metrics)
write_csv("task_metrics_recomputed.csv", task_metrics)
write_csv("cells_per_seed_and_sd.csv", cell_metrics)
write_csv("forgetting_per_task.csv", forgetting_metrics)
write_csv("class_final_deltas.csv", class_deltas)
write_csv("config_differences.csv", config_differences)
write_csv("validation_checks.csv", checks)
for key, filename in (
    ("task_summary", "raw_task_summary.csv"),
    ("task_accuracy", "raw_task_accuracy.csv"),
    ("class_accuracy", "raw_class_accuracy.csv"),
    ("epoch_metrics", "raw_epoch_metrics.csv"),
    ("gradient_metrics", "raw_gradient_metrics.csv"),
    ("adaptive_margin", "raw_adaptive_margin.csv"),
    ("adaptive_class_diagnostics", "raw_adaptive_class_diagnostics.csv"),
    ("ceos_summary", "raw_ceos_summary.csv"),
):
    combined = []
    for method in METHODS:
        combined.extend({"method": method, **row} for row in R[method][key])
    write_csv(filename, combined)

for method in METHODS:
    absolute_rows = []
    delta_rows = []
    for task in range(T):
        absolute_rows.append(
            {"trained_task": task + 1, **{f"eval_task_{i+1}": R[method]["a"][task, i] for i in range(T)}}
        )
        delta_rows.append(
            {
                "trained_task": task + 1,
                **{
                    f"eval_task_{i+1}": R[method]["a"][task, i] - base["a"][task, i]
                    for i in range(T)
                },
            }
        )
    write_csv(f"{method}_absolute_matrix.csv", absolute_rows)
    write_csv(f"{method}_delta_B0_matrix.csv", delta_rows)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
delta_limit = max(abs(row["paired_delta_E0_pp"]) for row in cell_metrics) or 1.0
delta_limit = math.ceil(delta_limit * 10.0) / 10.0
delta_norm = Normalize(-delta_limit, delta_limit)
absolute_norm = Normalize(0.0, 100.0)


def draw_heatmap(axis, method, kind, small=False):
    absolute = R[method]["a"]
    delta = absolute - base["a"]
    for trained in range(T):
        for evaluated in range(T):
            if evaluated > trained:
                continue
            use_absolute = kind == "absolute" or (kind == "A" and trained == evaluated)
            value = absolute[trained, evaluated] if use_absolute else delta[trained, evaluated]
            missing = not np.isfinite(value)
            if missing:
                fill = "#eeeeee"
            elif use_absolute:
                fill = ABS_CMAP(absolute_norm(value))
            elif abs(value) <= EPS:
                fill = "#eeeeee"
            else:
                fill = CMAP(delta_norm(value))
            axis.add_patch(
                Rectangle(
                    (evaluated - 0.5, trained - 0.5),
                    1,
                    1,
                    facecolor=fill,
                    edgecolor="white",
                    linewidth=2,
                    hatch="///" if missing else None,
                )
            )
            axis.text(
                evaluated,
                trained,
                format_value(value, signed=not use_absolute),
                ha="center",
                va="center",
                fontsize=8 if small else 12,
                color="white" if use_absolute and value > 65 else "#111111",
            )
    axis.set(
        xlim=(-0.5, T - 0.5),
        ylim=(T - 0.5, -0.5),
        xticks=range(T),
        yticks=range(T),
        xticklabels=[f"T{i+1}" for i in range(T)],
        yticklabels=[f"T{i+1}" for i in range(T)],
        xlabel="평가 대상 task",
        ylabel="학습 완료 task",
        title=("B0 절대 정확도" if kind == "absolute" else LABELS[method] + " vs B0"),
    )
    axis.set_aspect("equal")
    for spine in axis.spines.values():
        spine.set_visible(False)


figure, axis = plt.subplots(figsize=(7, 6), layout="constrained")
draw_heatmap(axis, BASE, "absolute")
figure.suptitle(UNIT + " · Structure 절대 정확도 행렬")
figure.colorbar(
    plt.cm.ScalarMappable(norm=absolute_norm, cmap=ABS_CMAP), ax=axis, label="Accuracy (%)"
)
save_figure(figure, "B0_absolute")

for kind in ("A", "B"):
    subtitle = (
        "대각선: 절대 정확도(%) / 아래: B0 대비 paired Δ(pp)"
        if kind == "A"
        else "대각선 포함 전체 유효 영역: B0 대비 paired Δ(pp)"
    )
    for method in METHODS[1:]:
        figure, axis = plt.subplots(figsize=(7, 6.3), layout="constrained")
        draw_heatmap(axis, method, kind)
        figure.suptitle(UNIT + "\n" + subtitle, fontsize=10)
        figure.colorbar(
            plt.cm.ScalarMappable(norm=delta_norm, cmap=CMAP),
            ax=axis,
            label="Δ accuracy (pp)",
            shrink=0.8,
        )
        if kind == "A":
            axis.legend(
                handles=[Patch(facecolor=ABS_CMAP(0.8), label="대각선: 절대 정확도")],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.16),
            )
        save_figure(figure, f"heatmap_{kind}_{method}")
    figure, axes = plt.subplots(3, 4, figsize=(18, 14), layout="constrained")
    for axis, method in zip(axes.flat, METHODS[1:]):
        draw_heatmap(axis, method, kind, small=True)
    figure.suptitle(UNIT + " · 후속 방법 전체 heatmap " + kind)
    figure.colorbar(
        plt.cm.ScalarMappable(norm=delta_norm, cmap=CMAP),
        ax=axes,
        label="Δ accuracy (pp)",
        shrink=0.75,
    )
    save_figure(figure, f"heatmap_{kind}_overview")


def line_plot(name, values_by_method, title, ylabel, delta=False, methods=METHODS):
    figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for method in methods:
        values = values_by_method(method)
        axis.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            linewidth=2.7 if method == BASE else 1.4,
            color=COLORS[method],
            label=LABELS[method],
        )
    if delta:
        axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set(xticks=range(1, T + 1), xlabel="학습 완료 task", ylabel=ylabel, title=title)
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, fontsize=8)
    save_figure(figure, name)


line_plot(
    "accuracy_by_task",
    lambda method: R[method]["A"],
    UNIT + " · task별 seen-class accuracy",
    "Accuracy (%)",
)
line_plot(
    "accuracy_delta_by_task",
    lambda method: np.asarray(R[method]["A"]) - np.asarray(base["A"]),
    UNIT + " · task별 B0 paired Δ",
    "Δ accuracy (pp)",
    delta=True,
    methods=METHODS[1:],
)

figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
for axis, key, title in zip(axes, ("old", "new"), ("Old-class accuracy", "Current-task accuracy")):
    for method in METHODS:
        axis.plot(
            range(1, T + 1),
            R[method][key],
            marker="o",
            linewidth=2.5 if method == BASE else 1.3,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.set(xticks=range(1, T + 1), xlabel="학습 완료 task", ylabel="Accuracy (%)", title=title)
    axis.grid(alpha=0.2)
axes[0].legend(ncol=2, fontsize=7)
figure.suptitle(UNIT + " · T1 old=N/A")
save_figure(figure, "old_new_accuracy")

figure, axis = plt.subplots(figsize=(11, 5), layout="constrained")
x = np.arange(T - 1)
width = 0.8 / len(METHODS)
for index, method in enumerate(METHODS):
    axis.bar(
        x - 0.4 + width / 2 + index * width,
        R[method]["forgetting_by_task"],
        width,
        color=COLORS[method],
        label=LABELS[method],
    )
axis.set(
    xticks=x,
    xticklabels=[f"T{i+1}" for i in range(T - 1)],
    xlabel="평가 task",
    ylabel="Forgetting (pp)",
    title=UNIT + " · task별 forgetting (낮을수록 좋음)",
)
axis.legend(ncol=3, fontsize=7)
axis.grid(axis="y", alpha=0.2)
save_figure(figure, "forgetting_by_task")

figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
for axis, key, label in zip(axes, ("Inc", "Last"), ("Δ Inc-Acc", "Δ Last-Acc")):
    values = [R[method][key] - base[key] for method in METHODS[1:]]
    axis.bar(
        [LABELS[method] for method in METHODS[1:]],
        values,
        color=[delta_color(value) for value in values],
    )
    axis.axhline(0, color="#222222", linewidth=1)
    axis.tick_params(axis="x", rotation=55)
    axis.set(ylabel="pp", title=label + " vs B0")
    axis.margins(y=0.18)
    for index, value in enumerate(values):
        axis.annotate(
            format_value(value, signed=True),
            (index, value),
            xytext=(0, 5 if value >= 0 else -14),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
figure.suptitle(UNIT + " · 양수 정확도 변화=관측 개선")
save_figure(figure, "performance_deltas")

figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for axis, delta in zip(axes, (False, True)):
    for method in METHODS:
        values = R[method]["a"][-1].copy()
        if delta:
            values -= base["a"][-1]
        axis.plot(
            range(1, T + 1),
            values,
            marker="o",
            color=COLORS[method],
            linewidth=2.5 if method == BASE else 1.3,
            label=LABELS[method],
        )
    if delta:
        axis.axhline(0, color="#777777", linewidth=0.8)
    axis.set(
        xticks=range(1, T + 1),
        xlabel="평가 대상 task",
        ylabel="Δ accuracy (pp)" if delta else "Accuracy (%)",
        title="B0 대비 paired Δ" if delta else "최종 모델 task별 정확도",
    )
    axis.grid(alpha=0.2)
axes[0].legend(ncol=2, fontsize=7)
figure.suptitle(UNIT + " · T5 학습 완료 모델")
save_figure(figure, "final_task_accuracy")

# Lambda sweep figures.
lambda_methods = ["L01", "L03", "L05", "L10"]
lambdas = np.array([0.1, 0.3, 0.5, 1.0])


def lambda_plot(name, values, title, ylabel, horizontal_zero=False):
    figure, axis = plt.subplots(figsize=(6.7, 4.7), layout="constrained")
    axis.plot(lambdas, values, marker="o", color="#2f5597", linewidth=2)
    if horizontal_zero:
        axis.axhline(0, color="#777777", linewidth=0.8)
    for x_value, value in zip(lambdas, values):
        axis.annotate(format_value(value, signed=horizontal_zero), (x_value, value), xytext=(0, 7), textcoords="offset points", ha="center")
    axis.set(xticks=lambdas, xlabel="λ_sep", ylabel=ylabel, title=UNIT + " · " + title)
    axis.grid(alpha=0.2)
    save_figure(figure, name)


lambda_plot(
    "lambda_inc_acc",
    [R[method]["Inc"] - base["Inc"] for method in lambda_methods],
    "λ vs Δ Inc-Acc",
    "Δ Inc-Acc (pp)",
    True,
)
lambda_plot(
    "lambda_last_acc",
    [R[method]["Last"] - base["Last"] for method in lambda_methods],
    "λ vs Δ Last-Acc",
    "Δ Last-Acc (pp)",
    True,
)

figure, axis = plt.subplots(figsize=(7, 4.8), layout="constrained")
axis.plot(lambdas, [R[m]["old"][-1] - base["old"][-1] for m in lambda_methods], marker="o", label="Old Δ")
axis.plot(lambdas, [R[m]["new"][-1] - base["new"][-1] for m in lambda_methods], marker="s", label="New Δ")
axis.axhline(0, color="#777777", linewidth=0.8)
axis.set(xlabel="λ_sep", ylabel="Δ accuracy (pp)", xticks=lambdas, title=UNIT + " · λ vs final old/new Δ")
axis.grid(alpha=0.2)
axis.legend()
save_figure(figure, "lambda_old_new_delta")

lambda_plot(
    "lambda_feature_norm",
    [number(last_feature_row(method, 5)["feature_norm_mean"]) for method in lambda_methods],
    "λ vs T5 final-epoch feature norm",
    "Mean feature L2 norm",
)
lambda_plot(
    "lambda_intrusion",
    [number(last_feature_row(method, 5)["intrusion_ratio"]) for method in lambda_methods],
    "λ vs T5 final-epoch intrusion",
    "Intrusion ratio",
)

# Common geometry plots use final-epoch summaries; bands are stored p10/p90,
# not pooled raw sample distributions.
geometry_methods = [BASE, "L03", "FCS_PIC", "E3_NORM", "E3_AHN", "CEOS50", "AM_DRIFT"]


def geometry_quantile_plot(name, key, title, ylabel):
    figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for method in geometry_methods:
        rows = [last_feature_row(method, task) for task in range(2, T + 1)]
        means = np.array([number(row.get(key + "_mean")) for row in rows])
        low = np.array([number(row.get(key + "_p10")) for row in rows])
        high = np.array([number(row.get(key + "_p90")) for row in rows])
        axis.plot(range(2, T + 1), means, marker="o", color=COLORS[method], label=LABELS[method])
        axis.fill_between(range(2, T + 1), low, high, color=COLORS[method], alpha=0.08)
    axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel=ylabel, title=UNIT + " · " + title)
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8)
    save_figure(figure, name)


geometry_quantile_plot("nearest_old_cosine", "nearest_old_cosine", "nearest-old cosine (mean, p10–p90)", "Cosine")
geometry_quantile_plot("cosine_gap", "cosine_gap", "correct-new minus nearest-old cosine", "Cosine gap")

rho_methods = ["L03", "E3_NORM", "AM_VARIANCE", "AM_AGE", "AM_DRIFT"]
figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
for method in rho_methods:
    rows = sorted(R[method]["followup_task_end"], key=lambda row: int(row["task_id"]))
    tasks = [int(row["task_id"]) for row in rows]
    means = np.array([number(row["nearest_rho_mean"]) for row in rows])
    low = np.array([number(row["nearest_rho_p10"]) for row in rows])
    high = np.array([number(row["nearest_rho_p90"]) for row in rows])
    axis.plot(tasks, means, marker="o", color=COLORS[method], label=LABELS[method])
    axis.fill_between(tasks, low, high, color=COLORS[method], alpha=0.1)
axis.axhline(1, color="#555555", linestyle="--", linewidth=1, label="boundary ρ=1")
axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel="ρ", title=UNIT + " · task-end rho (mean, p10–p90)")
axis.grid(alpha=0.2)
axis.legend(ncol=2, fontsize=8)
save_figure(figure, "rho_distribution")

figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
for method in ["L01", "L03", "L05", "L10", "E3_NORM", "E3_AHN", "AM_VARIANCE", "AM_AGE", "AM_DRIFT"]:
    values = [number(last_feature_row(method, task).get("intrusion_ratio")) for task in range(2, T + 1)]
    axis.plot(range(2, T + 1), values, marker="o", color=COLORS[method], label=LABELS[method])
axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel="Intrusion ratio", title=UNIT + " · final-epoch intrusion")
axis.grid(alpha=0.2)
axis.legend(ncol=3, fontsize=8)
save_figure(figure, "intrusion_by_task")


def gradient_task_mean(method, key, task):
    values = [
        number(row.get(key))
        for row in R[method]["gradient_metrics"]
        if int(row["task_id"]) == task and row.get(key) not in (None, "")
    ]
    return mean_finite(values)


gradient_specs = [
    ("L03", "grad_separation_norm", "grad_cosine_classification_separation"),
    ("FCS_PIC", "grad_pic_norm", "grad_cosine_classification_pic"),
    ("E3_AHN", "grad_ahn_norm", "grad_cosine_classification_ahn"),
    ("AM_DRIFT", "grad_separation_norm", "grad_cosine_classification_separation"),
]
figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
for method, norm_key, cosine_key in gradient_specs:
    tasks = range(2, T + 1)
    axes[0].plot(tasks, [gradient_task_mean(method, "grad_classification_norm", task) for task in tasks], linestyle="--", color=COLORS[method], alpha=0.65)
    axes[0].plot(tasks, [gradient_task_mean(method, norm_key, task) for task in tasks], marker="o", color=COLORS[method], label=LABELS[method] + " old-new")
    axes[1].plot(tasks, [gradient_task_mean(method, cosine_key, task) for task in tasks], marker="o", color=COLORS[method], label=LABELS[method])
axes[0].set(title="실선 old-new / 점선 classification", ylabel="Gradient L2 norm")
axes[1].set(title="cos(grad classification, grad old-new)", ylabel="Cosine")
axes[1].axhline(0, color="#777777", linewidth=0.8)
for axis in axes:
    axis.set(xticks=range(2, T + 1), xlabel="Task")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
figure.suptitle(UNIT + " · interval gradient diagnostics")
save_figure(figure, "gradient_norms")

figure, axis = plt.subplots(figsize=(8.5, 5), layout="constrained")
for method, _, cosine_key in gradient_specs:
    axis.plot(range(2, T + 1), [gradient_task_mean(method, cosine_key, task) for task in range(2, T + 1)], marker="o", color=COLORS[method], label=LABELS[method])
axis.axhline(0, color="#777777", linewidth=0.8)
axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel="Gradient cosine", title=UNIT + " · classification vs old-new gradient")
axis.grid(alpha=0.2)
axis.legend()
save_figure(figure, "gradient_cosine")


def final_ceos_row(method, task):
    rows = [row for row in R[method]["ceos_summary"] if int(row["task_id"]) == task]
    return max(rows, key=lambda row: int(row["epoch"]))


for figure_name, key, title, ylabel in (
    ("ceos_logit_gap", "ceos_old_new_logit_margin", "CEOS old-new logit gap", "Old − enemy logit"),
    ("ceos_boundary_proximity", "ceos_boundary_proximity", "CEOS boundary-proximity proxy", "|old − enemy logit|")
):
    figure, axis = plt.subplots(figsize=(7.5, 5), layout="constrained")
    for method in ("CEOS50", "CEOS100"):
        rows = [final_ceos_row(method, task) for task in range(2, T + 1)]
        means = np.array([number(row[key + "_mean"]) for row in rows])
        low = np.array([number(row[key + "_p10"]) for row in rows])
        high = np.array([number(row[key + "_p90"]) for row in rows])
        axis.plot(range(2, T + 1), means, marker="o", color=COLORS[method], label=LABELS[method])
        axis.fill_between(range(2, T + 1), low, high, color=COLORS[method], alpha=0.12)
    axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel=ylabel, title=UNIT + " · " + title + " (CA final epoch)")
    axis.grid(alpha=0.2)
    axis.legend()
    save_figure(figure, figure_name)


def adaptive_last_rows(method, task):
    rows = [row for row in R[method]["adaptive_margin"] if int(row["task_id"]) == task]
    epoch = max(int(row["epoch"]) for row in rows)
    return [row for row in rows if int(row["epoch"]) == epoch]


figure, axis = plt.subplots(figsize=(9, 5.2), layout="constrained")
for method in ("AM_VARIANCE", "AM_AGE", "AM_DRIFT"):
    means, lows, highs = [], [], []
    for task in range(2, T + 1):
        values = np.array([number(row["tau"]) for row in adaptive_last_rows(method, task)])
        means.append(float(values.mean()))
        lows.append(float(np.quantile(values, 0.1)))
        highs.append(float(np.quantile(values, 0.9)))
    axis.plot(range(2, T + 1), means, marker="o", color=COLORS[method], label=LABELS[method])
    axis.fill_between(range(2, T + 1), lows, highs, color=COLORS[method], alpha=0.1)
axis.set(xticks=range(2, T + 1), xlabel="Task", ylabel="Tau", title=UNIT + " · adaptive tau (class mean, p10–p90)")
axis.grid(alpha=0.2)
axis.legend()
save_figure(figure, "adaptive_tau_distribution")

figure, axis = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
for method in ("AM_VARIANCE", "AM_AGE", "AM_DRIFT"):
    rows = [row for row in R[method]["adaptive_class_diagnostics"] if int(row["task_id"]) == 5 and int(row["epoch"]) == 15]
    axis.scatter([number(row["tau"]) for row in rows], [number(row["intrusion_ratio"]) for row in rows], s=12, alpha=0.45, color=COLORS[method], label=LABELS[method])
axis.set(xlabel="Class tau", ylabel="Class intrusion ratio", title=UNIT + " · T5 tau vs intrusion (descriptive)")
axis.grid(alpha=0.2)
axis.legend()
save_figure(figure, "adaptive_tau_intrusion")

figure, axis = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
drift_rows = [row for row in R["AM_DRIFT"]["adaptive_class_diagnostics"] if int(row["task_id"]) == 5 and int(row["epoch"]) == 15]
scatter = axis.scatter([number(row["apr_shift_norm"]) for row in drift_rows], [number(row["adaptive_scale"]) for row in drift_rows], c=[number(row["intrusion_ratio"]) for row in drift_rows], cmap="viridis", s=18, alpha=0.7)
axis.set(xlabel="APR mean shift norm", ylabel="Adaptive tau scale", title=UNIT + " · AM-Drift T5")
axis.grid(alpha=0.2)
figure.colorbar(scatter, ax=axis, label="Class intrusion ratio")
save_figure(figure, "adaptive_drift_margin")

# Loss and resource figures supplement the twenty design figures.
selected_loss_methods = [BASE, "L03", "FCS_PIC", "E3_NORM", "E3_AHN", "AM_DRIFT"]
figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
for axis, key in zip(axes.flat, ("classification_raw", "scc_weighted", "separation_weighted", "total_loss")):
    for method in selected_loss_methods:
        rows = [row for row in R[method]["epoch_metrics"] if row["stage"] == "feature_training" and int(row["task_id"]) == 5 and row.get(key) not in (None, "")]
        if rows:
            axis.plot([int(row["epoch"]) for row in rows], [number(row[key]) for row in rows], color=COLORS[method], label=LABELS[method])
    axis.set(title=key, xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.2)
axes[0, 0].legend(ncol=2, fontsize=8)
figure.suptitle(UNIT + " · T5 feature training loss (목적함수 간 절대값 직접 비교 금지)")
save_figure(figure, "loss_task5")

figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
metric_lookup = {row["method"]: row for row in metrics}
for axis, key, title in zip(
    axes,
    ("wall_seconds", "peak_cuda_reserved_GiB", "final_statistics_MiB"),
    ("Run wall time (min)", "Peak CUDA reserved (GiB)", "Final statistics (MiB)"),
):
    values = [metric_lookup[method][key] / 60 if key == "wall_seconds" else metric_lookup[method][key] for method in METHODS]
    axis.bar([LABELS[method] for method in METHODS], values, color=[COLORS[method] for method in METHODS])
    axis.tick_params(axis="x", rotation=60)
    axis.set(title=title)
figure.suptitle(UNIT + " · resource records")
save_figure(figure, "resource_costs")

# ---------------------------------------------------------------------------
# Audited summaries and report text
# ---------------------------------------------------------------------------
inventory = []
for method in METHODS:
    inventory.append(
        {
            "method": method,
            "seed": SEED,
            "path": str(R[method]["run_dir"]),
            "status": R[method]["metadata"]["status"],
            "completed_tasks": R[method]["summary"]["completed_tasks"],
            "included": True,
            "reason": "common completed seed 1993",
        }
    )
    for missing_seed in (1996, 1997):
        run_dir = DATA / method / f"seed_{missing_seed}"
        inventory.append(
            {
                "method": method,
                "seed": missing_seed,
                "path": str(run_dir),
                "status": "missing" if not run_dir.exists() else "not included",
                "completed_tasks": 0,
                "included": False,
                "reason": "planned final seed; no completed result in this report",
            }
        )
write_csv("run_inventory.csv", inventory)

lambda_rows = []
for method, lambda_value in zip(lambda_methods, lambdas):
    lambda_rows.append(
        {
            "method": method,
            "lambda": lambda_value,
            "delta_Inc_pp": R[method]["Inc"] - base["Inc"],
            "delta_Last_pp": R[method]["Last"] - base["Last"],
            "delta_Forgetting_pp": R[method]["F"] - base["F"],
            "delta_final_old_pp": R[method]["old"][-1] - base["old"][-1],
            "delta_final_new_pp": R[method]["new"][-1] - base["new"][-1],
            "T5_feature_norm": number(last_feature_row(method, 5)["feature_norm_mean"]),
            "T5_intrusion": number(last_feature_row(method, 5)["intrusion_ratio"]),
        }
    )
write_csv("lambda_sweep.csv", lambda_rows)

numerical_rows = []
for method in METHODS:
    feature_rows = [
        row
        for row in R[method]["epoch_metrics"]
        if row["stage"] == "feature_training" and int(row["task_id"]) > 1
    ]
    active_components = []
    for component in ("separation", "pic", "ahn"):
        if any(truth(row.get(component + "_active")) for row in feature_rows):
            active_components.append(component)
    oldnew_gradient_key = None
    oldnew_cosine_key = None
    if "pic" in active_components:
        oldnew_gradient_key = "grad_pic_norm"
        oldnew_cosine_key = "grad_cosine_classification_pic"
    elif "ahn" in active_components:
        oldnew_gradient_key = "grad_ahn_norm"
        oldnew_cosine_key = "grad_cosine_classification_ahn"
    elif "separation" in active_components:
        oldnew_gradient_key = "grad_separation_norm"
        oldnew_cosine_key = "grad_cosine_classification_separation"
    gradient_values = (
        [number(row.get(oldnew_gradient_key)) for row in R[method]["gradient_metrics"] if row.get(oldnew_gradient_key) not in (None, "")]
        if oldnew_gradient_key
        else []
    )
    cosine_values = (
        [number(row.get(oldnew_cosine_key)) for row in R[method]["gradient_metrics"] if row.get(oldnew_cosine_key) not in (None, "")]
        if oldnew_cosine_key
        else []
    )
    final = last_feature_row(method, 5)
    numeric_keys = (
        "classification_raw",
        "scc_raw",
        "separation_raw",
        "pic_raw",
        "ahn_raw",
        "total_loss",
        "feature_norm_mean",
        "cosine_gap_mean",
        "intrusion_ratio",
    )
    nonfinite = 0
    for row in feature_rows:
        for key in numeric_keys:
            if row.get(key) not in (None, "") and not np.isfinite(number(row[key])):
                nonfinite += 1
    numerical_rows.append(
        {
            "method": method,
            "active_oldnew_components": "+".join(active_components) or "none",
            "T5_feature_norm_mean": number(final.get("feature_norm_mean")),
            "T5_feature_norm_p95": number(final.get("feature_norm_p95")),
            "T5_cosine_gap_mean": number(final.get("cosine_gap_mean")),
            "T5_intrusion_ratio": number(final.get("intrusion_ratio")),
            "mean_oldnew_gradient_norm": mean_finite(gradient_values),
            "mean_grad_cosine_cls_oldnew": mean_finite(cosine_values),
            "nan_inf_flags": sum(truth(row.get("nan_or_inf_detected")) for row in R[method]["epoch_metrics"]),
            "recorded_nonfinite_values": nonfinite,
        }
    )
write_csv("numerical_diagnostics.csv", numerical_rows)

ceos_rows = []
for method in ("CEOS50", "CEOS100"):
    for task in range(2, T + 1):
        rows = [row for row in R[method]["ceos_summary"] if int(row["task_id"]) == task]
        ceos_rows.append(
            {
                "method": method,
                "task": task,
                "epochs": len(rows),
                "ceos_fraction": number(rows[0]["ceos_replay_fraction"]),
                "total_per_class": int(rows[0]["total_samples_per_class"]),
                "ceos_per_old_class": int(rows[0]["ceos_samples_per_old_class"]),
                "gaussian_per_old_class": int(rows[0]["gaussian_samples_per_old_class"]),
                "old_dominance_rate": mean_finite([number(row["ceos_old_dominance_rate"]) for row in rows]),
                "predicted_as_source_old_rate": mean_finite([number(row["ceos_predicted_as_source_old_rate"]) for row in rows]),
                "old_new_logit_margin_mean": mean_finite([number(row["ceos_old_new_logit_margin_mean"]) for row in rows]),
                "boundary_proximity_mean": mean_finite([number(row["ceos_boundary_proximity_mean"]) for row in rows]),
            }
        )
write_csv("ceos_diagnostics.csv", ceos_rows)

adaptive_rows = []
for method in ("AM_VARIANCE", "AM_AGE", "AM_DRIFT"):
    for task in range(2, T + 1):
        rows = adaptive_last_rows(method, task)
        adaptive_rows.append(
            {
                "method": method,
                "task": task,
                "class_regions": len(rows),
                "scale_mean": mean_finite([number(row["adaptive_scale"]) for row in rows]),
                "scale_min": min(number(row["adaptive_scale"]) for row in rows),
                "scale_max": max(number(row["adaptive_scale"]) for row in rows),
                "tau_mean": mean_finite([number(row["tau"]) for row in rows]),
                "variance_mean": mean_finite([number(row["variance_mean"]) for row in rows]),
                "age_mean": mean_finite([number(row["class_age"]) for row in rows]),
                "apr_shift_mean": mean_finite([number(row["apr_shift_norm"]) for row in rows]),
            }
        )
write_csv("adaptive_summary.csv", adaptive_rows)

research_questions = [
    {
        "RQ": "RQ1",
        "question": "E3 제한은 아이디어 자체인가, loss 강도인가?",
        "finding": "lambda 효과가 비단조적이며 L01만 Inc-Acc가 소폭 양수. 모든 lambda가 final new accuracy를 낮춤.",
        "uncertainty": "seed 1개; 강도와 raw feature norm 반응이 함께 바뀜.",
    },
    {
        "RQ": "RQ2",
        "question": "PIC보다 covariance-aware E3가 유리한가?",
        "finding": "이 seed에서는 L03가 PIC보다 Inc/Last/Forgetting이 모두 높음. PIC는 new를 높이고 old를 낮춤.",
        "uncertainty": "구현은 FCS 전체가 아닌 adapted PIC component.",
    },
    {
        "RQ": "RQ3",
        "question": "E3가 feature norm을 키워 constraint를 회피하는가?",
        "finding": "lambda 증가와 함께 norm 증가, intrusion 감소가 선명하게 동반됨.",
        "uncertainty": "동반 관찰이며 인과 증명은 아님.",
    },
    {
        "RQ": "RQ4",
        "question": "AHN이 geometry를 안정화하는가?",
        "finding": "현재 AHN은 L03/B0보다 성능이 낮고 norm inflation도 유지되어 지지되지 않음.",
        "uncertainty": "lambda_ahn 및 margin sweep 없음.",
    },
    {
        "RQ": "RQ5",
        "question": "CEOS가 plasticity를 더 잘 보존하는가?",
        "finding": "new accuracy는 증가했으나 old/forgetting 손실이 더 커져 전체 성능은 저하.",
        "uncertainty": "Structure용 adapted CEOS이며 sample이 기록상 boundary에 가깝지 않음.",
    },
    {
        "RQ": "RQ6",
        "question": "adaptive tau가 trade-off를 개선하는가?",
        "finding": "AM-Drift만 Inc/Last/Forgetting을 동시에 관측 개선; variance는 forgetting 최저.",
        "uncertainty": "seed 1개이며 rule coefficient sweep 없음.",
    },
]
write_csv("research_questions.csv", research_questions)

metric_lookup = {row["method"]: row for row in metrics}
best_inc_method = max(METHODS, key=lambda method: R[method]["Inc"])
best_last_method = max(METHODS, key=lambda method: R[method]["Last"])
best_forgetting_method = min(METHODS, key=lambda method: R[method]["F"])

report_sections = []


def add(text):
    report_sections.append(text)


def picture(name, caption):
    add(
        f"![{name}](figures/{name}.png)\n\n{caption}  \n"
        f"[SVG 원본](figures/{name}.svg)"
    )


add(
    "# ImageNet-A 5-task: Structure E3 후속 실험 비교 보고서\n\n"
    "작성 기준: 동일 source snapshot에서 완료된 pilot seed 1993, 모든 비교 n=1. "
    "정확도 단위는 %, 차이와 forgetting은 percentage point(pp)다. 저장된 기록만 재집계했으며 "
    "추가 학습이나 checkpoint 재평가는 수행하지 않았다."
)

add(
    "## 1. 결과 요약\n\n"
    f"**이 pilot에서 Inc-Acc와 Last-Acc를 함께 가장 높인 방법은 {LABELS[best_inc_method]}**다. "
    f"B0 대비 Inc-Acc {format_value(R[best_inc_method]['Inc']-base['Inc'], True)} pp, "
    f"Last-Acc {format_value(R[best_inc_method]['Last']-base['Last'], True)} pp이며 "
    f"Forgetting도 {format_value(R[best_inc_method]['F']-base['F'], True)} pp 낮다. "
    f"가장 낮은 forgetting은 {LABELS[best_forgetting_method]}의 {R[best_forgetting_method]['F']:.2f} pp"
    f"(B0 대비 {format_value(R[best_forgetting_method]['F']-base['F'], True)} pp)다.\n\n"
    "그러나 old/new trade-off는 남는다. AM-Drift의 최종 old accuracy는 B0보다 "
    f"{format_value(R['AM_DRIFT']['old'][-1]-base['old'][-1], True)} pp 높지만 new accuracy는 "
    f"{format_value(R['AM_DRIFT']['new'][-1]-base['new'][-1], True)} pp 낮다. "
    "FCS PIC와 CEOS는 반대로 new accuracy를 높였지만 old 보존과 forgetting을 크게 악화했다. "
    "**n=1이므로 이 순위는 관측된 pilot 결과이지 평균 성능이나 통계적 유의성 결론이 아니다.**"
)

add(
    "## 2. 실험 목록과 비교 가능성\n\n"
    "13개 run은 모두 ImageNet-A 200 classes, 5 tasks×40 classes, seed 1993, single-head CIL이고 "
    "5/5 tasks를 완료했다. train 5,981장, test 1,519장, class order와 split content SHA-256이 "
    "완전히 일치한다. 모든 run의 source snapshot manifest SHA-256과 base working-tree digest도 "
    "동일하며, feature 학습 및 classifier retraining의 epoch·processed samples·optimizer steps·LR을 "
    "대조했다. CEOS도 class당 총 replay 256개를 유지한다."
)

inventory_table = []
for method in METHODS:
    config = R[method]["config"]
    inventory_table.append(
        [
            LABELS[method],
            IDEAS[method],
            "complete / 5",
            "1993 / 1",
            str(config.get("sep_lambda")),
            str(config.get("ceos_replay_fraction")),
            str(config.get("adaptive_margin_mode")),
        ]
    )
add(
    markdown_table(
        ["방법", "실제 변경", "상태/task", "seed/n", "sep λ", "CEOS 비율", "adaptive"],
        inventory_table,
    )
)
add(
    "모든 방법은 Multi-Center Off이고 K-means를 사용하지 않는다. 따라서 이전 E0–E7 보고서의 "
    "K-means 분할 분석은 이번 비교에 **미적용**이다. `B1_E3`는 L10과 동일 설정이라 중복 실행하지 "
    "않았다. 계획된 seed 1996·1997은 아직 없어 표본 SD와 paired Δ 분산은 N/A다."
)

add("## 3. 전체 성능")
performance_table = []
for method in METHODS:
    row = metric_lookup[method]
    performance_table.append(
        [
            LABELS[method],
            "1",
            format_value(row["Inc_Acc"]),
            format_value(row["delta_Inc_pp"], True),
            format_value(row["Last_Acc"]),
            format_value(row["delta_Last_pp"], True),
            format_value(row["Forgetting_pp"]),
            format_value(row["delta_Forgetting_pp"], True),
            format_value(row["final_old_accuracy"]),
            format_value(row["delta_final_old_pp"], True),
            format_value(row["final_new_accuracy"]),
            format_value(row["delta_final_new_pp"], True),
        ]
    )
add(
    markdown_table(
        [
            "방법",
            "n",
            "Inc",
            "ΔInc",
            "Last",
            "ΔLast",
            "F↓",
            "ΔF↓",
            "Old",
            "ΔOld",
            "New",
            "ΔNew",
        ],
        performance_table,
    )
)
add(
    "ΔInc/ΔLast/ΔOld/ΔNew는 양수가 관측 개선이고, ΔF는 음수가 개선이다. Last-Acc의 0.06583 pp는 "
    "최종 test set에서 정답 1개 차이에 해당하므로 작은 차이를 과해석하지 않는다. AM-Drift가 세 headline "
    "지표를 동시에 개선했지만 최종 new는 감소했다."
)
picture(
    "performance_deltas",
    "AM-Drift만 Inc-Acc와 Last-Acc가 모두 B0보다 높다. L01은 Inc만, L03·E3-Norm·AM-Variance·AM-Age는 Last만 높다. 초록/빨강은 관측 부호이며 유의성을 뜻하지 않는다.",
)
picture(
    "accuracy_by_task",
    "T1은 모든 방법이 동일하다. CEOS100은 T2부터 하락이 누적되고, AM-Drift는 T3–T5에서 B0보다 일관되게 높다. 선 간 작은 간격은 seed 변동이 없어 안정성을 판단할 수 없다.",
)
picture(
    "accuracy_delta_by_task",
    "0선 위가 같은 seed B0 대비 개선이다. FCS PIC는 T2–T3에서 소폭 높다가 T4–T5에 하락하고, CEOS 계열은 새 task 향상과 별개로 seen 전체 정확도가 낮다.",
)

add("## 4. 정확도 행렬 heatmap")
picture(
    "B0_absolute",
    "행은 학습 완료 task, 열은 평가 task다. 모든 값은 해당 시점의 모든 seen-class logits를 사용하는 single-head 정확도이며 미래 task 셀은 비워 두었다.",
)
picture(
    "heatmap_A_overview",
    "파란 대각선은 각 새 task를 막 학습한 절대 정확도이고 아래 삼각형은 B0 paired Δ다. CEOS·PIC의 대각선 개선과 과거 task 저하가 동시에 보여 plasticity/stability 교환을 확인할 수 있다.",
)
picture(
    "heatmap_B_overview",
    "대각선을 포함한 모든 유효 셀이 B0 paired Δ다. 모든 패널은 같은 대칭 색 범위를 사용하며 초록은 관측 개선, 빨강은 관측 저하다. 색은 통계적 유의성을 뜻하지 않는다.",
)
for method in METHODS[1:]:
    delta = R[method]["a"] - base["a"]
    valid = [(delta[t, i], t + 1, i + 1) for t in range(T) for i in range(t + 1)]
    minimum = min(valid)
    maximum = max(valid)
    add(
        f"### {LABELS[method]}\n\n"
        f"| 혼합 heatmap | 전체 paired Δ heatmap |\n|---|---|\n"
        f"| ![](figures/heatmap_A_{method}.png) | ![](figures/heatmap_B_{method}.png) |\n\n"
        f"전체 유효 셀 최대 변화는 T{maximum[1]}→T{maximum[2]}의 "
        f"{format_value(maximum[0], True)} pp, 최소 변화는 T{minimum[1]}→T{minimum[2]}의 "
        f"{format_value(minimum[0], True)} pp다. [A SVG](figures/heatmap_A_{method}.svg) · "
        f"[B SVG](figures/heatmap_B_{method}.svg)"
    )

add("## 5. Old/new와 forgetting")
picture(
    "old_new_accuracy",
    "PIC와 CEOS는 current-task accuracy를 높이는 대신 old accuracy를 낮춘다. diagonal E3 및 adaptive 계열은 대체로 old를 소폭 높이고 final new를 낮춘다. 이는 방법별 주효과 방향이 서로 반대임을 보여 주지만 원인을 확정하지는 않는다.",
)
picture(
    "final_task_accuracy",
    "최종 모델의 평가 task별 정확도다. 오른쪽 다섯 task Δ의 단순 평균은 표본 수가 서로 달라 ΔLast-Acc와 같지 않으며, Last는 correct/total로 가중 집계했다.",
)
picture(
    "forgetting_by_task",
    "AM-Variance의 평균 forgetting이 가장 낮고 AM-Drift·AM-Age·L03가 뒤를 잇는다. CEOS100은 과거 task 전체에서 forgetting이 크게 증가한다. forgetting은 낮을수록 좋고 음수 값을 임의로 0으로 자르지 않았다.",
)

add("## 6. Lambda sweep")
add(
    markdown_table(
        ["λ", "ΔInc", "ΔLast", "ΔF↓", "ΔOld", "ΔNew", "T5 norm", "T5 intrusion"],
        [
            [
                row["lambda"],
                format_value(row["delta_Inc_pp"], True),
                format_value(row["delta_Last_pp"], True),
                format_value(row["delta_Forgetting_pp"], True),
                format_value(row["delta_final_old_pp"], True),
                format_value(row["delta_final_new_pp"], True),
                format_value(row["T5_feature_norm"]),
                format_value(row["T5_intrusion"], digits=3),
            ]
            for row in lambda_rows
        ],
    )
)
for name, caption in (
    ("lambda_inc_acc", "Inc-Acc는 λ=.1에서만 B0보다 높고 이후 비단조적이다."),
    ("lambda_last_acc", "Last-Acc는 λ=.3과 1.0에서 소폭 높지만 정답 1–2개 규모다."),
    ("lambda_old_new_delta", "모든 λ에서 final old는 증가하고 new는 감소해 E3 방향의 trade-off가 반복된다."),
    ("lambda_feature_norm", "λ가 커질수록 T5 feature norm이 57.01→83.56으로 단조 증가한다."),
    ("lambda_intrusion", "λ가 커질수록 T5 intrusion이 0.802→0.353으로 감소한다. norm 증가와 동시에 일어나 인과 분리가 필요하다."),
):
    picture(name, caption)
add(
    "낮은 λ가 전체 trade-off를 자동으로 해결하지는 않았다. L01은 Inc가 +0.09 pp지만 final new가 −1.85 pp이며, "
    "L03은 Last와 forgetting이 더 좋지만 Inc는 −0.05 pp다. 따라서 이 seed에서 λ=.3을 후속 비교의 기준으로 "
    "사용한 것은 합리적이나 ‘best’로 확정할 근거는 부족하다."
)

add("## 7. PIC · E3-Norm · AHN과 geometry/gradient")
picture(
    "nearest_old_cosine",
    "선은 epoch 마지막 batch 집계가 아니라 저장된 epoch 전체 요약의 mean이고 음영은 p10–p90이다. PIC는 nearest-old cosine을 낮추지 않았음에도 new accuracy를 높였고 old accuracy는 낮아졌다.",
)
picture(
    "cosine_gap",
    "correct-current prototype과 nearest-old prototype의 cosine gap은 task가 진행될수록 모든 방법에서 줄어든다. PIC의 gap은 비교 방법보다 크지만 이것만으로 old 보존 성능을 설명하지 못한다.",
)
picture(
    "rho_distribution",
    "E3-Norm은 모든 task에서 rho가 1보다 훨씬 작고 intrusion이 1.0이어서 근사 normalized covariance와 birth tau가 제대로 보정되지 않았다. 성능은 B0에 가깝지만 이 ablation을 잘 보정된 normalized E3로 해석하면 안 된다.",
)
picture(
    "intrusion_by_task",
    "raw E3는 λ가 커질수록 intrusion이 줄지만 feature norm은 증가한다. AM-Age는 커진 tau 때문에 T5 intrusion이 1.0에 도달하고, AM-Drift는 tau를 완화해 낮은 intrusion과 비교적 작은 norm을 보인다.",
)
picture(
    "gradient_norms",
    "실선은 해당 old-new component, 점선은 classification gradient norm이다. 서로 목적함수 스케일이 다르므로 norm 크기만으로 우열을 판단하지 않는다. 기록은 configurable interval batch의 평균이다.",
)
picture(
    "gradient_cosine",
    "diagonal separation은 classification과 평균적으로 약한 음의 cosine을 보여 충돌 신호가 있다. PIC와 AHN은 양의 cosine이지만 최종 성능 개선으로 이어지지 않아 gradient 정렬만으로 충분하지 않다.",
)
add(
    f"FCS PIC는 ΔInc {format_value(R['FCS_PIC']['Inc']-base['Inc'], True)} pp, ΔLast "
    f"{format_value(R['FCS_PIC']['Last']-base['Last'], True)} pp, ΔF "
    f"{format_value(R['FCS_PIC']['F']-base['F'], True)} pp다. final new는 "
    f"{format_value(R['FCS_PIC']['new'][-1]-base['new'][-1], True)} pp지만 old는 "
    f"{format_value(R['FCS_PIC']['old'][-1]-base['old'][-1], True)} pp다. E3+AHN은 L03보다 "
    "Inc/Last가 낮고 T5 norm도 71.79로 높아 현재 설정에서는 norm 안정화 가설을 지지하지 않는다. "
    "E3-Norm의 T5 norm은 46.43으로 B0 45.67에 가깝지만 boundary calibration 실패 신호가 있다."
)

add("## 8. CEOS boundary replay")
add(
    markdown_table(
        ["방법", "ΔInc", "ΔLast", "ΔF↓", "ΔOld", "ΔNew", "old 예측률", "평균 logit gap"],
        [
            [
                LABELS[method],
                format_value(R[method]["Inc"] - base["Inc"], True),
                format_value(R[method]["Last"] - base["Last"], True),
                format_value(R[method]["F"] - base["F"], True),
                format_value(R[method]["old"][-1] - base["old"][-1], True),
                format_value(R[method]["new"][-1] - base["new"][-1], True),
                format_value(mean_finite([row["predicted_as_source_old_rate"] for row in ceos_rows if row["method"] == method]), digits=3),
                format_value(mean_finite([row["old_new_logit_margin_mean"] for row in ceos_rows if row["method"] == method])),
            ]
            for method in ("CEOS50", "CEOS100")
        ],
    )
)
picture(
    "ceos_logit_gap",
    "CEOS sample은 생성 시점 classifier에서 source old로 100% 예측되고 old−enemy logit gap도 약 8 수준이다. 즉 기록상 decision boundary에 가까운 hard sample이라기보다 old 쪽에 강하게 남아 있다.",
)
picture(
    "ceos_boundary_proximity",
    "gap이 모두 양수여서 이 실행에서는 |gap|과 raw gap의 평균이 같다. CEOS100의 gap이 CEOS50보다 약간 작지만 여전히 0에서 멀고, old test 보존은 오히려 크게 악화했다.",
)
add(
    "CEOS50/100 모두 class당 총 replay 256개와 5 CA epochs를 유지했으며 dominance constraint는 기록상 100%다. "
    "그럼에도 CEOS100은 final new +11.44 pp, old −8.73 pp, forgetting +13.39 pp로 심한 plasticity 편향을 보였다. "
    "현재 adapted CEOS는 ‘boundary-aware old 보존’ 가설을 지지하지 않으며, endpoint normalization·interpolation과 "
    "Structure classifier retraining의 결합을 다시 설계해야 한다."
)

add("## 9. Adaptive threshold")
add(
    markdown_table(
        ["방법", "T5 scale 평균[min,max]", "ΔInc", "ΔLast", "ΔF↓", "ΔOld", "ΔNew"],
        [
            [
                LABELS[method],
                f"{next(row for row in adaptive_rows if row['method']==method and row['task']==5)['scale_mean']:.3f}"
                f" [{next(row for row in adaptive_rows if row['method']==method and row['task']==5)['scale_min']:.3f},"
                f" {next(row for row in adaptive_rows if row['method']==method and row['task']==5)['scale_max']:.3f}]",
                format_value(R[method]["Inc"] - base["Inc"], True),
                format_value(R[method]["Last"] - base["Last"], True),
                format_value(R[method]["F"] - base["F"], True),
                format_value(R[method]["old"][-1] - base["old"][-1], True),
                format_value(R[method]["new"][-1] - base["new"][-1], True),
            ]
            for method in ("AM_VARIANCE", "AM_AGE", "AM_DRIFT")
        ],
    )
)
picture(
    "adaptive_tau_distribution",
    "AM-Age는 시간이 지날수록 tau를 크게 늘리고, AM-Drift는 약 0.8배로 완화한다. Variance rule은 평균 scale이 거의 1이라 주로 class 간 재배분 역할을 한다.",
)
picture(
    "adaptive_tau_intrusion",
    "T5 class별 tau와 intrusion의 기술적 산점도다. 같은 방법 안에서도 class별 분산이 크며, 이 상관을 tau의 인과 효과나 class 정확도 개선으로 해석하지 않는다.",
)
picture(
    "adaptive_drift_margin",
    "AM-Drift는 APR shift가 큰 class의 tau scale을 낮추도록 설계되었고 기록도 그 식을 반영한다. 색은 class intrusion이며, 실제 class accuracy와의 인과 연결은 추가 seed와 class-level paired 분석이 필요하다.",
)
add(
    "AM-Drift는 이 pilot에서 유일하게 Inc/Last/Forgetting을 동시에 개선하고 feature norm도 L03보다 낮췄다. "
    "AM-Variance는 headline 정확도는 낮지만 forgetting이 가장 낮다. AM-Age는 tau 증가와 intrusion 증가가 동반되어 "
    "현재 방향·강도는 지지되지 않는다. 세 규칙은 AMGC 재현이 아니라 E3용 독립 extension이다."
)

add("## 10. Loss · 수치 안정성 · 자원")
picture(
    "loss_task5",
    "T5 feature stage의 raw/weighted 항을 구분했다. PIC/AHN/E3-Norm은 목적함수 자체가 다르므로 total loss 절대값을 성능 순위로 사용하지 않는다. 모든 epoch 값은 batch-size weighted 평균이다.",
)
add(
    markdown_table(
        ["방법", "old-new 항", "T5 norm", "T5 p95 norm", "T5 gap", "T5 intrusion", "grad cos(cls,old-new)", "NaN/비유한"],
        [
            [
                LABELS[row["method"]],
                row["active_oldnew_components"],
                format_value(row["T5_feature_norm_mean"]),
                format_value(row["T5_feature_norm_p95"]),
                format_value(row["T5_cosine_gap_mean"], digits=3),
                format_value(row["T5_intrusion_ratio"], digits=3),
                format_value(row["mean_grad_cosine_cls_oldnew"], digits=3),
                f"{row['nan_inf_flags']}/{row['recorded_nonfinite_values']}",
            ]
            for row in numerical_rows
        ],
    )
)
add(
    "기록된 NaN/Inf flag와 검사한 비유한 수치는 모두 0이다. E3-Norm intrusion 1.0은 수치 폭주가 아니라 "
    "normalized covariance/tau calibration 문제이며, CEOS의 성능 저하는 replay sample 수 증가 때문이 아니다. "
    "gradient cosine은 설정된 interval batch에서만 측정했으므로 전체 학습의 정확한 적분값은 아니다."
)
picture(
    "resource_costs",
    "run wall time은 약 49–55분이고 peak CUDA reserved는 약 10.1 GiB로 비슷하다. Separation 계열은 old region covariance를 추가 저장해 final statistics가 약 451 MiB에서 902 MiB로 증가한다. wall time에는 common geometry/cache 진단이 포함되어 이전 보고서 E0 시간과 직접 비교하지 않는다.",
)
resource_table = []
for method in METHODS:
    row = metric_lookup[method]
    resource_table.append(
        [
            LABELS[method],
            hms(row["wall_seconds"]),
            f"{row['incremental_train_seconds']/60:.1f}",
            f"{row['peak_cuda_allocated_GiB']:.2f}",
            f"{row['peak_cuda_reserved_GiB']:.2f}",
            f"{row['final_statistics_MiB']:.1f}",
            f"{row['max_separation_cache_MiB']:.2f}",
        ]
    )
add(markdown_table(["방법", "wall", "train 분", "GPU alloc GiB", "GPU reserved GiB", "statistics MiB", "cache MiB"], resource_table))

add("## 11. 연구 질문별 답변")
for row in research_questions:
    add(
        f"### {row['RQ']}\n\n**질문:** {row['question']}  \n"
        f"**관측 결론:** {row['finding']}  \n"
        f"**불확실성:** {row['uncertainty']}"
    )

add(
    "## 12. 한계와 다음 실행\n\n"
    "가장 큰 한계는 seed 1993 하나뿐이라는 점이다. 현재 차이 상당수는 최종 정답 1–몇 개 규모여서 seed 1996·1997의 "
    "같은-seed paired Δ 없이 일반화할 수 없다. 다음 3-seed 우선순위는 AM-Drift, L03, E3-Norm, B0이며, "
    "CEOS100은 현 설정 그대로 확대하기보다 boundary proximity가 실제로 작아지도록 sampler를 수정한 뒤 별도 ablation하는 편이 "
    "타당하다. AHN은 λ_ahn/margin sweep 전에는 기각이 아니라 ‘현재 설정에서 미지지’로 남긴다.\n\n"
    "권장 순서: (1) B0/L03/AM-Drift/E3-Norm을 seed 1996·1997로 확인, (2) 공통 seed mean±sample SD 및 "
    "paired Δ SD 계산, (3) E3-Norm tau를 unit-sphere covariance에 맞게 다시 보정, (4) CEOS logit gap이 0 근처가 되도록 "
    "boundary sampler를 재검토, (5) 필요할 때 PIC/AHN coefficient sweep을 수행한다."
)

add(
    "## 13. 부록: 정의·원자료·재생성\n\n"
    "`a(t,i)`는 task t 학습과 classifier retraining을 완료한 모델을 task i test samples에 평가한 single-head 정확도다. "
    "`A(t)=100×Σ correct/Σ samples`, `Inc-Acc=mean_t A(t)`, `Last-Acc=A(T)`, "
    "`F_i=max_{u=i..T−1}a(u,i)−a(T,i)`, `Forgetting=mean_{i<T}F_i`다. "
    "paired Δ는 같은 seed·같은 셀에서 method−B0로 계산했고 forgetting은 낮을수록 좋다."
)
add(
    markdown_table(
        ["자료", "파일"],
        [
            ["seed별 핵심 성능·자원", "[metrics_per_seed.csv](tables/metrics_per_seed.csv)"],
            ["task별 A/old/new", "[task_metrics_recomputed.csv](tables/task_metrics_recomputed.csv)"],
            ["정확도 행렬 cell", "[cells_per_seed_and_sd.csv](tables/cells_per_seed_and_sd.csv)"],
            ["task별 forgetting", "[forgetting_per_task.csv](tables/forgetting_per_task.csv)"],
            ["class별 최종 paired Δ", "[class_final_deltas.csv](tables/class_final_deltas.csv)"],
            ["lambda sweep", "[lambda_sweep.csv](tables/lambda_sweep.csv)"],
            ["gradient·수치 진단", "[numerical_diagnostics.csv](tables/numerical_diagnostics.csv)"],
            ["CEOS 진단", "[ceos_diagnostics.csv](tables/ceos_diagnostics.csv)"],
            ["adaptive 요약", "[adaptive_summary.csv](tables/adaptive_summary.csv)"],
            ["연구 질문 요약", "[research_questions.csv](tables/research_questions.csv)"],
            ["검산", "[validation_checks.csv](tables/validation_checks.csv)"],
            ["출처·해시·정의", "[metadata.json](metadata.json)"],
        ],
    )
)
add(
    "각 방법의 절대/차이 행렬은 `tables/<METHOD>_absolute_matrix.csv`와 "
    "`tables/<METHOD>_delta_B0_matrix.csv`에 있다. raw task/class/epoch/gradient/adaptive/CEOS CSV도 `tables/raw_*`로 "
    "합쳤다. 대용량 CEOS per-sample CSV는 원본 run에 보존하고 보고서에는 저장된 epoch quantile summary만 사용했다."
)
add(
    "재생성 명령:\n\n"
    "```bash\n"
    "source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh\n"
    "conda activate cds\n"
    f"python {OUT / 'build_report.py'} \\\n"
    f"  --project-root {ROOT} \\\n"
    f"  --output-dir {OUT}\n"
    "```\n\n"
    "재생성에는 numpy와 matplotlib만 필요하며 GPU를 사용하지 않는다."
)

report = "\n\n".join(report_sections) + "\n"
for target in re.findall(r"\]\(([^)]+)\)", report):
    if not target.startswith(("http", "#")) and target != "metadata.json":
        validate("report link " + target, (OUT / target).exists())

write_csv("validation_checks.csv", checks)
(OUT / "report.md").write_text(report, encoding="utf-8")

metadata = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_root": str(DATA),
    "report_scope": "fixed seed-1993 pilot snapshot; seeds 1996/1997 missing",
    "used_seed": SEED,
    "n": 1,
    "sample_sd": "N/A",
    "runs": [str(R[method]["run_dir"]) for method in METHODS],
    "dataset": "imageneta",
    "num_classes": 200,
    "num_tasks": T,
    "protocol_manifest": protocol,
    "same_source_snapshot": True,
    "source_snapshot_manifest_sha256": next(iter(snapshot_hashes)),
    "base_working_tree_sha256": next(iter(source_hashes)),
    "evaluation_stage": "task_final after classifier retraining",
    "single_head": True,
    "checkpoint_selection": "fixed final task; no best-test selection",
    "training_performed_for_report": False,
    "checkpoint_re_evaluation_performed": False,
    "units": {
        "accuracy": "percent",
        "delta": "percentage point",
        "forgetting": "percentage point",
        "intrusion": "ratio 0..1",
        "MiB": 2**20,
        "GiB": 2**30,
    },
    "definitions": {
        "A": "100*sum(correct)/sum(samples)",
        "Inc": "mean_t A(t)",
        "Last": "A(T)",
        "Forgetting": "mean_i(max_{u=i..T-1} a(u,i)-a(T,i)); no clipping",
        "paired_delta": "same seed and same cell: method minus B0",
        "SD": "sample SD ddof=1 when n>=2; N/A here",
    },
    "delta_heatmap_range_pp": [-delta_limit, delta_limit],
    "zero_tolerance_pp": EPS,
    "missing_valid_cells": "N/A with hatch",
    "future_cells": "blank white",
    "method_labels": LABELS,
    "method_ideas": IDEAS,
    "config_differences": config_differences,
    "inventory": inventory,
    "source_files": sources,
    "validation_checks_passed": len(checks),
    "validation_failures": 0,
    "software": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
    "figure_count_png": len(list((OUT / "figures").glob("*.png"))),
    "table_count_csv": len(list((OUT / "tables").glob("*.csv"))),
}
(OUT / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(
    json.dumps(
        {
            "report": str(OUT / "report.md"),
            "figures": metadata["figure_count_png"],
            "tables": metadata["table_count_csv"],
            "checks": len(checks),
            "delta_range_pp": [-delta_limit, delta_limit],
        },
        ensure_ascii=False,
    )
)
