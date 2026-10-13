from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from statistics import mean

from analyze import matrix_markdown, save_heatmap
from efcil_diag.task_matrix import parse_lower_triangular_matrix, summarize_cil_matrix
from efcil_diag.utils import load_config


KEEP_METHODS = [
    "vanilla_lora",
    "keep_lora_wp_only",
    "keep_lora_m_only",
    "keep_lora_full",
]
DISPLAY_NAMES = {
    "vanilla_lora": "Vanilla LoRA (matched control)",
    "keep_lora_wp_only": "KeepLoRA Wp-only",
    "keep_lora_m_only": "KeepLoRA M-only",
    "keep_lora_full": "Full KeepLoRA",
    "pcds_full": "PCDS",
}
GENERATED_BEGIN = "<!-- BEGIN GENERATED PCDS COMPARISON -->"
GENERATED_END = "<!-- END GENERATED PCDS COMPARISON -->"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def percent(number: float) -> str:
    return "N/A" if not math.isfinite(number) else f"{100.0 * number:.2f}"


def signed_points(number: float) -> str:
    return f"{100.0 * number:+.2f}%p"


def pcds_summary(method_dir: Path, num_tasks: int) -> dict:
    tasks = read_rows(method_dir / "task_metrics.csv")
    statistics = read_rows(method_dir / "pcds_statistics.csv")
    if len(tasks) != num_tasks:
        raise RuntimeError(
            f"{method_dir}: expected {num_tasks} completed tasks, found {len(tasks)}"
        )
    if len(statistics) != num_tasks:
        raise RuntimeError(
            f"{method_dir}: expected {num_tasks} PCDS statistic rows, "
            f"found {len(statistics)}"
        )
    final = tasks[-1]
    return {
        "method": "pcds_full",
        "final_accuracy": value(final, "seen_accuracy"),
        "average_incremental_accuracy": value(
            final, "average_incremental_accuracy"
        ),
        "final_old_accuracy": value(final, "old_accuracy"),
        "final_new_accuracy": value(final, "new_accuracy"),
        "average_forgetting": value(final, "average_forgetting"),
        "peak_cuda_allocated_gb": max(
            value(row, "peak_cuda_allocated_gb") for row in tasks
        ),
        "optimizer_skipped_steps": sum(
            int(float(row["optimizer_skipped_steps"])) for row in statistics
        ),
    }


def load_keep_summaries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in KEEP_METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"KeepLoRA summary is missing methods: {missing}")
    return [by_method[method] for method in KEEP_METHODS]


def validate_pcds_matrix(
    matrix: list[list[float]],
    tasks: list[dict[str, str]],
    classes: list[dict[str, str]],
    classes_per_task: int,
) -> dict:
    matrix_summary = summarize_cil_matrix(matrix)
    errors: list[float] = []
    equal_test_counts = True
    expected_seen: list[int] = []
    observed_seen: list[int] = []
    for stage, matrix_row in enumerate(matrix):
        stage_classes = [row for row in classes if int(row["task_id"]) == stage]
        task_counts: list[int] = []
        for task_id, matrix_accuracy in enumerate(matrix_row):
            selected = [
                row for row in stage_classes if int(row["learned_task"]) == task_id
            ]
            correct = sum(int(row["correct"]) for row in selected)
            count = sum(int(row["count"]) for row in selected)
            if count == 0:
                raise RuntimeError(f"No class rows for A[{stage},{task_id}]")
            task_counts.append(count)
            errors.append(abs(correct / count - matrix_accuracy))
        equal_test_counts &= len(set(task_counts)) == 1
        errors.append(
            abs(matrix_summary.row_averages[stage] - value(tasks[stage], "seen_accuracy"))
        )
        expected_seen.append((stage + 1) * classes_per_task)
        observed_seen.append(int(tasks[stage]["seen_classes"]))
    maximum_error = max(errors, default=0.0)
    passed = (
        maximum_error <= 1.0e-10
        and equal_test_counts
        and observed_seen == expected_seen
    )
    return {
        "future_cells_are_empty": True,
        "equal_test_samples_per_task": equal_test_counts,
        "maximum_accuracy_consistency_error": maximum_error,
        "seen_classifier_class_counts": observed_seen,
        "expected_seen_classifier_class_counts": expected_seen,
        "passed": passed,
    }


def comparison_word(delta: float) -> str:
    if abs(delta) < 5.0e-13:
        return "같았다"
    return f"{abs(100.0 * delta):.2f}%p {'높았다' if delta > 0 else '낮았다'}"


def build_generated_report(
    config: dict,
    keep_config: dict,
    keep_rows: list[dict],
    pcds: dict,
    matrix: list[list[float]],
    keep_full_matrix: list[list[float]],
    statistics: list[dict[str, str]],
    ca_rows: list[dict[str, str]],
    metadata: dict,
) -> str:
    by_method = {row["method"]: row for row in [*keep_rows, pcds]}
    vanilla = by_method["vanilla_lora"]
    full = by_method["keep_lora_full"]
    final_ca = ca_rows[-1]
    applied_ca = [row for row in ca_rows if int(row["ca_applied"]) == 1]
    mean_ca_seen_delta = mean(
        value(row, "seen_accuracy_ca_delta") for row in applied_ca
    )
    final_gap_before = value(final_ca, "new_accuracy_before_ca") - value(
        final_ca, "old_accuracy_before_ca"
    )
    final_gap_after = value(final_ca, "new_accuracy_after_ca") - value(
        final_ca, "old_accuracy_after_ca"
    )
    final_cells = matrix[-1]
    keep_final_cells = keep_full_matrix[-1]
    middle_delta = mean(
        final_cells[index] - keep_final_cells[index] for index in (3, 4, 5)
    )
    keep_lora_feature_params = (
        12
        * len(keep_config["lora_targets"])
        * 768
        * int(keep_config["lora_rank"])
    )
    pcds_feature_params = int(metadata["adapter_trainable_parameters"])
    pcds_vs_vanilla = pcds["final_accuracy"] - vanilla["final_accuracy"]
    pcds_vs_full = pcds["final_accuracy"] - full["final_accuracy"]
    pcds_old_vs_vanilla = pcds["final_old_accuracy"] - vanilla["final_old_accuracy"]
    pcds_old_vs_full = pcds["final_old_accuracy"] - full["final_old_accuracy"]
    pcds_forgetting_vs_vanilla = (
        pcds["average_forgetting"] - vanilla["average_forgetting"]
    )
    pcds_forgetting_vs_full = (
        pcds["average_forgetting"] - full["average_forgetting"]
    )
    pcds_avg_vs_full = (
        pcds["average_incremental_accuracy"]
        - full["average_incremental_accuracy"]
    )

    lines = [
        GENERATED_BEGIN,
        "",
        "## PCDS Experiment",
        "",
        "이 절의 결과는 원 논문의 공식 CIFAR-100 재현값이 아니라, 동일한 CIFAR-100 "
        "single-head EFCIL protocol에 PCDS 방법론을 적용한 `PCDS CIFAR-100 adaptation`이다.",
        "",
        "### Method",
        "",
        "PCDS는 새 task 학습으로 발생한 feature-space migration을 두 구조로 다룬다. "
        "APR은 새 class의 학습 전/후 center 이동으로 old mean과 covariance를 보정하고, "
        "SCC는 현재 sample과 old prototype 사이의 cosine-relation vector를 직전 frozen "
        "model과 일치시킨다. 이후 Eq. (11)의 class-balanced Gaussian synthetic feature로 "
        "cosine classifier만 adaptation한다. Old raw train image는 다시 읽거나 저장하지 않는다.",
        "",
        "Released CDS 코드의 FFN-output nonlinear bottleneck adapter를 동일 timm ViT에 "
        "이식했다. Released covariance helper의 heuristic 대신, 실험 지시서에 명시된 "
        "paper Eq. (4)-(8)의 `rho * Delta_k Delta_k^T`를 그대로 사용했다.",
        "",
        "### Hyperparameters",
        "",
        "| Item | Value | Source |",
        "|---|---:|---|",
        f"| Adapter bottleneck | {config['adapter_bottleneck']} | released CDS config |",
        f"| SCC lambda | {config['scc_weight']} | released CDS config |",
        f"| APR gamma | {config['apr_gamma']} | released CDS config |",
        f"| APR rho | {config['apr_rho']} | released CDS config |",
        f"| Prototype/APR epsilon | {config['prototype_covariance_epsilon']} / {config['apr_epsilon']} | released implementation |",
        f"| Cosine scale tau | {config['classifier_scale']} | released CDS config |",
        f"| Optimizer | SGD, lr={config['learning_rate']}, momentum={config['momentum']}, wd={config['weight_decay']} | released CDS config |",
        f"| Scheduler / batch / epochs | cosine / {config['batch_size']} / {config['epochs_per_task']} per task | released config + matched budget |",
        f"| Classifier adaptation | {config['ca_epochs']} epochs, {config['ca_samples_per_class']} features/class/epoch | released implementation |",
        f"| Feature adaptation params | {pcds_feature_params:,} | runtime model metadata |",
        "",
        "CIFAR-100에 맞춘 hyperparameter search는 수행하지 않았다.",
        "",
        "### Main Results",
        "",
        "| Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |",
        "|---:|---:|---:|---:|---:|",
        f"| {percent(pcds['average_incremental_accuracy'])} | {percent(pcds['final_accuracy'])} | {percent(pcds['final_old_accuracy'])} | {percent(pcds['final_new_accuracy'])} | {percent(pcds['average_forgetting'])} |",
        "",
        "### Task-wise Accuracy Matrix",
        "",
        "각 `A[t,j]`는 Task t까지 학습한 하나의 expanding head로 Task j sample을 "
        "평가한다. Task-local masking이나 Task-ID는 사용하지 않는다.",
        "",
        *matrix_markdown(matrix),
        "",
        "Heatmap: [`task_matrix_pcds_full.png`](task_matrix_pcds_full.png)",
        "",
        "### APR Statistics",
        "",
        "| Task | Applied | mean ||delta_j|| | mean ||Delta_k|| | max ||Delta_k|| | Cov trace before | Cov trace after |",
        "|---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        lines.append(
            f"| {int(row['task_id'])} | {'O' if int(row['apr_applied']) else 'X'} | "
            f"{value(row, 'mean_new_displacement_norm'):.6f} | "
            f"{value(row, 'mean_apr_correction_norm'):.6f} | "
            f"{value(row, 'max_apr_correction_norm'):.6f} | "
            f"{value(row, 'mean_covariance_trace_before'):.6f} | "
            f"{value(row, 'mean_covariance_trace_after'):.6f} |"
        )
    lines.extend([
        "",
        "Task 0에는 old prototype이 없어 APR을 적용하지 않는다.",
        "",
        "### SCC Statistics",
        "",
        "| Task | SCC | Mean CE | Final CE | Mean SCC | Final SCC | Mean Total | Final Total |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in statistics:
        enabled = int(row["scc_enabled"]) == 1
        lines.append(
            f"| {int(row['task_id'])} | {'O' if enabled else 'X'} | "
            f"{value(row, 'mean_ce_loss'):.6f} | {value(row, 'final_ce_loss'):.6f} | "
            f"{value(row, 'mean_scc_loss'):.6f} | {value(row, 'final_scc_loss'):.6f} | "
            f"{value(row, 'mean_total_loss'):.6f} | {value(row, 'final_total_loss'):.6f} |"
        )
    lines.extend([
        "",
        "### Classifier Adaptation Analysis",
        "",
        "| Task | CA | Seen before | Seen after | Delta | Old before | Old after | New before | New after |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ca_rows:
        lines.append(
            f"| {int(row['task_id'])} | {'O' if int(row['ca_applied']) else 'X'} | "
            f"{percent(value(row, 'seen_accuracy_before_ca'))} | "
            f"{percent(value(row, 'seen_accuracy_after_ca'))} | "
            f"{signed_points(value(row, 'seen_accuracy_ca_delta'))} | "
            f"{percent(value(row, 'old_accuracy_before_ca'))} | "
            f"{percent(value(row, 'old_accuracy_after_ca'))} | "
            f"{percent(value(row, 'new_accuracy_before_ca'))} | "
            f"{percent(value(row, 'new_accuracy_after_ca'))} |"
        )
    lines.extend([
        "",
        f"Task 1-9에서 CA의 mean Seen-accuracy 변화는 {signed_points(mean_ca_seen_delta)}였다. "
        f"마지막 task의 old/new 격차는 CA 전 {100.0 * final_gap_before:.2f}%p에서 "
        f"CA 후 {100.0 * final_gap_after:.2f}%p로 변했다.",
        "",
        "## Combined Comparison",
        "",
        "### Combined CIFAR-100 Single-Head EFCIL Results",
        "",
        "| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for method in [*KEEP_METHODS, "pcds_full"]:
        row = by_method[method]
        lines.append(
            f"| {DISPLAY_NAMES[method]} | "
            f"{percent(row['average_incremental_accuracy'])} | "
            f"{percent(row['final_accuracy'])} | "
            f"{percent(row['final_old_accuracy'])} | "
            f"{percent(row['final_new_accuracy'])} | "
            f"{percent(row['average_forgetting'])} |"
        )
    lines.extend([
        "",
        "### Old vs New Accuracy and Forgetting",
        "",
        f"PCDS final accuracy는 Vanilla보다 {comparison_word(pcds_vs_vanilla)}, "
        f"Full KeepLoRA보다 {comparison_word(pcds_vs_full)}. Old accuracy는 각각 "
        f"{comparison_word(pcds_old_vs_vanilla)}, {comparison_word(pcds_old_vs_full)}. "
        f"Forgetting은 Vanilla 대비 {signed_points(pcds_forgetting_vs_vanilla)}, "
        f"Full KeepLoRA 대비 {signed_points(pcds_forgetting_vs_full)} 변했다. "
        f"Average Incremental Accuracy는 Full KeepLoRA 대비 {signed_points(pcds_avg_vs_full)}였다.",
        "",
        "### Task-wise Comparison",
        "",
        "마지막 model의 Task 3-5 accuracy는 다음과 같다.",
        "",
        "| Evaluated task | Full KeepLoRA | PCDS | PCDS - Full |",
        "|---:|---:|---:|---:|",
    ])
    for task_id in (3, 4, 5):
        lines.append(
            f"| {task_id} | {percent(keep_final_cells[task_id])} | "
            f"{percent(final_cells[task_id])} | "
            f"{signed_points(final_cells[task_id] - keep_final_cells[task_id])} |"
        )
    lines.extend([
        "",
        f"세 task 평균 차이는 {signed_points(middle_delta)}였다.",
        "",
        "### Methodological Differences",
        "",
        "| Method | Raw Replay | Synthetic Feature Replay | Task ID | PTM Protection | Previous-task Subspace | APR | SCC |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        "| Vanilla | X | X | X | X | X | X | X |",
        "| Wp-only | X | X | X | O | X | X | X |",
        "| M-only | X | X | X | X | O | X | X |",
        "| Full KeepLoRA | X | X | X | O | O | X | X |",
        "| PCDS | X | O (classifier only) | X | X | X | O | O |",
        "",
        "PCDS synthetic replay는 저장된 prototype Gaussian에서 매 CA epoch 생성되며 "
        "classifier만 최적화한다. Image나 per-sample old feature replay가 아니다.",
        "",
        f"KeepLoRA의 학습 feature-adaptation parameter는 task당 {keep_lora_feature_params:,}개이고, "
        f"PCDS FFN adapter는 {pcds_feature_params:,}개다. Classifier row parameter는 task에 "
        "따라 추가되므로 이 두 숫자에는 포함하지 않았다.",
        "",
        "## Interpretation",
        "",
        "### Q1. PCDS가 Vanilla보다 forgetting을 줄이는가?",
        "",
        f"Average forgetting은 Vanilla {percent(vanilla['average_forgetting'])}, "
        f"PCDS {percent(pcds['average_forgetting'])}로, PCDS가 "
        f"Vanilla보다 {comparison_word(pcds_forgetting_vs_vanilla)}. "
        "방향과 크기만 기술하며 seed 0에서 통계적 우월성을 주장하지 않는다.",
        "",
        "### Q2. PCDS old accuracy가 KeepLoRA보다 높은가?",
        "",
        f"PCDS old accuracy {percent(pcds['final_old_accuracy'])}는 Full KeepLoRA "
        f"{percent(full['final_old_accuracy'])}보다 {comparison_word(pcds_old_vs_full)}.",
        "",
        "### Q3. Synthetic classifier adaptation이 recency bias를 줄였는가?",
        "",
        f"Final-task old/new gap은 CA 전 {100.0 * final_gap_before:.2f}%p, CA 후 "
        f"{100.0 * final_gap_after:.2f}%p였다. 따라서 gap 변화는 "
        f"{signed_points(final_gap_after - final_gap_before)}다.",
        "",
        "### Q4. Drift prevention과 structure rectification 비교",
        "",
        f"Full KeepLoRA/PCDS final accuracy는 {percent(full['final_accuracy'])}/"
        f"{percent(pcds['final_accuracy'])}이며 PCDS가 {comparison_word(pcds_vs_full)}. "
        "KeepLoRA는 update 방향 자체를 보호하고 PCDS는 SCC로 relation을 제약하면서 "
        "APR와 synthetic CA로 drift 결과를 보정한다. CA 및 parameter-count 차이가 있어 "
        "이 값만으로 한 핵심 알고리즘의 순수 효과를 분리할 수 없다.",
        "",
        "### Q5. KeepLoRA에서 무너진 Task 3-5가 회복되는가?",
        "",
        f"Final Task 3-5 평균에서 PCDS는 Full KeepLoRA 대비 {signed_points(middle_delta)}였다. "
        "위 task-wise 표가 class group별 회복 여부의 직접 근거다.",
        "",
        "## Limitations",
        "",
        "- PCDS 원 논문은 이 정확한 CIFAR-100 protocol의 결과를 보고하지 않았다. 본 결과는 PCDS methodology의 CIFAR-100 adaptation이다.",
        "",
        "- 현재 비교는 seed 0 하나뿐이므로 통계적 결론이 아니다.",
        "",
        "- PCDS는 classifier adaptation에 synthetic Gaussian feature를 사용하지만 KeepLoRA run은 synthetic replay를 사용하지 않았다.",
        "",
        f"- Feature adaptation parameter 수가 KeepLoRA {keep_lora_feature_params:,}, PCDS {pcds_feature_params:,}로 다르다.",
        "",
        "- Optimizer와 adaptation module도 각 방법의 released setting을 따르므로 final accuracy만으로 모든 방법론적 요인을 분리하지 못한다.",
        "",
        "- PCDS는 released CDS FFN adapter를 사용했다. 따라서 논문 본문의 LoRA 표기와 released implementation 사이의 차이를 포함한 implementation adaptation이다.",
        "",
        f"- 최대 CUDA allocated memory는 {pcds['peak_cuda_allocated_gb']:.2f} GiB였고 optimizer skip은 {pcds['optimizer_skipped_steps']}회였다.",
        "",
        GENERATED_END,
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PCDS and merge KeepLoRA results")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    seed = int(config["seed"])
    num_tasks = int(config["num_tasks"])
    method_dir = (
        Path(config["output_root"])
        / str(config["experiment_name"])
        / f"seed_{seed}"
        / "pcds_full"
    )
    if not (method_dir / "COMPLETED.json").exists():
        raise RuntimeError(f"PCDS run is incomplete: {method_dir}")
    report_dir = method_dir.parent / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_rows(method_dir / "task_metrics.csv")
    classes = read_rows(method_dir / "class_metrics.csv")
    statistics = read_rows(method_dir / "pcds_statistics.csv")
    ca_rows = read_rows(method_dir / "classifier_adaptation.csv")
    matrix = parse_lower_triangular_matrix(
        read_rows(method_dir / "task_accuracy_matrix.csv"), num_tasks
    )
    check = validate_pcds_matrix(
        matrix, tasks, classes, int(config["classes_per_task"])
    )
    if not check["passed"]:
        raise RuntimeError(f"PCDS task-matrix consistency failure: {check}")

    keep_summary_path = repo_root / str(config["keep_lora_summary"])
    keep_rows = load_keep_summaries(keep_summary_path)
    keep_config = load_config(repo_root / str(config["keep_lora_config"]))
    keep_seed_root = repo_root / str(config["keep_lora_seed_root"])
    keep_full_matrix = parse_lower_triangular_matrix(
        read_rows(keep_seed_root / "keep_lora_full" / "accuracy_matrix.csv"),
        num_tasks,
    )
    with (method_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    pcds = pcds_summary(method_dir, num_tasks)

    combined = [*keep_rows, pcds]
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, allow_nan=True)
    write_rows(report_dir / "combined_summary.csv", combined)
    with (report_dir / "task_matrix_consistency.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(check, handle, indent=2)
    pcds_heatmap = report_dir / "task_matrix_pcds_full.png"
    save_heatmap(pcds_heatmap, matrix, "PCDS")

    generated = build_generated_report(
        config,
        keep_config,
        keep_rows,
        pcds,
        matrix,
        keep_full_matrix,
        statistics,
        ca_rows,
        metadata,
    )
    keep_results_path = repo_root / str(config["keep_lora_results"])
    keep_report_dir = keep_results_path.parent
    shutil.copy2(pcds_heatmap, keep_report_dir / pcds_heatmap.name)
    for method in KEEP_METHODS:
        source = keep_report_dir / f"task_matrix_{method}.png"
        if source.exists():
            shutil.copy2(source, report_dir / source.name)
    existing = keep_results_path.read_text(encoding="utf-8")
    base = existing.split(GENERATED_BEGIN, maxsplit=1)[0].rstrip()
    merged = base + "\n\n" + generated + "\n"
    # The completed KeepLoRA document is extended in place and mirrored under
    # PCDS outputs. The marker makes repeated analysis idempotent.
    keep_results_path.write_text(merged, encoding="utf-8")
    (report_dir / "RESULTS.md").write_text(merged, encoding="utf-8")
    print(f"Wrote combined report: {keep_results_path}")
    print(f"Mirrored report: {report_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
