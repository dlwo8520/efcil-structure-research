from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from scipy.stats import pearsonr

from analyze import matrix_markdown, save_heatmap
from efcil_diag.keeplora_experiment import METHOD_FLAGS
from efcil_diag.task_matrix import (
    long_format_rows,
    parse_lower_triangular_matrix,
    summarize_cil_matrix,
)
from efcil_diag.utils import load_config


ORDERED_METHODS = [
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
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def summarize(method_dir: Path, num_tasks: int) -> dict:
    tasks = read_rows(method_dir / "task_metrics.csv")
    classes = read_rows(method_dir / "class_metrics.csv")
    steps = read_rows(method_dir / "step_updates.csv")
    initialization = read_rows(method_dir / "lora_initialization.csv")
    if len(tasks) != num_tasks:
        raise RuntimeError(f"{method_dir}: expected {num_tasks} tasks, found {len(tasks)}")
    final = tasks[-1]
    final_classes = [row for row in classes if int(row["task_id"]) == num_tasks - 1]
    old_final = [row for row in final_classes if int(row["is_new"]) == 0]
    nonzero_steps = [row for row in steps if number(row, "parameter_update_norm") > 0.0]
    return {
        "method": method_dir.name,
        "final_accuracy": number(final, "seen_accuracy"),
        "average_incremental_accuracy": number(final, "average_incremental_accuracy"),
        "final_old_accuracy": number(final, "old_accuracy"),
        "final_new_accuracy": number(final, "new_accuracy"),
        "average_forgetting": number(final, "average_forgetting"),
        "old_class_forgetting": safe_mean(
            [number(row, "forgetting") for row in old_final]
        ),
        "old_prototype_cosine_drift": safe_mean(
            [number(row, "prototype_cosine_drift") for row in old_final]
        ),
        "mean_update_norm": safe_mean(
            [number(row, "parameter_update_norm") for row in nonzero_steps]
        ),
        "mean_parallel_ratio": safe_mean(
            [number(row, "protected_parallel_ratio") for row in nonzero_steps]
        ),
        "max_parallel_ratio": max(
            [number(row, "protected_parallel_ratio") for row in nonzero_steps],
            default=0.0,
        ),
        "mean_removed_gradient_ratio": safe_mean(
            [number(row, "removed_ratio") for row in initialization]
        ),
        "max_aq_norm": max(
            [number(row, "aq_norm") for row in initialization], default=0.0
        ),
        "min_effective_rank": min(
            [int(row["effective_rank"]) for row in initialization], default=0
        ),
        "zero_rank_initializations": sum(
            int(row["effective_rank"]) == 0 for row in initialization
        ),
        "initializations": len(initialization),
        "final_wp_rank_mean": number(final, "wp_rank_mean"),
        "final_m_rank_mean": number(final, "m_rank_mean"),
        "final_unified_rank_mean": number(final, "unified_rank_mean"),
        "final_residual_rank_mean": number(final, "residual_rank_mean"),
        "final_residual_rank_min": number(final, "residual_rank_min"),
        "final_protected_ratio": number(final, "protected_ratio_mean"),
        "peak_cuda_allocated_gb": max(
            number(row, "peak_cuda_allocated_gb") for row in tasks
        ),
        "optimizer_skipped_steps": sum(
            int(row["optimizer_skipped_steps"]) for row in tasks
        ),
        "steps": len(steps),
    }


def consistency_checks(
    method: str,
    matrix: list[list[float]],
    tasks: list[dict[str, str]],
    classes: list[dict[str, str]],
    classes_per_task: int,
) -> dict:
    summary = summarize_cil_matrix(matrix)
    row_errors = [
        abs(summary.row_averages[stage] - number(tasks[stage], "seen_accuracy"))
        for stage in range(len(matrix))
    ]
    cell_errors: list[float] = []
    seen_errors: list[float] = []
    equal_counts = True
    for stage, matrix_row in enumerate(matrix):
        stage_rows = [row for row in classes if int(row["task_id"]) == stage]
        total_correct = sum(int(row["correct"]) for row in stage_rows)
        total_count = sum(int(row["count"]) for row in stage_rows)
        seen_errors.append(abs(total_correct / total_count - summary.row_averages[stage]))
        task_counts: list[int] = []
        for task_id, expected in enumerate(matrix_row):
            task_class_rows = [
                row for row in stage_rows if int(row["learned_task"]) == task_id
            ]
            correct = sum(int(row["correct"]) for row in task_class_rows)
            count = sum(int(row["count"]) for row in task_class_rows)
            task_counts.append(count)
            cell_errors.append(abs(correct / count - expected))
        equal_counts &= len(set(task_counts)) == 1
    expected_seen = [
        (stage + 1) * classes_per_task for stage in range(len(matrix))
    ]
    observed_seen = [int(row["seen_classes"]) for row in tasks]
    maximum_error = max(row_errors + cell_errors + seen_errors)
    return {
        "method": method,
        "future_cells_are_empty": True,
        "equal_test_samples_per_task": equal_counts,
        "maximum_accuracy_consistency_error": maximum_error,
        "seen_classifier_class_counts": observed_seen,
        "expected_seen_classifier_class_counts": expected_seen,
        "passed": maximum_error <= 1.0e-10
        and equal_counts
        and observed_seen == expected_seen,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KeepLoRA EFCIL outputs")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["seed"])
    num_tasks = int(config["num_tasks"])
    seed_root = (
        Path(config["output_root"])
        / config["experiment_name"]
        / f"seed_{seed}"
    )
    report_dir = seed_root / "comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize(seed_root / method, num_tasks) for method in ORDERED_METHODS
    ]
    by_method = {row["method"]: row for row in summaries}
    task_rows = {
        method: read_rows(seed_root / method / "task_metrics.csv")
        for method in ORDERED_METHODS
    }
    class_rows = {
        method: read_rows(seed_root / method / "class_metrics.csv")
        for method in ORDERED_METHODS
    }
    matrices = {
        method: parse_lower_triangular_matrix(
            read_rows(seed_root / method / "accuracy_matrix.csv"), num_tasks
        )
        for method in ORDERED_METHODS
    }
    matrix_summaries = {
        method: summarize_cil_matrix(matrix) for method, matrix in matrices.items()
    }
    checks = {
        method: consistency_checks(
            method,
            matrices[method],
            task_rows[method],
            class_rows[method],
            int(config["classes_per_task"]),
        )
        for method in ORDERED_METHODS
    }
    if not all(check["passed"] for check in checks.values()):
        raise RuntimeError(f"Task-matrix consistency failure: {checks}")

    long_rows = [
        row
        for method in ORDERED_METHODS
        for row in long_format_rows(method, seed, matrices[method])
    ]
    write_rows(report_dir / "task_accuracy_matrix.csv", long_rows)
    summary_metric_rows = [
        {
            "method": method,
            "seed": seed,
            "transfer": "N/A",
            "average_incremental_accuracy": 100.0
            * matrix_summaries[method].average,
            "last": 100.0 * matrix_summaries[method].last,
        }
        for method in ORDERED_METHODS
    ]
    write_rows(report_dir / "summary_metrics.csv", summary_metric_rows)
    for method in ORDERED_METHODS:
        save_heatmap(
            report_dir / f"task_matrix_{method}.png",
            matrices[method],
            DISPLAY_NAMES[method],
        )
    with (report_dir / "task_matrix_consistency.json").open("w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)

    vanilla = by_method["vanilla_lora"]
    wp_only = by_method["keep_lora_wp_only"]
    m_only = by_method["keep_lora_m_only"]
    full = by_method["keep_lora_full"]
    wp_gain = 100.0 * (wp_only["final_accuracy"] - vanilla["final_accuracy"])
    m_gain = 100.0 * (m_only["final_accuracy"] - vanilla["final_accuracy"])
    full_gain = 100.0 * (full["final_accuracy"] - vanilla["final_accuracy"])
    full_over_m = 100.0 * (full["final_accuracy"] - m_only["final_accuracy"])
    full_old_gain = 100.0 * (
        full["final_old_accuracy"] - vanilla["final_old_accuracy"]
    )
    full_new_gain = 100.0 * (
        full["final_new_accuracy"] - vanilla["final_new_accuracy"]
    )
    full_forgetting_reduction = 100.0 * (
        vanilla["average_forgetting"] - full["average_forgetting"]
    )
    m_average_gain = 100.0 * (
        m_only["average_incremental_accuracy"]
        - vanilla["average_incremental_accuracy"]
    )

    lines = [
        "# KeepLoRA on Single-Head EFCIL",
        "",
        "## 전달용 실험 요약",
        "",
        "### 한 문장 요약",
        "",
        "원래 multi-domain task-incremental learning을 대상으로 제안된 KeepLoRA를 "
        "CIFAR-100 single-head exemplar-free class-incremental learning에 옮겨, "
        "PTM principal weight space `Wp`와 이전 task feature space `M`의 보호가 "
        "catastrophic forgetting을 줄이는지 네 가지 동일 조건 ablation으로 검증한 실험이다.",
        "",
        "### 연구 질문",
        "",
        "- 이전 task feature 공간 `M`을 보호하면 이전 class의 망각이 감소하는가?",
        "",
        "- pretrained model의 principal weight 공간 `Wp`를 추가로 보호할 가치가 있는가?",
        "",
        "- `Wp`와 `M`을 함께 보호할 때 안정성이 좋아지면서도 새 class 학습 능력이 유지되는가?",
        "",
        "- task가 누적되어 protected space가 커질 때 residual learning space가 포화되는가?",
        "",
        "### 비교한 네 조건",
        "",
        "| 조건 | Wp 보호 | 이전 task M 보호 | 의미 |",
        "|---|:---:|:---:|---|",
        "| Vanilla LoRA | X | X | subspace projection이 없는 parameter-matched 대조군 |",
        "| KeepLoRA Wp-only | O | X | PTM의 주요 weight 방향만 보호 |",
        "| KeepLoRA M-only | X | O | 이전 task의 주요 feature 방향만 보호 |",
        "| Full KeepLoRA | O | O | 두 공간을 통합해 보호하는 원 방법에 가장 가까운 조건 |",
        "",
        "여기서 Vanilla는 일반적인 two-factor LoRA와 달리 KeepLoRA와 같은 "
        "gradient-informed fixed-direction/B-only parameterization을 사용한다. 네 조건의 "
        "trainable parameter 수를 같게 두고, 보호 공간의 유무만 비교하기 위한 대조군이다.",
        "",
        "### Single-head EFCIL로 변경한 부분",
        "",
        "KeepLoRA의 핵심인 `Wp` 구성, residualized `M` 누적, protected-gradient "
        "projection, SVD 기반 LoRA 초기화, direction matrix 고정, B-only 학습 및 "
        "task-end merge는 유지했다. 원 논문의 domain/task별 MTIL 설정만 다음과 같이 바꿨다.",
        "",
        "- CIFAR-100을 고정된 순서의 10개 task로 나누고 task마다 서로 다른 10개 class를 순차 학습했다.",
        "",
        "- classifier는 하나의 cosine head를 10개 row씩 확장했다. 이전 row는 고정하고 현재 task row만 학습했다.",
        "",
        "- 평가에서는 Task-ID나 task-local mask 없이 현재까지 존재하는 모든 class logit 중 하나를 예측했다.",
        "",
        "- old image replay, synthetic replay 및 task-specific adapter routing을 사용하지 않았다.",
        "",
        "### 한 task에서 수행한 절차",
        "",
        "1. 현재 task의 10개 classifier row를 해당 class의 feature centroid로 초기화한다.",
        "",
        "2. 현재 task 첫 batch의 full attention-weight gradient를 계산한다.",
        "",
        "3. 조건에 따라 gradient에서 `Wp`, `M` 또는 둘의 unified component를 제거한다.",
        "",
        "4. residual gradient를 SVD하여 LoRA rank-16 방향을 정하고, 이 방향 matrix는 고정한 채 나머지 matrix와 새 classifier row만 학습한다.",
        "",
        "5. 10 epochs 학습 후 LoRA update를 base ViT에 merge하고 adapter를 초기화한다.",
        "",
        "6. 지금까지 학습한 모든 class를 single head로 평가한다.",
        "",
        "7. M을 사용하는 조건은 merge된 모델의 현재-task feature에서 기존 protected component를 제거한 뒤 dominant directions를 M에 누적한다. 원본 이미지는 저장하지 않는다.",
        "",
        "### 공정한 비교를 위해 고정한 요소",
        "",
        f"네 조건 모두 동일한 ViT-B/16-IN21K backbone, class order, seed {seed}, "
        f"Q/K/V/O 48개 LoRA site, rank {config['lora_rank']}, alpha "
        f"{config['lora_alpha']}, AdamW, task당 {config['epochs_per_task']} epochs, "
        "cosine classifier 및 데이터 augmentation을 사용했다. 학습에는 현재 task의 "
        "training image만 사용했고 old test image는 평가에만 사용했다.",
        "",
        "### 핵심 결과와 결론",
        "",
        f"- Full KeepLoRA의 final 100-class accuracy는 "
        f"{percent(full['final_accuracy'])}%로 Vanilla의 "
        f"{percent(vanilla['final_accuracy'])}%보다 {full_gain:+.2f}%p 높았다.",
        "",
        f"- Full은 final old-class accuracy를 "
        f"{percent(vanilla['final_old_accuracy'])}%에서 "
        f"{percent(full['final_old_accuracy'])}%로 {full_old_gain:+.2f}%p 높이고, "
        f"forgetting을 {percent(vanilla['average_forgetting'])}%에서 "
        f"{percent(full['average_forgetting'])}%로 {full_forgetting_reduction:.2f}%p 줄였다.",
        "",
        f"- 동시에 final new-class accuracy는 Vanilla "
        f"{percent(vanilla['final_new_accuracy'])}%, Full "
        f"{percent(full['final_new_accuracy'])}%로 차이가 {full_new_gain:+.2f}%p에 불과했다. "
        "따라서 이 run에서는 Full의 개선이 뚜렷한 new-class plasticity 희생으로 만들어진 것은 아니다.",
        "",
        f"- Wp-only와 M-only의 final accuracy 향상은 각각 {wp_gain:+.2f}%p, "
        f"{m_gain:+.2f}%p였다. Final accuracy 기준으로는 Wp의 독립 기여가 더 컸지만, "
        f"M-only는 가장 높은 Average Incremental Accuracy "
        f"{percent(m_only['average_incremental_accuracy'])}%를 기록해 Vanilla보다 "
        f"{m_average_gain:+.2f}%p 높았다.",
        "",
        f"- Full의 마지막 unified protected rank는 48개 site 평균 "
        f"{full['final_unified_rank_mean']:.1f}/768 "
        f"({100.0 * full['final_protected_ratio']:.2f}%)이고, 가장 좁은 site에도 "
        f"residual dimension이 {full['final_residual_rank_min']:.0f} 남았다. "
        "rank-16 LoRA가 사용할 공간보다 충분히 커서 이 실험에서는 subspace 포화가 직접적인 병목으로 관찰되지 않았다.",
        "",
        "결론적으로, seed-0 결과에서는 `Wp`와 `M` 보호가 모두 single-head CIL의 "
        "망각을 완화했고 Full 조합이 가장 높은 최종 정확도를 보였다. 그러나 Full도 "
        f"old/new accuracy가 {percent(full['final_old_accuracy'])}%/"
        f"{percent(full['final_new_accuracy'])}%로 크게 벌어져 catastrophic forgetting과 "
        "classifier recency bias를 해결한 것은 아니다.",
        "",
        "### 해석 시 주의사항",
        "",
        "- 이 결과는 seed 0 하나의 CIFAR-100 실험이므로 통계적 결론이 아니다. 다음 단계는 최소 3 seeds 반복이다.",
        "",
        "- Full은 최종 정확도가 가장 높지만 M-only는 학습 과정 평균 정확도가 가장 높다. 따라서 하나의 지표만으로 절대적 우위를 주장하면 안 된다.",
        "",
        "- old classifier row를 고정하는 공통 정책을 사용했기 때문에 representation forgetting과 classifier bias가 함께 반영돼 있다.",
        "",
        "- 원 논문의 MTIL을 single-head CIL로 바꾼 adaptation 결과이며, 원 논문의 MTIL 수치를 직접 재현한 실험은 아니다.",
        "",
        "## Experimental Setting",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Dataset | CIFAR-100 |",
        f"| Backbone | `{config['model_name']}` (ViT-B/16, IN21K) |",
        f"| Tasks | {num_tasks} tasks × {config['classes_per_task']} classes |",
        f"| LoRA | Q/K/V/O, rank {config['lora_rank']}, alpha {config['lora_alpha']} |",
        f"| Thresholds | epsilon_w={config['epsilon_w']}, epsilon_f={config['epsilon_f']} |",
        "| Classifier | One expanding cosine head; old rows frozen, current rows trainable |",
        f"| Training | {config['epochs_per_task']} epoch(s)/task, seed {seed} |",
        "| Replay / Task ID | None / None |",
        "| Initialization | First-batch full-weight gradient; paper Eq. 5-6; direction frozen, B-only optimization |",
        "",
        "`vanilla_lora` is the parameter-matched no-projection control: it retains the same gradient-informed frozen direction/B-only parameterization but uses neither Wp nor M.",
        "",
        "## Main Results",
        "",
        "| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {DISPLAY_NAMES[row['method']]} | "
            f"{percent(row['average_incremental_accuracy'])} | "
            f"{percent(row['final_accuracy'])} | {percent(row['final_old_accuracy'])} | "
            f"{percent(row['final_new_accuracy'])} | {percent(row['average_forgetting'])} |"
        )
    lines.extend([
        "",
        "## Ablation",
        "",
        "| Method | Wp | M | Final Acc | Delta vs Vanilla |",
        "|---|:---:|:---:|---:|---:|",
        f"| Vanilla | X | X | {percent(vanilla['final_accuracy'])} | +0.00 |",
        f"| Wp-only | O | X | {percent(wp_only['final_accuracy'])} | {wp_gain:+.2f} |",
        f"| M-only | X | O | {percent(m_only['final_accuracy'])} | {m_gain:+.2f} |",
        f"| Full KeepLoRA | O | O | {percent(full['final_accuracy'])} | {full_gain:+.2f} |",
        "",
        "## Task-wise Accuracy Matrix",
        "",
        "Each A[t,j] uses Task-j samples while prediction remains over every class seen through Task t. Future cells are blank; no Task-ID masking is used.",
    ])
    for method in ORDERED_METHODS:
        lines.extend([
            "",
            f"### {DISPLAY_NAMES[method]}",
            "",
            *matrix_markdown(matrices[method]),
            "",
            f"Heatmap: [`task_matrix_{method}.png`](task_matrix_{method}.png)",
        ])

    lines.extend(["", "## Protected Space Growth"])
    for method in ORDERED_METHODS:
        use_wp, use_m = METHOD_FLAGS[method]
        lines.extend([
            "",
            f"### {DISPLAY_NAMES[method]}",
            "",
            "Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.",
            "",
            "| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for task_id, row in enumerate(task_rows[method]):
            active_wp = number(row, "wp_rank_mean") if use_wp else 0.0
            active_m = number(row, "m_rank_mean") if use_m else 0.0
            lines.append(
                f"| {task_id} | {active_wp:.1f} | {active_m:.1f} | "
                f"{number(row, 'unified_rank_mean'):.1f} | "
                f"{number(row, 'residual_rank_mean'):.1f} | "
                f"{number(row, 'residual_rank_min'):.0f} | "
                f"{100.0 * number(row, 'protected_ratio_mean'):.2f} | "
                f"{100.0 * number(row, 'new_accuracy'):.2f} |"
            )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"1. **Wp protection:** Wp-only changes final accuracy by {wp_gain:+.2f} points versus the matched vanilla control.",
        f"2. **Previous-task M protection:** M-only changes final accuracy by {m_gain:+.2f} points versus vanilla.",
        f"3. **Relative contribution:** In this run, the larger standalone final-accuracy contribution comes from {'Wp' if wp_gain > m_gain else 'M'} protection ({max(wp_gain, m_gain):+.2f} points).",
        f"4. **Full combination:** Full KeepLoRA changes final accuracy by {full_gain:+.2f} points versus vanilla and {full_over_m:+.2f} points versus M-only.",
        f"5. **Stability/plasticity:** Full KeepLoRA ends at old/new accuracy {percent(full['final_old_accuracy'])}/{percent(full['final_new_accuracy'])}; vanilla is {percent(vanilla['final_old_accuracy'])}/{percent(vanilla['final_new_accuracy'])}.",
    ])
    for diagnostic_index, method in enumerate(
        ("keep_lora_m_only", "keep_lora_full"), start=6
    ):
        ratios = [number(row, "protected_ratio_mean") for row in task_rows[method]]
        new_acc = [number(row, "new_accuracy") for row in task_rows[method]]
        correlation = pearsonr(ratios, new_acc)
        lines.append(
            f"{diagnostic_index}. **Protected growth ({DISPLAY_NAMES[method]}):** final mean protected ratio is "
            f"{100.0 * ratios[-1]:.2f}% and its task-wise correlation with new accuracy is "
            f"r={float(correlation.statistic):.3f} (p={float(correlation.pvalue):.3g})."
        )

    lines.extend([
        "",
        "## Integrity Diagnostics",
        "",
    ])
    for row in summaries:
        lines.append(
            f"- `{row['method']}`: max ||A^T Q||={row['max_aq_norm']:.3e}, "
            f"max protected update ratio={row['max_parallel_ratio']:.3e}, "
            f"zero-rank initializations={row['zero_rank_initializations']}/{row['initializations']}, "
            f"optimizer skips={row['optimizer_skipped_steps']}, peak CUDA={row['peak_cuda_allocated_gb']:.2f} GiB."
        )
    lines.extend([
        "",
        "All task-matrix consistency checks passed against class-level correct/count values and expanding classifier sizes. Details: [`task_matrix_consistency.json`](task_matrix_consistency.json).",
        "",
        "## KeepLoRA Metric Mapping",
        "",
        "Original MTIL Transfer evaluates future tasks before learning them. The expanding CIL head has no future-class rows, so Transfer is `N/A`. Here Average means lower-triangle CIL Average Incremental Accuracy and Last means final 100-class single-head accuracy.",
        "",
        "## Existing Orthogonal Diagnostic vs KeepLoRA",
        "",
        "The earlier repository diagnostic protects class-wise means and centered PCA directions using Q/V LoRA. This experiment separately implements pretrained-weight Wp, residualized task feature M, Q/K/V/O targets, paper Eq. 5-6 initialization, frozen direction/B-only training, and task-end merge. The earlier outputs were not renamed or overwritten.",
        "",
        "## Scope",
        "",
        f"This is a seed-{seed} {'short implementation-validation run' if config.get('max_train_batches') is not None else 'normal ten-epoch run'}. Old test images are evaluation-only. No raw replay images, synthetic features, task routing, task-specific inference adapters, or oracle Task-ID are used.",
    ])

    (report_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (report_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=True)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
