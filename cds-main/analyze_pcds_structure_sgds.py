"""Analyze the completed controlled PCDS + SGDS activation experiment."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[1])))

import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean


CDS_ROOT = Path(__file__).resolve().parent
EFCIL_ROOT = Path(("${RESEARCH_WORKSPACE}/projects/efcil_direction_diagnostic").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)))
if str(EFCIL_ROOT) not in sys.path:
    sys.path.insert(0, str(EFCIL_ROOT))

from efcil_diag.task_matrix import (  # noqa: E402
    parse_lower_triangular_matrix,
    summarize_cil_matrix,
)


BEGIN = "<!-- BEGIN GENERATED PCDS STRUCTURE SGDS -->"
END = "<!-- END GENERATED PCDS STRUCTURE SGDS -->"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict, key: str) -> float:
    return float(row[key])


def pct(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{100.0 * value:.2f}"


def points(value: float) -> str:
    return f"{100.0 * value:+.2f}%p"


def matrix_markdown(matrix: list[list[float]]) -> list[str]:
    tasks = len(matrix)
    lines = [
        "| trained through | "
        + " | ".join(f"Task{task}" for task in range(tasks))
        + " | Seen Avg |",
        "|---|" + "---:|" * (tasks + 1),
    ]
    for stage, row in enumerate(matrix):
        values = [f"{100.0 * value:.2f}" for value in row]
        values.extend("-" for _ in range(tasks - len(row)))
        lines.append(
            f"| Task{stage} | "
            + " | ".join(values)
            + f" | {100.0 * mean(row):.2f} |"
        )
    return lines


def load_baseline(config: dict) -> dict:
    with Path(config["pcds_baseline_summary"]).open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    matches = [row for row in rows if row.get("method") == "pcds_full"]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one pcds_full baseline summary")
    return matches[0]


def current_summary(output_dir: Path, num_tasks: int) -> dict:
    tasks = read_rows(output_dir / "task_metrics.csv")
    pcds = read_rows(output_dir / "pcds_statistics.csv")
    if len(tasks) != num_tasks or len(pcds) != num_tasks:
        raise RuntimeError("PCDS+SGDS task outputs are incomplete")
    final = tasks[-1]
    return {
        "method": "pcds_structure_sgds",
        "average_incremental_accuracy": number(final, "average_incremental_accuracy"),
        "final_accuracy": number(final, "seen_accuracy"),
        "final_old_accuracy": number(final, "old_accuracy"),
        "final_new_accuracy": number(final, "new_accuracy"),
        "average_forgetting": number(final, "average_forgetting"),
        "peak_cuda_allocated_gb": max(
            number(row, "peak_cuda_allocated_gb") for row in tasks
        ),
        "optimizer_skipped_steps": sum(
            int(number(row, "optimizer_skipped_steps")) for row in pcds
        ),
        "nonfinite_steps": sum(int(number(row, "nonfinite_steps")) for row in pcds),
    }


def build_section(config: dict, output_dir: Path, baseline: dict, current: dict) -> str:
    semantic = read_rows(output_dir / "sgds_semantic_strategy.csv")
    semantic_tasks = read_rows(output_dir / "sgds_semantic_strategy_tasks.csv")
    activation = read_rows(output_dir / "sgds_activation_statistics.csv")
    ranks = read_rows(output_dir / "sgds_effective_rank.csv")
    ca = read_rows(output_dir / "classifier_adaptation.csv")
    tasks = read_rows(output_dir / "task_metrics.csv")
    matrix = parse_lower_triangular_matrix(
        read_rows(output_dir / "task_accuracy_matrix.csv"), int(config["num_tasks"])
    )
    matrix_summary = summarize_cil_matrix(matrix)
    if abs(matrix_summary.last - current["final_accuracy"]) > 1.0e-10:
        raise RuntimeError("Task matrix final average differs from task metrics")
    with (output_dir / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    rank_by_task = {
        task_id: [
            int(row["effective_rank_99"])
            for row in ranks
            if int(row["task_id"]) == task_id
        ]
        for task_id in range(int(config["num_tasks"]))
    }
    reuse_ratios = [number(row, "knowledge_reuse_ratio") for row in semantic_tasks]
    first_half_reuse = mean(reuse_ratios[: len(reuse_ratios) // 2])
    second_half_reuse = mean(reuse_ratios[len(reuse_ratios) // 2 :])
    actual_zero = mean(number(row, "actual_zero_ratio") for row in activation)
    actual_nonzero = mean(number(row, "actual_nonzero_ratio") for row in activation)
    rank_first = mean(rank_by_task[0])
    rank_last = mean(rank_by_task[int(config["num_tasks"]) - 1])
    final_ca = ca[-1]

    deltas = {
        "average_incremental_accuracy": current["average_incremental_accuracy"]
        - baseline["average_incremental_accuracy"],
        "final_accuracy": current["final_accuracy"] - baseline["final_accuracy"],
        "old_accuracy": current["final_old_accuracy"] - baseline["final_old_accuracy"],
        "new_accuracy": current["final_new_accuracy"] - baseline["final_new_accuracy"],
        "forgetting": current["average_forgetting"] - baseline["average_forgetting"],
    }

    lines = [
        BEGIN,
        "",
        "## PCDS + SGDS Activation Guidance Experiment",
        "",
        "### Method",
        "",
        "이 실험은 SGDS 공식 reproduction이 아니다. 기존 PCDS의 APR, SCC, "
        "cosine classifier, Gaussian synthetic-feature classifier adaptation을 유지하고, "
        "[SGDS 논문](https://arxiv.org/abs/2601.21345)의 Semantic Strategy "
        "Formulation, Semantic Exploration, Activation Compaction(Eq. 4-10)만 마지막 "
        "ViT block의 FFN-output adapter 입력에 추가했다.",
        "",
        "SGDS의 task-specific adapter bank, inference-time retrieval/fusion, entropy "
        "selection, SGDS prototype classifier, MOS alignment은 사용하지 않았다. 학습 "
        "objective는 기존과 동일한 `CE + 2.0 * SCC`이며 SGDS loss는 추가하지 않았다.",
        "",
        "- Eq.4: frozen pretrained backbone으로 누적 저장한 class semantic prototype의 cosine-softmax",
        "- Eq.5-6: old/current probability mass. Eq.6은 수식과 Algorithm에 따라 자기 class를 포함",
        "- Eq.7: semantically weighted old class counter를 이용한 Knowledge Reuse",
        "- Eq.8-9: global counter 기반 New Subspace Allocation과 class별 strategy switch",
        "- Eq.10: epoch 6-10 class-specific counter 기반 Activation Compaction",
        "- Counter: Bernoulli와 Top-K를 모두 지난 실제 non-zero dimension만 sample당 1회 count",
        "",
        "논문은 `k=0.6`을 sparsity ratio라고 부르지만 Algorithm 1은 "
        "`TopK(floor(kN))`을 유지한다. 본 구현은 수식/알고리즘을 따라 60% retain, "
        "40% zero로 해석했다. ViT token tensor는 sample별 mean absolute magnitude로 "
        "dimension score를 만들고 동일 dimension mask를 모든 token에 broadcast했다. "
        "Stochastic mask는 training-only이며 evaluation에는 적용하지 않았다.",
        "",
        "### Experimental Setting",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Dataset | CIFAR-100 |",
        f"| Backbone | `{config['model_name']}` |",
        f"| Tasks / seed | {config['num_tasks']} × {config['classes_per_task']} / {config['seed']} |",
        f"| Epoch / batch | {config['epochs_per_task']} per task / {config['batch_size']} |",
        f"| Adapter | FFN-output nonlinear bottleneck {config['adapter_bottleneck']} |",
        f"| APR | gamma={config['apr_gamma']}, rho={config['apr_rho']} |",
        f"| SCC | lambda={config['scc_weight']} |",
        f"| Classifier adaptation | {config['ca_epochs']} epochs, {config['ca_samples_per_class']} Gaussian features/class |",
        "| SGDS target | final ViT block only (layer 11) |",
        f"| SGDS k / beta / gamma | {config['sgds_sparsity']} / {config['sgds_beta']} / {config['sgds_gamma']} |",
        "| Resolved retain / zero target | 60% / 40% |",
        "| Replay / Task ID | no raw replay / none |",
        f"| Adapter trainable params | {int(metadata['adapter_trainable_parameters']):,} |",
        f"| SGDS trainable params | {int(metadata['sgds_trainable_parameters']):,} |",
        "",
        "### Main Results",
        "",
        "| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PCDS | {pct(baseline['average_incremental_accuracy'])} | {pct(baseline['final_accuracy'])} | {pct(baseline['final_old_accuracy'])} | {pct(baseline['final_new_accuracy'])} | {pct(baseline['average_forgetting'])} |",
        f"| PCDS + SGDS | {pct(current['average_incremental_accuracy'])} | {pct(current['final_accuracy'])} | {pct(current['final_old_accuracy'])} | {pct(current['final_new_accuracy'])} | {pct(current['average_forgetting'])} |",
        f"| Delta | {points(deltas['average_incremental_accuracy'])} | {points(deltas['final_accuracy'])} | {points(deltas['old_accuracy'])} | {points(deltas['new_accuracy'])} | {points(deltas['forgetting'])} |",
        "",
        "### Task-wise Accuracy Matrix",
        "",
        *matrix_markdown(matrix),
        "",
        "### SGDS Semantic Strategy Statistics",
        "",
        "| Task | New classes | Reuse | New allocation | Reuse ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in semantic_tasks:
        lines.append(
            f"| {row['task_id']} | {row['new_classes']} | "
            f"{row['knowledge_reuse_classes']} | {row['new_subspace_allocation_classes']} | "
            f"{100.0 * number(row, 'knowledge_reuse_ratio'):.1f}% |"
        )
    lines.extend([
        "",
        "Class-level Eq.4-6 results:",
        "",
        "| Task | Class | S_old | S_new | Strategy |",
        "|---:|---:|---:|---:|---|",
    ])
    for row in semantic:
        lines.append(
            f"| {row['task_id']} | {row['class_id']} | "
            f"{number(row, 's_old'):.6f} | {number(row, 's_new'):.6f} | "
            f"{row['strategy']} |"
        )
    lines.extend([
        "",
        "### Activation Sparsity Statistics",
        "",
        "| Task | Layer | Target zero | Actual non-zero | Actual zero | Mean active dims | Global max | Class max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in activation:
        lines.append(
            f"| {row['task_id']} | {row['target_layer']} | "
            f"{100.0 * number(row, 'requested_zero_ratio'):.2f} | "
            f"{100.0 * number(row, 'actual_nonzero_ratio'):.2f} | "
            f"{100.0 * number(row, 'actual_zero_ratio'):.2f} | "
            f"{number(row, 'mean_active_dimensions'):.1f} | "
            f"{number(row, 'global_counter_max'):.0f} | "
            f"{number(row, 'class_counter_max'):.0f} |"
        )
    lines.extend([
        "",
        "### Representation Effective Rank",
        "",
        "`r_0.99`는 current-task class covariance eigen-energy 99%를 설명하는 최소 rank다.",
        "",
        "| Task | Mean r_0.99 | Min | Max |",
        "|---:|---:|---:|---:|",
    ])
    for task_id, values in rank_by_task.items():
        lines.append(
            f"| {task_id} | {mean(values):.1f} | {min(values)} | {max(values)} |"
        )
    lines.extend([
        "",
        "### Classifier Adaptation Analysis",
        "",
        "| Task | Seen before | Seen after | Delta | Old before | Old after | New before | New after |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ca:
        lines.append(
            f"| {row['task_id']} | {pct(number(row, 'seen_accuracy_before_ca'))} | "
            f"{pct(number(row, 'seen_accuracy_after_ca'))} | "
            f"{points(number(row, 'seen_accuracy_ca_delta'))} | "
            f"{pct(number(row, 'old_accuracy_before_ca'))} | "
            f"{pct(number(row, 'old_accuracy_after_ca'))} | "
            f"{pct(number(row, 'new_accuracy_before_ca'))} | "
            f"{pct(number(row, 'new_accuracy_after_ca'))} |"
        )
    lines.extend([
        "",
        "### Interpretation",
        "",
        f"1. Final accuracy 변화: {points(deltas['final_accuracy'])}. 따라서 seed-0에서 SGDS activation guidance는 PCDS final accuracy를 {'올렸다' if deltas['final_accuracy'] > 0 else '낮췄다' if deltas['final_accuracy'] < 0 else '바꾸지 않았다'}.",
        f"2. Forgetting 변화: {points(deltas['forgetting'])}. 음수이면 감소를 뜻한다.",
        f"3. Old accuracy 변화: {points(deltas['old_accuracy'])}.",
        f"4. New-class accuracy 변화: {points(deltas['new_accuracy'])}. 음수이면 plasticity 손실이다.",
        f"5. 실제 평균 zero ratio는 {100.0 * actual_zero:.2f}%이고 non-zero ratio는 {100.0 * actual_nonzero:.2f}%다. 논문의 명칭상 60% sparsity와는 다르지만 Algorithm의 k=0.6 retain 해석과 비교해야 한다.",
        f"6. Knowledge Reuse 비율은 전반부 평균 {100.0 * first_half_reuse:.1f}%, 후반부 평균 {100.0 * second_half_reuse:.1f}%로 변했다.",
        f"7. Mean r_0.99는 Task0 {rank_first:.1f}에서 Task9 {rank_last:.1f}로 변했다. 기존 PCDS run에는 동일 rank diagnostic이 없어 SGDS가 baseline 대비 rank를 감소시켰는지는 이 값만으로 확정할 수 없다.",
        f"8. Task9 CA는 Seen accuracy를 {pct(number(final_ca, 'seen_accuracy_before_ca'))}에서 {pct(number(final_ca, 'seen_accuracy_after_ca'))}로 {points(number(final_ca, 'seen_accuracy_ca_delta'))} 변화시켰다. 전체 PCDS+SGDS 조합 실험이므로 APR/SCC/CA와 SGDS 각각의 인과 효과를 완전히 분리할 수 없다.",
        "",
        "### Integrity Diagnostics",
        "",
        f"- Peak CUDA allocated: {current['peak_cuda_allocated_gb']:.2f} GiB",
        f"- Optimizer skipped steps: {current['optimizer_skipped_steps']}",
        f"- NaN/Inf steps: {current['nonfinite_steps']}",
        "- Evaluation masking: disabled; single expanding head over all seen classes",
        "- Counter source: final Bernoulli + Top-K non-zero dimensions only",
        "",
        "### Limitations",
        "",
        "- Seed 0 한 번의 실험이므로 통계적 결론이 아니다.",
        "- SGDS 공식 reproduction이 아니라 PCDS에 activation mechanism만 추가한 조합 실험이다.",
        "- SGDS 원 논문의 CIFAR-100 protocol은 20×5 classes와 20 epochs/task지만, 본 실험은 PCDS 비교를 위해 10×10과 10 epochs/task를 사용했다.",
        "- SGDS 원 논문의 task-specific adapter retrieval/fusion을 사용하지 않았다.",
        "- 논문의 k 명칭과 Top-K 알고리즘 사이 모호성 때문에 수식상 60% retain으로 구현했다.",
        "- 따라서 이 결과를 SGDS 논문의 reported CIFAR-100 score와 직접 비교할 수 없다.",
        "",
        END,
    ])
    return "\n".join(lines)


def analyze_results(config: dict, output_dir: Path) -> Path:
    if not (output_dir / "COMPLETED.json").exists():
        raise RuntimeError(f"Cannot analyze incomplete run: {output_dir}")
    baseline = load_baseline(config)
    current = current_summary(output_dir, int(config["num_tasks"]))
    section = build_section(config, output_dir, baseline, current)
    result_path = Path(config["result_document"])
    if result_path.exists():
        existing = result_path.read_text(encoding="utf-8")
        base = existing.split(BEGIN, maxsplit=1)[0].rstrip()
    else:
        base = "\n".join([
            "# PCDS / PCMD Experiments",
            "",
            "## Existing PCDS Baseline",
            "",
            "`RESULTS_PCMD.md` did not previously exist in this codebase. This baseline "
            "section was initialized from the completed PCDS seed-0 output specified in "
            "the experiment config; metrics were not hard-coded.",
            "",
            "| Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |",
            "|---:|---:|---:|---:|---:|",
            f"| {pct(baseline['average_incremental_accuracy'])} | {pct(baseline['final_accuracy'])} | {pct(baseline['final_old_accuracy'])} | {pct(baseline['final_new_accuracy'])} | {pct(baseline['average_forgetting'])} |",
        ])
    merged = base + "\n\n" + section + "\n"
    result_path.write_text(merged, encoding="utf-8")
    (output_dir / "RESULTS.md").write_text(merged, encoding="utf-8")
    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"pcds": baseline, "pcds_structure_sgds": current},
            handle,
            indent=2,
            allow_nan=True,
        )
    print(f"[PCDS+SGDS] appended report: {result_path}")
    return result_path

