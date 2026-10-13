"""Compare completed Cars-196 PCDS and PCDS+SGDS runs and append a report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


BEGIN = "<!-- BEGIN GENERATED CARS196 PCDS SGDS -->"
END = "<!-- END GENERATED CARS196 PCDS SGDS -->"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(row: dict, key: str) -> float:
    return float(row[key])


def pct(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{100.0 * value:.2f}"


def points(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{100.0 * value:+.2f}%p"


def corr(left: list[float], right: list[float]) -> float:
    pairs = [
        (x, y)
        for x, y in zip(left, right)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return float("nan")
    xs, ys = zip(*pairs)
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def summary(output_dir: Path, method: str, num_tasks: int) -> dict:
    tasks = read_rows(output_dir / "task_metrics.csv")
    stats = read_rows(output_dir / "pcds_statistics.csv")
    if len(tasks) != num_tasks or len(stats) != num_tasks:
        raise RuntimeError(f"Incomplete {method} Cars output")
    final = tasks[-1]
    result = {
        "method": method,
        "average_incremental_accuracy": num(final, "average_incremental_accuracy"),
        "final_accuracy": num(final, "seen_accuracy"),
        "final_old_accuracy": num(final, "old_accuracy"),
        "final_new_accuracy": num(final, "new_accuracy"),
        "average_forgetting": num(final, "average_forgetting"),
        "old_new_gap": num(final, "old_accuracy") - num(final, "new_accuracy"),
        "peak_cuda_allocated_gb": max(num(row, "peak_cuda_allocated_gb") for row in tasks),
        "optimizer_skipped_steps": sum(int(num(row, "optimizer_skipped_steps")) for row in stats),
        "nonfinite_steps": sum(int(num(row, "nonfinite_steps")) for row in stats),
    }
    if result["optimizer_skipped_steps"] or result["nonfinite_steps"]:
        raise RuntimeError(f"{method} contains skipped or non-finite training steps")
    return result


def task_matrix(output_dir: Path, num_tasks: int) -> list[list[float]]:
    rows = read_rows(output_dir / "task_accuracy_matrix.csv")
    if len(rows) != num_tasks:
        raise RuntimeError("Task matrix is incomplete")
    matrix = []
    for stage, row in enumerate(rows):
        matrix.append([float(row[f"task_{task}"]) for task in range(stage + 1)])
    return matrix


def matrix_md(
    method: str, matrix: list[list[float]], task_rows: list[dict[str, str]]
) -> list[str]:
    count = len(matrix)
    lines = [
        f"#### {method}",
        "",
        "| trained through | "
        + " | ".join(f"Task{task}" for task in range(count))
        + " | Seen Acc |",
        "|---|" + "---:|" * (count + 1),
    ]
    for stage, row in enumerate(matrix):
        values = [f"{100.0 * value:.2f}" for value in row]
        values += ["-"] * (count - len(row))
        lines.append(
            f"| Task{stage} | "
            + " | ".join(values)
            + f" | {100.0 * num(task_rows[stage], 'seen_accuracy'):.2f} |"
        )
    return lines


def rank_by_task(output_dir: Path, num_tasks: int) -> dict[int, list[int]]:
    rows = read_rows(output_dir / "representation_effective_rank.csv")
    result = {
        task: [
            int(row["effective_rank_99"])
            for row in rows
            if int(row["task_id"]) == task
        ]
        for task in range(num_tasks)
    }
    if any(not values for values in result.values()):
        raise RuntimeError("Effective-rank output is incomplete")
    return result


def build_section(config: dict, baseline_dir: Path, sgds_dir: Path) -> tuple[str, dict]:
    num_tasks = int(config["num_tasks"])
    pcds = summary(baseline_dir, "pcds_full", num_tasks)
    sgds = summary(sgds_dir, "pcds_structure_sgds", num_tasks)
    deltas = {
        key: sgds[key] - pcds[key]
        for key in (
            "average_incremental_accuracy",
            "final_accuracy",
            "final_old_accuracy",
            "final_new_accuracy",
            "average_forgetting",
            "old_new_gap",
        )
    }

    pcds_tasks = read_rows(baseline_dir / "task_metrics.csv")
    sgds_tasks = read_rows(sgds_dir / "task_metrics.csv")
    pcds_matrix = task_matrix(baseline_dir, num_tasks)
    sgds_matrix = task_matrix(sgds_dir, num_tasks)
    semantic = read_rows(sgds_dir / "sgds_semantic_strategy.csv")
    semantic_tasks = read_rows(sgds_dir / "sgds_semantic_strategy_tasks.csv")
    activation = read_rows(sgds_dir / "sgds_activation_statistics.csv")
    pcds_rank = rank_by_task(baseline_dir, num_tasks)
    sgds_rank = rank_by_task(sgds_dir, num_tasks)
    pcds_ca = read_rows(baseline_dir / "classifier_adaptation.csv")
    sgds_ca = read_rows(sgds_dir / "classifier_adaptation.csv")
    final_class_rows = {
        int(row["class_id"]): row
        for row in read_rows(sgds_dir / "class_metrics.csv")
        if int(row["task_id"]) == num_tasks - 1
    }

    old_counts = [num(row, "old_classes") for row in semantic_tasks]
    mean_old_scores = [num(row, "mean_s_old") for row in semantic_tasks]
    old_count_score_corr = corr(old_counts, mean_old_scores)
    similarities = [num(row, "max_old_cosine_similarity") for row in semantic]
    reuse_codes = [1.0 if row["strategy"] == "KNOWLEDGE_REUSE" else 0.0 for row in semantic]
    final_accuracies = [
        num(final_class_rows[int(row["class_id"])], "accuracy") for row in semantic
    ]
    similarity_reuse_corr = corr(similarities, reuse_codes)
    similarity_accuracy_corr = corr(similarities, final_accuracies)
    reuse_similarity = [
        value
        for value, code in zip(similarities, reuse_codes)
        if code == 1.0 and math.isfinite(value)
    ]
    allocation_similarity = [
        value
        for value, code in zip(similarities, reuse_codes)
        if code == 0.0 and math.isfinite(value)
    ]

    cifar_summary = json.loads(Path(config["cifar_comparison_summary"]).read_text())
    cifar_pcds = cifar_summary["pcds"]
    cifar_sgds = cifar_summary["pcds_structure_sgds"]
    cifar_delta = cifar_sgds["final_accuracy"] - cifar_pcds["final_accuracy"]
    cars_beats_cifar = deltas["final_accuracy"] > cifar_delta
    cifar_semantic_tasks = read_rows(Path(config["cifar_sgds_semantic_tasks"]))
    cifar_varied_tasks = sum(
        0.0 < num(row, "knowledge_reuse_ratio") < 1.0
        for row in cifar_semantic_tasks
    )
    cars_varied_tasks = sum(
        0.0 < num(row, "knowledge_reuse_ratio") < 1.0
        for row in semantic_tasks
    )

    class_order = json.loads((sgds_dir / "class_order.json").read_text())
    order_sha = hashlib.sha256(
        json.dumps(class_order, separators=(",", ":")).encode()
    ).hexdigest()
    pcds_metadata = json.loads((baseline_dir / "run_metadata.json").read_text())
    sgds_metadata = json.loads((sgds_dir / "run_metadata.json").read_text())

    lines = [
        BEGIN,
        "",
        "## Cars-196: PCDS vs PCDS + SGDS",
        "",
        "### Experimental Setting",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Dataset | Stanford Cars-196 official train/test split |",
        f"| Task split | `{config['task_sizes']}` = 20 classes × 9 + 16 classes |",
        f"| Training seed / class-order seed | {config['seed']} / {config['class_order_seed']} |",
        f"| Class order | `{config['class_order']}`; SHA-256 `{order_sha}` |",
        f"| Backbone | `{config['model_name']}` |",
        f"| Epochs | Task0 {config['initial_epochs']}; Task1-9 {config['incremental_epochs']} |",
        f"| Batch / SGD lr | {config['batch_size']} / {config['learning_rate']} |",
        f"| Adapter | parallel attention K/V nonlinear bottleneck {config['adapter_bottleneck']} |",
        f"| Cosine classifier | scale={config['classifier_scale']}, margin={config['classifier_margin']} |",
        f"| APR / SCC | gamma={config['apr_gamma']}, rho={config['apr_rho']}; lambda={config['scc_weight']} |",
        f"| CA | {config['ca_epochs']} epochs, {config['ca_samples_per_class']} Gaussian features/class |",
        f"| SGDS | final block, k={config['sgds_sparsity']} retain, beta={config['sgds_beta']}, gamma={config['sgds_gamma']} |",
        "| Replay / Task ID | none / none |",
        f"| GPU | physical GPU {config['physical_gpu']} only |",
        f"| PCDS / SGDS trainable adapter params | {int(pcds_metadata['adapter_trainable_parameters']):,} / {int(sgds_metadata['adapter_trainable_parameters']):,} |",
        "",
        "정확한 196-class order는 각 run의 `class_order.json`에 저장했다:",
        "",
        "```text",
        json.dumps(class_order),
        "```",
        "",
        "### Main Results",
        "",
        "| Method | Avg Inc. Acc | Final Acc | Old Acc | New Acc | Forgetting | Old-New gap | Peak CUDA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| PCDS | {pct(pcds['average_incremental_accuracy'])} | {pct(pcds['final_accuracy'])} | {pct(pcds['final_old_accuracy'])} | {pct(pcds['final_new_accuracy'])} | {pct(pcds['average_forgetting'])} | {points(pcds['old_new_gap'])} | {pcds['peak_cuda_allocated_gb']:.2f} GiB |",
        f"| PCDS + SGDS | {pct(sgds['average_incremental_accuracy'])} | {pct(sgds['final_accuracy'])} | {pct(sgds['final_old_accuracy'])} | {pct(sgds['final_new_accuracy'])} | {pct(sgds['average_forgetting'])} | {points(sgds['old_new_gap'])} | {sgds['peak_cuda_allocated_gb']:.2f} GiB |",
        f"| Delta | {points(deltas['average_incremental_accuracy'])} | {points(deltas['final_accuracy'])} | {points(deltas['final_old_accuracy'])} | {points(deltas['final_new_accuracy'])} | {points(deltas['average_forgetting'])} | {points(deltas['old_new_gap'])} | {sgds['peak_cuda_allocated_gb'] - pcds['peak_cuda_allocated_gb']:+.2f} GiB |",
        "",
        "### Task-wise Results",
        "",
        *matrix_md("PCDS", pcds_matrix, pcds_tasks),
        "",
        *matrix_md("PCDS + SGDS", sgds_matrix, sgds_tasks),
        "",
        "### SGDS Semantic Strategy",
        "",
        "| Task | Old | New | Reuse | Allocation | Reuse ratio | mean S_old | mean S_new |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in semantic_tasks:
        lines.append(
            f"| {row['task_id']} | {row['old_classes']} | {row['new_classes']} | "
            f"{row['knowledge_reuse_classes']} | {row['new_subspace_allocation_classes']} | "
            f"{100.0 * num(row, 'knowledge_reuse_ratio'):.1f}% | "
            f"{num(row, 'mean_s_old'):.6f} | {num(row, 'mean_s_new'):.6f} |"
        )
    lines += [
        "",
        f"`corr(|Y_old|, mean S_old) = {old_count_score_corr:.4f}`. 이는 class 수 증가와 old probability mass 증가가 얼마나 함께 움직였는지 나타내며 수식 자체는 변경하지 않았다.",
        "",
        "### Fine-Grained Semantic Similarity",
        "",
        "| Task | Class | max old cosine | mean old cosine | Strategy | Final class acc |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for row in semantic:
        lines.append(
            f"| {row['task_id']} | {row['class_id']} | "
            f"{num(row, 'max_old_cosine_similarity'):.6f} | "
            f"{num(row, 'mean_old_cosine_similarity'):.6f} | {row['strategy']} | "
            f"{pct(num(final_class_rows[int(row['class_id'])], 'accuracy'))} |"
        )
    lines += [
        "",
        f"- corr(max old cosine, Knowledge Reuse indicator): {similarity_reuse_corr:.4f}",
        f"- corr(max old cosine, final class accuracy): {similarity_accuracy_corr:.4f}",
        f"- mean max cosine, Reuse / Allocation: {mean(reuse_similarity) if reuse_similarity else float('nan'):.4f} / {mean(allocation_similarity) if allocation_similarity else float('nan'):.4f}",
        "",
        "### Effective Rank",
        "",
        "| Task | PCDS rank | PCDS+SGDS rank | Delta |",
        "|---:|---:|---:|---:|",
    ]
    rank_deltas = []
    for task in range(num_tasks):
        left, right = mean(pcds_rank[task]), mean(sgds_rank[task])
        rank_deltas.append(right - left)
        lines.append(f"| {task} | {left:.1f} | {right:.1f} | {right-left:+.1f} |")
    lines += [
        "",
        "### Classifier Adaptation",
        "",
        "| Method | Task | Seen before/after | Old before/after | New before/after |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, rows in (("PCDS", pcds_ca), ("PCDS+SGDS", sgds_ca)):
        for row in rows:
            lines.append(
                f"| {label} | {row['task_id']} | {pct(num(row, 'seen_accuracy_before_ca'))} / {pct(num(row, 'seen_accuracy_after_ca'))} | "
                f"{pct(num(row, 'old_accuracy_before_ca'))} / {pct(num(row, 'old_accuracy_after_ca'))} | "
                f"{pct(num(row, 'new_accuracy_before_ca'))} / {pct(num(row, 'new_accuracy_after_ca'))} |"
            )
    lines += [
        "",
        "### CIFAR-100 Comparison",
        "",
        "| Dataset | PCDS Final | PCDS+SGDS Final | Delta Final | PCDS forgetting | SGDS forgetting |",
        "|---|---:|---:|---:|---:|---:|",
        f"| CIFAR-100 | {pct(cifar_pcds['final_accuracy'])} | {pct(cifar_sgds['final_accuracy'])} | {points(cifar_delta)} | {pct(cifar_pcds['average_forgetting'])} | {pct(cifar_sgds['average_forgetting'])} |",
        f"| Cars-196 | {pct(pcds['final_accuracy'])} | {pct(sgds['final_accuracy'])} | {points(deltas['final_accuracy'])} | {pct(pcds['average_forgetting'])} | {pct(sgds['average_forgetting'])} |",
        "",
        "### Interpretation",
        "",
        f"1. Cars-196 Final Acc 변화는 {points(deltas['final_accuracy'])}이므로 SGDS는 PCDS 성능을 {'개선했다' if deltas['final_accuracy'] > 0 else '낮췄다' if deltas['final_accuracy'] < 0 else '바꾸지 않았다'}.",
        f"2. CIFAR-100의 +0.07%p와 비교하면 Cars 개선폭은 {'더 크다' if cars_beats_cifar else '더 크지 않다'} ({points(deltas['final_accuracy'])}).",
        f"3. Old Acc 변화는 {points(deltas['final_old_accuracy'])}이다.",
        f"4. Forgetting 변화는 {points(deltas['average_forgetting'])}이다. 음수이면 감소다.",
        f"5. New Acc 변화는 {points(deltas['final_new_accuracy'])}이다. 음수이면 plasticity 감소다.",
        f"6. 평균 effective-rank delta는 {mean(rank_deltas):+.2f}다. 음수이면 SGDS rank가 감소했다.",
        f"7. 0%와 100% 사이의 mixed-strategy task는 Cars {cars_varied_tasks}개, CIFAR {cifar_varied_tasks}개다.",
        f"8. 마지막 task Reuse ratio는 {100.0 * num(semantic_tasks[-1], 'knowledge_reuse_ratio'):.1f}%다.",
        f"9. max cosine과 Reuse의 상관은 {similarity_reuse_corr:.4f}다.",
        f"10. max cosine과 final class accuracy 상관은 {similarity_accuracy_corr:.4f}다. 이 단일 seed 상관만으로 fine-grained geometry가 인과 원인이라고 주장할 수 없다.",
        "",
        "### Activation Sparsity",
        "",
        "| Task | Requested zero | Actual zero | Actual non-zero |",
        "|---:|---:|---:|---:|",
    ]
    for row in activation:
        lines.append(
            f"| {row['task_id']} | {pct(num(row, 'requested_zero_ratio'))} | "
            f"{pct(num(row, 'actual_zero_ratio'))} | {pct(num(row, 'actual_nonzero_ratio'))} |"
        )
    lines += [
        "",
        "### Integrity and Limitations",
        "",
        f"- PCDS / SGDS optimizer skipped steps: {pcds['optimizer_skipped_steps']} / {sgds['optimizer_skipped_steps']}",
        f"- PCDS / SGDS NaN/Inf steps: {pcds['nonfinite_steps']} / {sgds['nonfinite_steps']}",
        "- Seed 0 단일 screening run이므로 statistical significance를 주장하지 않는다.",
        "- 논문 보고값 80.23±0.68은 참고값이며 이번 baseline으로 하드코딩하지 않았다.",
        "- PCDS+SGDS는 SGDS 공식 reproduction이 아니라 Eq. 4-10 activation guidance 조합이다.",
        "- SGDS의 task-specific adapter retrieval/fusion은 사용하지 않았다.",
        "",
        END,
    ]
    comparison = {
        "pcds": pcds,
        "pcds_structure_sgds": sgds,
        "delta": deltas,
        "semantic_diagnostics": {
            "old_class_count_vs_mean_s_old_correlation": old_count_score_corr,
            "max_old_cosine_vs_reuse_correlation": similarity_reuse_corr,
            "max_old_cosine_vs_final_class_accuracy_correlation": similarity_accuracy_corr,
            "cars_mixed_strategy_tasks": cars_varied_tasks,
            "cifar_mixed_strategy_tasks": cifar_varied_tasks,
        },
        "effective_rank_mean_delta": mean(rank_deltas),
        "cars_final_delta_exceeds_cifar": cars_beats_cifar,
    }
    return "\n".join(lines), comparison


def analyze_results(config: dict, sgds_dir: Path) -> Path:
    baseline_dir = Path(config["cars_pcds_output_dir"])
    for directory in (baseline_dir, sgds_dir):
        if not (directory / "COMPLETED.json").exists():
            raise RuntimeError(f"Cannot analyze incomplete Cars run: {directory}")
    section, comparison = build_section(config, baseline_dir, sgds_dir)
    result_path = Path(config["result_document"])
    existing = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
    if BEGIN in existing:
        prefix, remainder = existing.split(BEGIN, 1)
        if END not in remainder:
            raise RuntimeError("Cars report begin marker exists without end marker")
        _, suffix = remainder.split(END, 1)
        merged = prefix.rstrip() + "\n\n" + section + suffix
    else:
        merged = existing.rstrip() + "\n\n" + section + "\n"
    result_path.write_text(merged.lstrip(), encoding="utf-8")
    (sgds_dir / "RESULTS.md").write_text(merged.lstrip(), encoding="utf-8")
    (sgds_dir / "comparison_summary.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"[Cars-196 PCDS+SGDS] appended report: {result_path}")
    return result_path
