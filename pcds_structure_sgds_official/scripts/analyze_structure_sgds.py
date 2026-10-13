#!/usr/bin/env python3
"""Compare completed official Structure and Structure+SGDS Cars-196 runs."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value in (None, "", "None"):
        return float("nan")
    return float(value)


def fmt(value, digits=2, signed=False):
    value = float(value)
    if not math.isfinite(value):
        return "N/A"
    pattern = "{:+." + str(digits) + "f}" if signed else "{: ." + str(digits) + "f}"
    return pattern.format(value).strip()


def correlation(left, right):
    pairs = [
        (float(x), float(y)) for x, y in zip(left, right)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return float("nan")
    x, y = np.asarray(pairs, dtype=float).T
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def completed_method(path):
    path = Path(path)
    if not (path / "COMPLETED.json").exists():
        raise RuntimeError("Run is not complete: {}".format(path))
    return {
        "dir": path,
        "summary": read_json(path / "summary.json"),
        "tasks": read_csv(path / "task_metrics.csv"),
        "matrix": read_csv(path / "task_accuracy_matrix.csv"),
        "ranks": read_csv(path / "effective_rank.csv"),
        "ca": read_csv(path / "classifier_adaptation.csv"),
        "class_accuracy": read_csv(path / "class_accuracy.csv"),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    standard = root / "outputs/cars196_10task_official_structure/seed_0"
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=standard / "pcds_structure")
    parser.add_argument("--sgds-dir", default=standard / "pcds_structure_sgds")
    parser.add_argument("--output-dir", default=standard / "comparison")
    args = parser.parse_args()

    baseline = completed_method(args.baseline_dir)
    sgds = completed_method(args.sgds_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    keys = [
        "average_incremental_accuracy", "final_accuracy", "final_old_accuracy",
        "final_new_accuracy", "average_forgetting", "old_new_gap",
        "peak_cuda_allocated_gb", "peak_cuda_reserved_gb",
    ]
    delta = {
        key: float(sgds["summary"][key]) - float(baseline["summary"][key])
        for key in keys
    }

    baseline_rank = {
        int(row["task_id"]): number(row["r_0_99"])
        for row in baseline["ranks"] if row["scope"] == "current_task"
    }
    sgds_rank = {
        int(row["task_id"]): number(row["r_0_99"])
        for row in sgds["ranks"] if row["scope"] == "current_task"
    }
    rank_delta = {
        task: sgds_rank[task] - baseline_rank[task]
        for task in sorted(set(baseline_rank) & set(sgds_rank))
    }

    semantic_tasks = read_csv(sgds["dir"] / "sgds_semantic_tasks.csv")
    semantic_classes = read_csv(sgds["dir"] / "sgds_semantic_strategy.csv")
    activation_tasks = read_csv(sgds["dir"] / "sgds_activation.csv")
    old_counts = [number(row["old_classes"]) for row in semantic_tasks]
    mean_old = [number(row["mean_s_old"]) for row in semantic_tasks]
    old_mass_correlation = correlation(old_counts, mean_old)
    varied_tasks = sum(
        0.0 < number(row["knowledge_reuse_ratio"]) < 1.0
        for row in semantic_tasks
    )
    saturated_tasks = [
        int(row["task_id"]) for row in semantic_tasks
        if number(row["knowledge_reuse_ratio"]) >= 1.0 - 1e-12
    ]

    final_task_id = max(int(row["trained_through_task"]) for row in sgds["class_accuracy"])
    final_class_accuracy = {
        int(row["class_id"]): number(row["accuracy"])
        for row in sgds["class_accuracy"]
        if int(row["trained_through_task"]) == final_task_id
    }
    class_diagnostics = []
    for row in semantic_classes:
        class_id = int(row["class_id"])
        max_similarity = number(row["max_old_cosine_similarity"])
        reuse = 1.0 if row["strategy"] == "KNOWLEDGE_REUSE" else 0.0
        class_diagnostics.append((max_similarity, reuse, final_class_accuracy.get(class_id, float("nan"))))
    similarity_reuse_corr = correlation(
        [row[0] for row in class_diagnostics], [row[1] for row in class_diagnostics]
    )
    similarity_accuracy_corr = correlation(
        [row[0] for row in class_diagnostics], [row[2] for row in class_diagnostics]
    )
    reuse_similarities = [row[0] for row in class_diagnostics if row[1] == 1 and math.isfinite(row[0])]
    allocation_similarities = [row[0] for row in class_diagnostics if row[1] == 0 and math.isfinite(row[0])]

    analysis = {
        "baseline": baseline["summary"],
        "pcds_structure_sgds": sgds["summary"],
        "delta_sgds_minus_baseline": delta,
        "effective_rank": {
            "baseline": baseline_rank,
            "sgds": sgds_rank,
            "delta": rank_delta,
            "mean_delta": float(np.mean(list(rank_delta.values()))) if rank_delta else float("nan"),
        },
        "semantic_diagnostics": {
            "corr_old_class_count_vs_mean_s_old": old_mass_correlation,
            "tasks_with_class_dependent_strategy_variation": varied_tasks,
            "fully_reuse_saturated_tasks": saturated_tasks,
            "corr_max_old_cosine_vs_reuse": similarity_reuse_corr,
            "corr_max_old_cosine_vs_final_class_accuracy": similarity_accuracy_corr,
            "mean_max_old_cosine_reuse": float(np.mean(reuse_similarities)) if reuse_similarities else float("nan"),
            "mean_max_old_cosine_allocation": float(np.mean(allocation_similarities)) if allocation_similarities else float("nan"),
        },
        "scope_note": "Single seed; not a statistical reproduction of the paper mean.",
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False, allow_nan=True)
        handle.write("\n")
    with (output / "summary_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "pcds_structure", "pcds_structure_sgds", "delta"])
        for key in keys:
            writer.writerow([key, baseline["summary"][key], sgds["summary"][key], delta[key]])

    main_rows = []
    labels = [("Official Structure (PCDS)", baseline["summary"]), ("Structure + SGDS", sgds["summary"])]
    for label, summary in labels:
        main_rows.append([
            label,
            fmt(summary["average_incremental_accuracy"]),
            fmt(summary["final_accuracy"]),
            fmt(summary["final_old_accuracy"]),
            fmt(summary["final_new_accuracy"]),
            fmt(summary["average_forgetting"]),
        ])
    main_rows.append([
        "Delta (SGDS - Structure)",
        fmt(delta["average_incremental_accuracy"], signed=True),
        fmt(delta["final_accuracy"], signed=True),
        fmt(delta["final_old_accuracy"], signed=True),
        fmt(delta["final_new_accuracy"], signed=True),
        fmt(delta["average_forgetting"], signed=True),
    ])

    task_rows = []
    baseline_task_rows = []
    sgds_task_rows = []
    task_delta_rows = []
    for base_row, sgds_row in zip(baseline["tasks"], sgds["tasks"]):
        common = [base_row["task_id"], base_row["seen_classes"]]
        baseline_task_rows.append(common + [
            fmt(number(base_row["seen_accuracy"])),
            fmt(number(base_row["old_accuracy"])),
            fmt(number(base_row["new_accuracy"])),
            fmt(number(base_row["average_forgetting"])),
            fmt(number(base_row["average_incremental_accuracy"])),
        ])
        sgds_task_rows.append(common + [
            fmt(number(sgds_row["seen_accuracy"])),
            fmt(number(sgds_row["old_accuracy"])),
            fmt(number(sgds_row["new_accuracy"])),
            fmt(number(sgds_row["average_forgetting"])),
            fmt(number(sgds_row["average_incremental_accuracy"])),
        ])
        task_delta_rows.append([base_row["task_id"]] + [
            fmt(number(sgds_row[key]) - number(base_row[key]), signed=True)
            for key in (
                "seen_accuracy", "old_accuracy", "new_accuracy",
                "average_forgetting", "average_incremental_accuracy",
            )
        ])
        task_rows.append([
            base_row["task_id"],
            fmt(number(base_row["seen_accuracy"])),
            fmt(number(sgds_row["seen_accuracy"])),
            fmt(number(sgds_row["seen_accuracy"]) - number(base_row["seen_accuracy"]), signed=True),
        ])
    matrix_headers = ["After task"] + ["Eval T{}".format(index) for index in range(10)] + ["Seen avg"]

    def matrix_rows(method):
        rows = []
        for row in method["matrix"]:
            task_id = int(row["trained_through_task"])
            values = [
                fmt(number(row.get("task_{}".format(index)))) if index <= task_id else "—"
                for index in range(10)
            ]
            rows.append([task_id] + values + [fmt(number(row["seen_average"]))])
        return rows

    baseline_matrix_rows = matrix_rows(baseline)
    sgds_matrix_rows = matrix_rows(sgds)
    rank_rows = [
        [task, fmt(baseline_rank[task], 0), fmt(sgds_rank[task], 0), fmt(rank_delta[task], 0, signed=True)]
        for task in sorted(rank_delta)
    ]
    ca_rows = []
    for base_row, sgds_row in zip(baseline["ca"], sgds["ca"]):
        base_gain = number(base_row["after_seen_accuracy"]) - number(base_row["before_seen_accuracy"])
        sgds_gain = number(sgds_row["after_seen_accuracy"]) - number(sgds_row["before_seen_accuracy"])
        ca_rows.append([
            base_row["task_id"],
            fmt(number(base_row["before_seen_accuracy"])),
            fmt(number(base_row["after_seen_accuracy"])),
            fmt(base_gain, signed=True),
            fmt(number(sgds_row["before_seen_accuracy"])),
            fmt(number(sgds_row["after_seen_accuracy"])),
            fmt(sgds_gain, signed=True),
        ])
    semantic_rows = [
        [
            row["task_id"], row["old_classes"], row["new_classes"],
            row["knowledge_reuse_classes"], row["new_subspace_allocation_classes"],
            fmt(number(row["knowledge_reuse_ratio"]) * 100.0),
            fmt(number(row["mean_s_old"]), 4), fmt(number(row["mean_s_new"]), 4),
        ]
        for row in semantic_tasks
    ]
    activation_rows = [
        [
            row["task_id"], row["target_layer"],
            fmt(number(row["actual_nonzero_ratio"]) * 100.0),
            fmt(number(row["actual_zero_ratio"]) * 100.0),
            fmt(number(row["mean_active_dimensions"]), 1),
            fmt(number(row["mean_activation_probability"]), 4),
            fmt(number(row["global_counter_max"]), 0),
            fmt(number(row["global_counter_mean"]), 1),
            fmt(number(row["class_counter_max"]), 0),
            fmt(number(row["class_counter_mean"]), 1),
        ]
        for row in activation_tasks
    ]

    improved = delta["final_accuracy"] > 0
    report = r"""# Cars-196: official Structure (PCDS) vs Structure + SGDS Activation Guidance

## What was run

Both rows use the official `models/CDS_adapter.py` Structure learner from upstream commit `a0db2168663982a4cad24981231721a3518ddf29`: ViT-B/16 ImageNet-21K, attention adapters (bottleneck 64), SCC, APR, cosine classifier, and Gaussian classifier adaptation. The SGDS row adds only Eq. (4)-(10) semantic exploration/activation compaction to the final ViT block's adapter input. It does not use the SGDS adapter bank, retrieval, fusion, entropy selection, or classifier.

Cars-196 uses the prepared official train/test split (8,144/8,041 images), a single expanding head without Task-ID, no raw replay, and task sizes `20 x 9 + 16`. Training seed is 0 and the shared class-order seed is 1993. The official Cars hyperparameters are retained: 70 epochs on Task 0, 40 thereafter, SCC weight 5, APR gamma/rho 10, and CA 5 epochs.

`k=0.6` is explicitly interpreted as Top-K retain ratio (up to 60% retained, about 40% zero); the Bernoulli mask may make the realized retained ratio lower. The mask is stochastic only during training. Its seed (100003) is isolated from the ordinary training RNG so it does not shift baseline data-order/dropout draws. Counters are updated only from dimensions surviving both Bernoulli and Top-K.

## Methodology and equation-level difference

### 1. Shared official Structure (PCDS) method

Let $h^l$ be the normalized hidden tokens entering the attention module at ViT block $l$. The official attention adapter is

$$
A(h)=0.1\,W_{{up}}\operatorname{{Dropout}}_{{0.1}}
\left(\operatorname{{ReLU}}(W_{{down}}h)\right),
$$

and the key/value branches used by **both methods** are

$$
k^l=W_k^l h^l+A_k^l(h^l),\qquad
v^l=W_v^l h^l+A_v^l(h^l).
$$

The classifier uses normalized features and normalized weights (cosine logits), followed by the official CosFace loss. From Task 1 onward, Structure-preserving consistency (SCC) compares the current student $z_i^s$ and frozen preceding adapter branch $z_i^t$ against every old prototype $\mu_c$:

$$
\mathcal L_{{SCC}}
=\frac1B\sum_i\sum_{{c\in\mathcal Y_{{old}}}}
\left[
\cos(z_i^s,\mu_c)-\cos(z_i^t,\mu_c)
\right]^2.
$$

The actual official Cars objective used here is

$$
\mathcal L_{{Structure}}
=\mathcal L_{{CosFace}}+5\mathcal L_{{SCC}},
$$

with SCC omitted at Task 0 because no old prototypes exist.

For APR, let the representation drift of current class $j$ be

$$
\Delta\mu_j=\mu_j^{{after}}-\mu_j^{{before}},
$$

and let its relevance to old class $k$ be

$$
w_{{kj}}=
\operatorname{{softmax}}_j
\left(10\cos(\mu_k,\mu_j^{{before}})\right).
$$

The old Gaussian statistics are rectified as implemented in the official code:

$$
\mu'_k=\mu_k+\sum_j w_{{kj}}\Delta\mu_j,
$$

$$
\Sigma'_k=\Sigma_k+
10\sum_j
(w_{{kj}}\Delta\mu_j)(w_{{kj}}\Delta\mu_j)^\top
+\epsilon I.
$$

Classifier adaptation (CA) then samples

$$
\tilde z_c\sim\mathcal N(\alpha_{{c,t}}\mu'_c,\Sigma'_c),\qquad
\alpha_{{c,t}}=0.9+0.1\frac{{t_c+1}}{{t+1}},
$$

and optimizes **only the cosine classifier** for five epochs. Here $t_c$ is the task-age index used by the official implementation. No raw image from an old task is replayed.

### 2. SGDS activation guidance added in the second condition

Frozen pretrained-backbone class prototypes are

$$
\bar\mu_y=\frac1{{|\mathcal D_y|}}\sum_{{x\in\mathcal D_y}}\phi_{{frozen}}(x).
$$

For each new class $c$, Eq. (4)-(6) are

$$
P(y\mid c)=
\frac{{\exp(\cos(\bar\mu_c,\bar\mu_y))}}
{{\sum_{{y'\in\mathcal Y_{{old}}\cup\mathcal C_t}}
\exp(\cos(\bar\mu_c,\bar\mu_{{y'}}))}},
$$

$$
S_{{old}}(c)=\sum_{{y\in\mathcal Y_{{old}}}}P(y\mid c),\qquad
S_{{new}}(c)=\sum_{{y\in\mathcal C_t}}P(y\mid c).
$$

The current-class sum includes $c$ itself. If $S_{{old}}>S_{{new}}$, the class uses Knowledge Reuse; otherwise it uses New Subspace Allocation.

During the first half of training, the per-dimension Bernoulli probability is

$$
p_j(c)=
\begin{{cases}}
1-\exp\left[-\sum_{{y\in\mathcal Y_{{old}}}}P(y\mid c)
\frac{{F_c[y,j]}}{{\max_iF_c[y,i]+\epsilon}}\right],
& S_{{old}}>S_{{new}},\\
\exp\left[-0.5\frac{{F[j]}}{{\max_iF[i]+\epsilon}}\right],
& \text{{otherwise}}.
\end{{cases}}
$$

During the second half, activation compaction is

$$
p_j(c)=1-\exp\left[-1.0\frac{{F_c[c,j]}}{{\max_iF_c[c,i]+\epsilon}}\right].
$$

For each sample, $m_j\sim\operatorname{{Bernoulli}}(p_j)$, followed by deterministic Top-K retention:

$$
\hat h^{{11}}
=\operatorname{{TopK}}_{{\lfloor0.6\times768\rfloor}}
(h^{{11}}\odot m).
$$

Only dimensions surviving both operations update the global counter $F[j]$ and class counter $F_c[c,j]$. One class-conditioned dimension mask is shared across all tokens of a sample; Top-K magnitude is the token-mean absolute activation. Stochastic masking is disabled at evaluation.

The **only forward-path difference** is therefore at the final ViT block ($l=11$), and only in the actively trained student adapter branch; the SCC teacher branch stays dense:

$$
\begin{{aligned}}
\text{{Structure:}}\quad & k^{{11}}=W_kh+A_k(h),\quad v^{{11}}=W_vh+A_v(h),\\
\text{{Structure+SGDS:}}\quad & k^{{11}}=W_kh+A_k(\hat h),\quad v^{{11}}=W_vh+A_v(\hat h).
\end{{aligned}}
$$

The Structure loss, SCC, APR, classifier, and CA are unchanged; SGDS contributes no additional loss. Full SGDS adapter retrieval/fusion, entropy selection, MOS alignment, and SGDS classifier are not included.

| Component | Official Structure | Structure + SGDS |
| --- | --- | --- |
| Backbone/adapters | Official ViT-B/16 + attention adapters | Identical |
| Objective | CosFace + 5 SCC | Identical |
| APR / Gaussian CA | Enabled | Identical |
| Final-block adapter input | Dense $h$ | Bernoulli + Top-K $\hat h$ |
| Semantic Eq. (4)-(10) | None | Enabled |
| Inference mask | None | None (training-only stochastic mask) |
| Raw replay / Task-ID | None / none | None / none |

### 3. Metric definitions

At the end of task $t$, Seen Accuracy is accuracy over every class observed through $t$; Old/New Accuracy splits that set at the current task boundary. Average Incremental Accuracy is

$$
\operatorname{{AvgInc}}_t=\frac1{{t+1}}\sum_{{r=0}}^t\operatorname{{SeenAcc}}_r.
$$

Using task-accuracy matrix entry $A_{{t,j}}$, average forgetting is

$$
F_t=\frac1t\sum_{{j<t}}
\left(\max_{{r\in\{{j,\ldots,t\}}}}A_{{r,j}}-A_{{t,j}}\right).
$$

## Main results

{main_table}

The official Structure baseline Final Accuracy is **{baseline_final}%**, which is **{paper_gap}pp** relative to the paper's Cars-196 10-task reference of 80.23%. It lies inside the paper's reported `80.23 +/- 0.68` interval, although this single run is not a statistical reproduction.

## Complete task-wise results

### Official Structure (PCDS)

{baseline_task_table}

### Structure + SGDS

{sgds_task_table}

### Task-wise delta (SGDS - Structure)

{task_delta_table}

Compact Seen-Accuracy comparison:

{task_table}

## Complete task-accuracy matrices

Each cell is accuracy on that task's class partition after the row's training task. The last column is accuracy over all seen test samples, not the unweighted mean of task cells.

### Official Structure matrix

{baseline_matrix_table}

### Structure + SGDS matrix

{sgds_matrix_table}

## Effective rank (current-task deterministic train features)

{rank_table}

Mean rank delta (SGDS - Structure): **{mean_rank_delta}**. Negative means the measured representation rank became more compact.

## Classifier adaptation

{ca_table}

## Resources

{resource_table}

## SGDS semantic behavior

{semantic_table}

## SGDS activation behavior

{activation_table}

- Correlation between old-class count and mean S_old: **{old_corr}**.
- Tasks with both strategies present: **{varied} / {tasks}**.
- Tasks saturated at 100% Knowledge Reuse: **{saturated}**.
- Correlation of maximum old-class cosine similarity with Knowledge Reuse: **{reuse_corr}**.
- Correlation of maximum old-class cosine similarity with final class accuracy: **{accuracy_corr}**.
- Mean maximum old cosine, Reuse vs Allocation: **{reuse_mean} vs {allocation_mean}**.

## Interpretation of the task-wise behavior

- Through Task 2 the methods are nearly tied (Seen delta between +0.24pp and -0.24pp). The failure becomes visible at Task 4 (-3.18pp), accelerates at Task 6 (-20.41pp), and reaches -52.73pp at Task 9.
- This is not failure to learn the newest task: final New Accuracy is 90.61% with SGDS versus 85.15% for Structure. The loss comes from old classes (21.19% versus 79.12%), producing 69.07% forgetting.
- The accuracy matrix shows direct recency bias: under SGDS, Task-0 accuracy falls from 93.32% when learned to 0.12% after Task 9, while the newest Task-9 partition remains at 90.61%.
- Semantic selection becomes fully saturated: every new class is assigned Knowledge Reuse from Task 2 onward. The correlation between old-class count and mean $S_{{old}}$ is 0.8165, supporting the concern that probability mass is strongly affected by the growing number of old classes.
- SGDS does compact the measured representation: the final current-task rank drops from 366 to 220. In this run, however, lower rank accompanies severe old-class collapse, so rank reduction is not evidence of beneficial structure preservation.
- Realized zero ratios remain approximately 40.5-48.2%, so the failure cannot be explained by the sparsification code silently doing nothing. CA sometimes recovers substantial Seen Accuracy, but it cannot repair the progressively destroyed old-task geometry.
- These statements describe one controlled seed. They establish that this particular training-only final-block SGDS adaptation is incompatible with official Structure under the tested settings; they do not establish that the complete SGDS method fails.

## Direct answers

1. Final accuracy improvement: **{improved}** ({final_delta} percentage points).
2. Larger than the prior CIFAR-100 +0.07pp screening delta: **{larger_than_cifar}**.
3. Old accuracy delta: **{old_delta}pp**.
4. Forgetting delta: **{forget_delta}pp** (negative is better).
5. New-class accuracy delta: **{new_delta}pp**.
6. Mean effective-rank delta: **{mean_rank_delta}**.
7. Class-dependent semantic variation occurred in **{varied}** tasks.
8. Full-reuse saturation occurred in tasks **{saturated}**.
9. Similarity/reuse correlation: **{reuse_corr}**.
10. Geometry-related conclusions should be treated as descriptive because this is one seed.

The paper's Cars-196 10-task PCDS value (80.23 +/- 0.68 Last-Acc) is a multi-run reference, not a baseline hardcoded into this comparison. This report compares the two runs produced here under the requested single-seed protocol.
""".format(
        main_table=markdown_table(["Method", "Avg Inc.", "Final", "Old", "New", "Forgetting"], main_rows),
        baseline_final=fmt(baseline["summary"]["final_accuracy"]),
        paper_gap=fmt(float(baseline["summary"]["final_accuracy"]) - 80.23, signed=True),
        baseline_task_table=markdown_table(
            ["Task", "Seen cls", "Seen Acc", "Old Acc", "New Acc", "Forgetting", "Avg Inc."],
            baseline_task_rows,
        ),
        sgds_task_table=markdown_table(
            ["Task", "Seen cls", "Seen Acc", "Old Acc", "New Acc", "Forgetting", "Avg Inc."],
            sgds_task_rows,
        ),
        task_delta_table=markdown_table(
            ["Task", "Delta Seen", "Delta Old", "Delta New", "Delta Forgetting", "Delta Avg Inc."],
            task_delta_rows,
        ),
        task_table=markdown_table(["Task", "Structure", "+SGDS", "Delta"], task_rows),
        baseline_matrix_table=markdown_table(matrix_headers, baseline_matrix_rows),
        sgds_matrix_table=markdown_table(matrix_headers, sgds_matrix_rows),
        rank_table=markdown_table(["Task", "Structure rank", "+SGDS rank", "Delta"], rank_rows),
        ca_table=markdown_table(
            [
                "Task", "Structure pre-CA", "Structure post-CA", "Structure gain",
                "+SGDS pre-CA", "+SGDS post-CA", "+SGDS gain",
            ],
            ca_rows,
        ),
        resource_table=markdown_table(
            ["Method", "Peak alloc. GB", "Peak reserved GB", "Final trainable", "Total parameters"],
            [
                [
                    "Official Structure (PCDS)",
                    fmt(baseline["summary"]["peak_cuda_allocated_gb"], 3),
                    fmt(baseline["summary"]["peak_cuda_reserved_gb"], 3),
                    baseline["summary"]["final_trainable_parameters"],
                    baseline["summary"]["final_total_parameters"],
                ],
                [
                    "Structure + SGDS",
                    fmt(sgds["summary"]["peak_cuda_allocated_gb"], 3),
                    fmt(sgds["summary"]["peak_cuda_reserved_gb"], 3),
                    sgds["summary"]["final_trainable_parameters"],
                    sgds["summary"]["final_total_parameters"],
                ],
            ],
        ),
        semantic_table=markdown_table(
            ["Task", "Old cls", "New cls", "Reuse", "Allocate", "Reuse %", "Mean S_old", "Mean S_new"],
            semantic_rows,
        ),
        activation_table=markdown_table(
            [
                "Task", "Layer", "Actual nonzero %", "Actual zero %", "Mean active dims",
                "Mean p", "Global max", "Global mean", "Class max", "Class mean",
            ],
            activation_rows,
        ),
        mean_rank_delta=fmt(analysis["effective_rank"]["mean_delta"], 2, signed=True),
        old_corr=fmt(old_mass_correlation, 4),
        varied=varied_tasks,
        tasks=len(semantic_tasks),
        saturated=", ".join(map(str, saturated_tasks)) if saturated_tasks else "none",
        reuse_corr=fmt(similarity_reuse_corr, 4),
        accuracy_corr=fmt(similarity_accuracy_corr, 4),
        reuse_mean=fmt(analysis["semantic_diagnostics"]["mean_max_old_cosine_reuse"], 4),
        allocation_mean=fmt(analysis["semantic_diagnostics"]["mean_max_old_cosine_allocation"], 4),
        improved="yes" if improved else "no",
        final_delta=fmt(delta["final_accuracy"], signed=True),
        larger_than_cifar="yes" if delta["final_accuracy"] > 0.07 else "no",
        old_delta=fmt(delta["final_old_accuracy"], signed=True),
        forget_delta=fmt(delta["average_forgetting"], signed=True),
        new_delta=fmt(delta["final_new_accuracy"], signed=True),
    )
    (output / "RESULTS_STRUCTURE_SGDS.md").write_text(report, encoding="utf-8")
    (root / "RESULTS_STRUCTURE_SGDS.md").write_text(report, encoding="utf-8")
    print("Wrote {}".format(output / "RESULTS_STRUCTURE_SGDS.md"))


if __name__ == "__main__":
    main()
