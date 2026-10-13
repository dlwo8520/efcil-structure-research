#!/usr/bin/env python3
"""Regenerate the complete PTM-guided task-wise Markdown report."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

try:
    from plot_ptm_guided_results import generate_plots
except ImportError:
    from scripts.plot_ptm_guided_results import generate_plots


DATASETS = [
    ("imageneta", "ImageNet-A", [5, 10, 20]),
    ("cars196", "Cars-196", [5, 10, 20]),
    ("objectnet", "ObjectNet", [10]),
    ("omnibenchmark", "OmniBenchmark", [10]),
    ("imagenetr", "ImageNet-R", [10]),
]
METHODS = ["adapter_baseline", "ptm_place_only", "ptm_place_move"]
METHOD_LABELS = {
    "adapter_baseline": "Adapter Baseline",
    "ptm_place_only": "PTM Place Only",
    "ptm_place_move": "PTM Place + Move",
}
SEEDS = [1993, 1996, 1997]
LOSS_KEYS = [
    "classification_loss", "placement_loss", "movement_loss", "total_loss"
]


def methods_for_protocol(dataset, tasks):
    if tasks == 10 and dataset in {"imageneta", "cars196"}:
        return METHODS
    return ["ptm_place_move"]


def format_stat(values, digits=2):
    if not values:
        return "PENDING"
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    pattern = "{:.%df} ± {:.%df}" % (digits, digits)
    return pattern.format(statistics.mean(values), deviation)


def mean(values):
    return statistics.mean(values) if values else float("nan")


def load_runs(root, dataset, tasks, method):
    runs = {}
    base = root / dataset / "{}_tasks".format(tasks) / method
    for seed in SEEDS:
        path = base / "seed_{}".format(seed) / "metrics.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("status") != "COMPLETE" or len(data.get("task_metrics", [])) != tasks:
            continue
        runs[seed] = data
    return runs


def task_values(runs, task_index, key, scale=1.0):
    values = []
    for run in runs.values():
        rows = run.get("task_metrics", [])
        if task_index < len(rows) and rows[task_index].get(key) is not None:
            values.append(float(rows[task_index][key]) * scale)
    return values


def load_task_loss_stats(root, dataset, tasks, method, runs):
    """Return per-task lists of per-seed epoch-mean loss values."""
    result = {task: {key: [] for key in LOSS_KEYS} for task in range(tasks)}
    for seed in runs:
        path = (
            root / dataset / "{}_tasks".format(tasks) / method
            / "seed_{}".format(seed) / "loss_components.csv"
        )
        if not path.is_file():
            continue
        grouped = defaultdict(lambda: defaultdict(list))
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                task = int(row["task"])
                for key in LOSS_KEYS:
                    grouped[task][key].append(float(row[key]))
        for task, values in grouped.items():
            for key in LOSS_KEYS:
                if values[key]:
                    result[task][key].append(statistics.mean(values[key]))
    return result


def append_methodology(lines):
    lines += [
        "## 2. 방법론과 목적함수",
        "",
        "### 2.1 공통 adapter와 classification objective",
        "",
        "Frozen ViT-B/16-IN21K backbone에 bottleneck adapter만 학습한다. layer $\\ell$의 비선형 residual adapter는 다음과 같다.",
        "",
        r"$$A_{\ell,\phi}(h)=s_\ell W^{\ell}_{\mathrm{up}}\,\mathrm{ReLU}(W^{\ell}_{\mathrm{down}}h+b^{\ell}_{\mathrm{down}})+b^{\ell}_{\mathrm{up}}.$$",
        "",
        "분류기는 기존 코드와 동일한 CosFace objective를 사용한다.",
        "",
        r"$$\mathcal L_{\mathrm{cls}}=-\frac1B\sum_{i=1}^{B}\log\frac{\exp(s(\cos\theta_{y_i}-m))}{\exp(s(\cos\theta_{y_i}-m))+\sum_{j\ne y_i}\exp(s\cos\theta_j)}.$$",
        "",
        "### 2.2 Frozen-PTM placement prior",
        "",
        "클래스 $c$가 처음 등장할 때 adapter를 명시적으로 끈 frozen PTM feature로 prototype을 한 번 계산한다.",
        "",
        r"$$p_c=\operatorname{Norm}\!\left(\frac{1}{|D_c|}\sum_{x\in D_c}f_{\theta_0}(x;\,\text{adapter off})\right).$$",
        "",
        "학습 batch에서 adapter feature의 클래스 평균을 $\\bar z_c$라 하면 placement loss는",
        "",
        r"$$\bar z_c=\operatorname{Norm}\!\left(\frac{1}{|B_c|}\sum_{i:y_i=c}f_{\theta_0,\phi}(x_i)\right),\qquad \mathcal L_{\mathrm{place}}=\frac1{|C_B|}\sum_{c\in C_B}\left(1-\cos(\bar z_c,p_c)\right).$$",
        "",
        "이며 강도는 학습 후반에 0으로 감소한다.",
        "",
        r"$$\lambda_{\mathrm{place}}(e)=\lambda_f+\frac{\lambda_0-\lambda_f}{2}\left(1+\cos\frac{\pi e}{E-1}\right),\qquad \lambda_f=0.$$",
        "",
        "### 2.3 Old-anchor functional movement constraint",
        "",
        "클래스가 학습될 때 각 adapter 입력의 클래스 평균만 anchor로 저장한다. raw image나 sample feature는 보존하지 않는다.",
        "",
        r"$$a_{\ell,c}=\frac1{|D_c|}\sum_{x\in D_c}h_\ell(x),\qquad r^{t-1}_{\ell,c}=A_{\ell,\phi^{t-1}}(a_{\ell,c}).$$",
        "",
        "다음 task 시작 시 old anchor에서의 직전 adapter response를 target으로 고정하고 현재 response 변화만 soft하게 제한한다.",
        "",
        r"$$\mathcal L_{\mathrm{move}}=\frac1{|\mathcal A_{\mathrm{old}}|}\sum_{(\ell,c)\in\mathcal A_{\mathrm{old}}}\operatorname{SmoothL1}\!\left(A_{\ell,\phi^t}(a_{\ell,c}),r^{t-1}_{\ell,c}\right).$$",
        "",
        "Task 1에는 old anchor가 없으므로 $\\mathcal L_{\\mathrm{move}}=0$이다.",
        "",
        "### 2.4 세 ablation의 정확한 차이",
        "",
        r"$$\begin{aligned}\text{Adapter Baseline:}\quad &\mathcal L=\mathcal L_{\mathrm{cls}},\\\text{PTM Place Only:}\quad &\mathcal L=\mathcal L_{\mathrm{cls}}+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}},\\\text{PTM Place+Move:}\quad &\mathcal L=\mathcal L_{\mathrm{cls}}+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}}+\lambda_{\mathrm{move}}\mathcal L_{\mathrm{move}}.\end{aligned}$$",
        "",
        "| Method | 직접 제약하는 대상 | 사용하는 old 정보 | 시점 |",
        "|---|---|---|---|",
        "| Adapter Baseline | new-class classification | 없음 | task 학습 중 |",
        "| PTM Place Only | new class의 adapted mean과 frozen PTM 위치 | 없음 | task 학습 중, cosine decay |",
        "| PTM Place + Move | placement + old anchor에서 adapter 기능 변화 | class/layer 평균 anchor와 response | 다음 task 학습 중 |",
        "| 기존 CDSP | $D Q_{old}=0$ gradient hard projection | old discriminative subspace | optimizer update 전/후 |",
        "",
        "### 2.5 Structure와의 수식적 차이",
        "",
        "Structure의 SCC는 현재 task sample을 reference point, old prototype을 anchor로 삼아 old/new model의 angular relation vector를 보존한다.",
        "",
        r"$$r_i^{t}=[\cos(f_{\theta^t}(x_i),\mu_1^{t-1}),\ldots,\cos(f_{\theta^t}(x_i),\mu_K^{t-1})]^\top,$$",
        "",
        r"$$\mathcal L_{\mathrm{SCC}}=\frac1B\sum_{i=1}^{B}\|r_i^t-r_i^{t-1}\|_2^2,\qquad \mathcal L_{\mathrm{Structure}}=\mathcal L_{\mathrm{CE}}+\lambda\mathcal L_{\mathrm{SCC}}.$$",
        "",
        "APR은 new-class prototype의 관측 이동 $\\delta_j=\\mu_j^t-\\mu_j^{t-1}$을 이용해 old class $k$의 이동을 추정한다.",
        "",
        r"$$\Delta_k=\sum_{j\in C_t}w_{k,j}\delta_j,\qquad w_{k,j}=\frac{\exp(\gamma s_{k,j})}{\sum_{l\in C_t}\exp(\gamma s_{k,l})},$$",
        "",
        r"$$\mu_k^t=\mu_k^{t-1}+\Delta_k,\qquad \Sigma_k^t=\Sigma_k^{t-1}+\rho\Delta_k\Delta_k^\top+\epsilon I.$$",
        "",
        "그 후 rectified distribution에서 $\\tilde z_k\\sim\\mathcal N(\\mu_k^t,\\Sigma_k^t)$를 sampling해 classifier를 다시 학습한다. 즉 Structure는 SCC로 학습 중 sample-to-prototype 관계를 유지하고 APR과 synthetic feature로 사후 분포·classifier를 보정한다. 본 PTM 방법은 APR/SCC/synthetic CA 없이, new class에는 frozen-PTM 위치 prior를 주고 old class에는 nonlinear adapter response의 국소 변화만 제한한다.",
        "",
        "모든 조건은 exemplar-free이며 raw old image replay를 사용하지 않는다.",
        "",
    ]


def render(reference, result_root, figure_root=Path("figures/ptm_guided")):
    lines = [
        "# PTM-Guided Adapter vs Structure (CVPR 2026)",
        "",
        "> 이 문서는 `scripts/summarize_ptm_guided_results.py`로 생성된다. Structure 수치는 논문 reference이며, 이 repository에서 실행한 결과가 아니다.",
        "",
        "## 1. 실험 가설",
        "",
        "Frozen ViT-B/16-IN21K geometry를 new-class placement prior로 사용하고, bottleneck adapter가 classification에 필요한 만큼만 representation을 refine하게 한다. 이후 task에서는 old class layer anchor에서 비선형 adapter response의 과도한 변화만 soft constraint로 억제한다.",
        "",
    ]
    append_methodology(lines)
    lines += [
        "## 3. Dataset/task protocol",
        "",
        "| Dataset | Classes | Tasks | Per-task class sizes |",
        "|---|---:|---:|---|",
        "| ImageNet-A | 200 | 5 / 10 / 20 | 40 / 20 / 10 |",
        "| Cars-196 | 196 | 5 / 10 / 20 | 40×4+36 / 20×9+16 / 10×19+6 |",
        "| ObjectNet | 200 | 10 | 20 |",
        "| OmniBenchmark | 300 | 10 | 30 |",
        "| ImageNet-R | 200 | 10 | 20 |",
        "",
        "각 config는 class-order seed 1993, 1996, 1997을 실행한다. Last-Acc는 해당 stage 종료 후 모든 seen class accuracy, Inc-Acc는 그 stage까지의 Last-Acc 평균이다.",
        "",
        "## 4. Structure CVPR 2026 reference",
        "",
        "| Dataset | Tasks | Last-Acc | Inc-Acc |",
        "|---|---:|---:|---:|",
    ]
    for key, label, tasks_list in DATASETS:
        for tasks in tasks_list:
            ref = reference["results"][key][str(tasks)]
            lines.append(
                "| {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} |".format(
                    label, tasks, ref["last_acc_mean"], ref["last_acc_std"],
                    ref["inc_acc_mean"], ref["inc_acc_std"],
                )
            )

    lines += [
        "",
        "## 5. 최종 결과 요약 (3 seeds, mean ± sample std)",
        "",
        "| Dataset | Tasks | Method | Seeds | Last-Acc | Inc-Acc | Our - Structure Last | Our - Structure Inc |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    run_cache = {}
    for key, label, tasks_list in DATASETS:
        for tasks in tasks_list:
            methods = methods_for_protocol(key, tasks)
            ref = reference["results"][key][str(tasks)]
            for method in methods:
                runs = load_runs(result_root, key, tasks, method)
                run_cache[(key, tasks, method)] = runs
                last_values = [run["last_accuracy"] for run in runs.values()]
                inc_values = [run["incremental_accuracy"] for run in runs.values()]
                if method == "ptm_place_move" and last_values:
                    last_delta = "{:+.2f}".format(mean(last_values) - ref["last_acc_mean"])
                    inc_delta = "{:+.2f}".format(mean(inc_values) - ref["inc_acc_mean"])
                else:
                    last_delta = inc_delta = "PENDING" if not last_values else "N/A"
                lines.append(
                    "| {} | {} | `{}` | {}/3 | {} | {} | {} | {} |".format(
                        label, tasks, method, len(runs), format_stat(last_values),
                        format_stat(inc_values), last_delta, inc_delta,
                    )
                )

    lines += ["", "### 5.1 Phase 1 결과 해석", ""]
    for dataset, label in (("imageneta", "ImageNet-A"), ("cars196", "Cars-196")):
        baseline = run_cache[(dataset, 10, "adapter_baseline")]
        if len(baseline) != len(SEEDS):
            continue
        base_last = mean([run["last_accuracy"] for run in baseline.values()])
        base_inc = mean([run["incremental_accuracy"] for run in baseline.values()])
        base_forget = mean([run["forgetting"] for run in baseline.values()])
        lines.append("- **{}**".format(label))
        for method in ("ptm_place_only", "ptm_place_move"):
            runs = run_cache[(dataset, 10, method)]
            if len(runs) != len(SEEDS):
                continue
            delta_last = mean([run["last_accuracy"] for run in runs.values()]) - base_last
            delta_inc = mean([run["incremental_accuracy"] for run in runs.values()]) - base_inc
            delta_forget = mean([run["forgetting"] for run in runs.values()]) - base_forget
            lines.append(
                "  - `{}` vs baseline: Last {:+.2f}, Inc {:+.2f}, Forgetting {:+.2f}.".format(
                    method, delta_last, delta_inc, delta_forget
                )
            )
    lines += [
        "",
        "ImageNet-A에서는 PTM constraint가 representation drift를 줄였지만 정확도 향상으로 이어지지 않았다. Cars-196에서는 placement가 Last/Inc-Acc와 forgetting을 개선했으나, movement를 추가한 이득은 Place Only 대비 매우 작다.",
        "",
        "### 5.2 결과 그래프",
        "",
        "색 기준이 서로 다르므로 주의한다. 최종 막대그래프는 **Structure 논문값**을 기준으로 초록/빨강을 표시한다. 10-task 비교 heatmap은 **동일 실행의 Adapter Baseline** 대비 변화다. 5/20-task는 대응 Adapter run과 Structure task matrix가 없으므로, 각 평가 task를 처음 학습했을 때의 정확도 대비 이후 유지·망각을 표시한다. 따라서 5/20-task heatmap의 초록은 Structure 대비 우수하다는 뜻이 아니다.",
        "",
        "#### 5/10/20-task 최종 성능 vs Structure",
        "",
        "파란 막대는 Structure 논문값이고, PTM Place+Move 막대는 Structure보다 높으면 초록, 낮으면 빨강이다. 괄호는 `PTM − Structure` 차이다.",
        "",
        "![PTM Place and Move across all completed task granularities compared with Structure]({})".format(
            (figure_root / "all_tasks_final_vs_structure.png").as_posix()
        ),
        "",
        "#### 10-task ablation 최종 성능 vs Structure",
        "",
        "![10-task ablation final performance compared with Structure]({})".format(
            (figure_root / "phase1_final_vs_structure.png").as_posix()
        ),
        "",
        "#### ImageNet-A 5-task 유지·망각",
        "",
        "![ImageNet-A 5-task PTM Place and Move retention heatmap]({})".format(
            (figure_root / "imageneta_5task_ptm_place_move_retention_heatmap.png").as_posix()
        ),
        "",
        "#### ImageNet-A 10-task method별 Adapter 대비 변화",
        "",
        "![ImageNet-A PTM Place Only versus Adapter Baseline task heatmap]({})".format(
            (figure_root / "imageneta_ptm_place_only_vs_adapter_task_heatmap.png").as_posix()
        ),
        "",
        "![ImageNet-A PTM Place and Move versus Adapter Baseline task heatmap]({})".format(
            (figure_root / "imageneta_ptm_place_move_vs_adapter_task_heatmap.png").as_posix()
        ),
        "",
        "#### ImageNet-A 20-task 유지·망각",
        "",
        "![ImageNet-A 20-task PTM Place and Move retention heatmap]({})".format(
            (figure_root / "imageneta_20task_ptm_place_move_retention_heatmap.png").as_posix()
        ),
        "",
        "#### Cars-196 5-task 유지·망각",
        "",
        "![Cars-196 5-task PTM Place and Move retention heatmap]({})".format(
            (figure_root / "cars196_5task_ptm_place_move_retention_heatmap.png").as_posix()
        ),
        "",
        "#### Cars-196 10-task method별 Adapter 대비 변화",
        "",
        "![Cars-196 PTM Place Only versus Adapter Baseline task heatmap]({})".format(
            (figure_root / "cars196_ptm_place_only_vs_adapter_task_heatmap.png").as_posix()
        ),
        "",
        "![Cars-196 PTM Place and Move versus Adapter Baseline task heatmap]({})".format(
            (figure_root / "cars196_ptm_place_move_vs_adapter_task_heatmap.png").as_posix()
        ),
        "",
        "#### Cars-196 20-task 유지·망각",
        "",
        "![Cars-196 20-task PTM Place and Move retention heatmap]({})".format(
            (figure_root / "cars196_20task_ptm_place_move_retention_heatmap.png").as_posix()
        ),
        "",
        "## 6. 모든 stage별 결과 (3 seeds mean ± std)",
        "",
        "`Stage t`는 t번째 task까지 학습한 직후를 뜻한다. Old/New은 각각 이전 task 클래스와 현재 task 클래스 정확도다.",
        "",
    ]
    stage_section = 0
    for dataset, label in (("imageneta", "ImageNet-A"), ("cars196", "Cars-196")):
        for tasks in (5, 10, 20):
            available = [
                (method, run_cache[(dataset, tasks, method)])
                for method in methods_for_protocol(dataset, tasks)
                if run_cache[(dataset, tasks, method)]
            ]
            if not available:
                continue
            stage_section += 1
            lines += [
                "### 6.{} {} {}-task".format(stage_section, label, tasks), ""
            ]
            for method, runs in available:
                lines += [
                    "#### `{}`".format(method),
                    "",
                    "| Stage | Seen classes | Last-Acc | Inc-Acc | Top-5 | Old Acc | New Acc | Forgetting |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
                for task in range(tasks):
                    seen = int(task_values(runs, task, "seen_classes")[0])
                    lines.append(
                        "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                            task + 1, seen,
                            format_stat(task_values(runs, task, "last_accuracy")),
                            format_stat(task_values(runs, task, "incremental_accuracy")),
                            format_stat(task_values(runs, task, "top5_accuracy")),
                            format_stat(task_values(runs, task, "old_accuracy")),
                            format_stat(task_values(runs, task, "new_accuracy")),
                            format_stat(task_values(runs, task, "forgetting")),
                        )
                    )
                lines.append("")

    lines += [
        "## 7. Task accuracy matrix (3-seed mean ± std)",
        "",
        "행은 학습을 마친 stage, 열은 평가 대상 task다. `—`는 아직 학습하지 않은 task를 뜻한다.",
        "",
    ]
    for dataset, label in (("imageneta", "ImageNet-A"), ("cars196", "Cars-196")):
        for tasks in (5, 10, 20):
            for method in methods_for_protocol(dataset, tasks):
                runs = run_cache[(dataset, tasks, method)]
                if not runs:
                    continue
                lines += [
                    "### {} {}-task / `{}`".format(label, tasks, method),
                    "",
                    "| Trained through | " + " | ".join(
                        "Eval T{}".format(i) for i in range(1, tasks + 1)
                    ) + " |",
                    "|---:|" + "---:|" * tasks,
                ]
                for row in range(tasks):
                    cells = []
                    for column in range(tasks):
                        values = [
                            float(run["task_accuracy_matrix"][row][column])
                            for run in runs.values()
                            if row < len(run["task_accuracy_matrix"])
                            and column < len(run["task_accuracy_matrix"][row])
                        ]
                        cells.append(format_stat(values, digits=1) if values else "—")
                    lines.append(
                        "| T{} | {} |".format(row + 1, " | ".join(cells))
                    )
                lines.append("")

    lines += [
        "## 8. 모든 stage별 loss, drift, 시간 및 메모리",
        "",
        "Loss는 각 seed에서 해당 stage의 epoch 평균을 먼저 구한 뒤 3 seeds로 집계했다. 시간은 prototype 수집·학습·통계·평가를 포함한 stage wall time이며, GPU memory는 process peak allocated memory다. 여러 config가 GPU 3에서 동시에 실행된 구간이 있으므로 wall time에는 GPU 경합 영향이 포함될 수 있다.",
        "",
    ]
    for dataset, label in (("imageneta", "ImageNet-A"), ("cars196", "Cars-196")):
        for tasks in (5, 10, 20):
            available = [
                (method, run_cache[(dataset, tasks, method)])
                for method in methods_for_protocol(dataset, tasks)
                if run_cache[(dataset, tasks, method)]
            ]
            if not available:
                continue
            lines += ["### {} {}-task diagnostics".format(label, tasks), ""]
            for method, runs in available:
                loss_stats = load_task_loss_stats(
                    result_root, dataset, tasks, method, runs
                )
                lines += [
                    "#### `{}`".format(method),
                    "",
                    "| Stage | Cls loss | Place loss | Move loss | Total loss | PTM proto drift | PTM geometry | Anchor drift | Time (min) | Peak GPU (GiB) |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
                for task in range(tasks):
                    losses = loss_stats[task]
                    lines.append(
                        "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                            task + 1,
                            format_stat(losses["classification_loss"], 4),
                            format_stat(losses["placement_loss"], 4),
                            format_stat(losses["movement_loss"], 6),
                            format_stat(losses["total_loss"], 4),
                            format_stat(task_values(runs, task, "mean_ptm_prototype_drift"), 4),
                            format_stat(task_values(runs, task, "ptm_geometry_drift"), 4),
                            format_stat(task_values(runs, task, "mean_anchor_response_drift"), 4),
                            format_stat(task_values(runs, task, "task_elapsed_seconds", 1.0 / 60.0), 1),
                            format_stat(task_values(runs, task, "peak_gpu_memory_bytes", 1.0 / (1024 ** 3)), 2),
                        )
                    )
                lines.append("")

    lines += ["## 9. Seed별 최종 결과", ""]
    any_seed = False
    for (dataset, tasks, method), runs in run_cache.items():
        for seed, run in sorted(runs.items()):
            any_seed = True
            lines.append(
                "- `{}/{}/{} seed {}`: Last-Acc {:.2f}, Inc-Acc {:.2f}, Forgetting {:.2f}".format(
                    dataset, tasks, method, seed, run["last_accuracy"],
                    run["incremental_accuracy"], run.get("forgetting", float("nan")),
                )
            )
    if not any_seed:
        lines.append("PENDING")

    lines += ["", "## 10. 최종-stage forgetting / geometry drift (seed별)", ""]
    diagnostics = []
    for key, runs in run_cache.items():
        for seed, run in runs.items():
            diagnostics.append((key, seed, run["task_metrics"][-1]))
    if diagnostics:
        lines += [
            "| Dataset/Tasks/Method | Seed | Forgetting | PTM prototype drift | PTM geometry drift | Anchor response drift |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key, seed, final in diagnostics:
            lines.append(
                "| `{}/{}/{}` | {} | {:.4f} | {:.6f} | {:.6f} | {:.6f} |".format(
                    key[0], key[1], key[2], seed, final.get("forgetting", 0.0),
                    final.get("mean_ptm_prototype_drift", 0.0),
                    final.get("ptm_geometry_drift", 0.0),
                    final.get("mean_anchor_response_drift", 0.0),
                )
            )
    else:
        lines.append("PENDING")

    lines += ["", "## 11. 아직 실행되지 않은 실험", ""]
    pending = []
    for key, label, tasks_list in DATASETS:
        for tasks in tasks_list:
            methods = methods_for_protocol(key, tasks)
            for method in methods:
                runs = run_cache[(key, tasks, method)]
                missing = [str(seed) for seed in SEEDS if seed not in runs]
                if missing:
                    pending.append(
                        "- {} {}-task `{}`: missing seeds {}".format(
                            label, tasks, method, ", ".join(missing)
                        )
                    )
    lines.extend(pending or ["모든 예정 실험 완료"])
    lines += [
        "",
        "## 12. Reference",
        "",
        "- {}. [{}]({})".format(reference["paper"], reference["venue"], reference["source"]),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default="results/ptm_guided")
    parser.add_argument("--reference", default="references/structure_cvpr2026_results.json")
    parser.add_argument("--output", default="RESULTS_PTM_GUIDED_VS_STRUCTURE.md")
    parser.add_argument("--figure-root", default="figures/ptm_guided")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    generate_plots(
        Path(args.result_root), Path(args.reference), Path(args.figure_root)
    )
    output = render(reference, Path(args.result_root), Path(args.figure_root))
    Path(args.output).write_text(output, encoding="utf-8")
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
