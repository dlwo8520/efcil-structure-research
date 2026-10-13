from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from classifier_diagnostic.pipeline import read_csv, write_csv


def _truth(row: dict[str, Any], key: str) -> bool:
    return str(row[key]).lower() == "true"


def _float(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _bootstrap_interval(repair: np.ndarray, harm: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    if not repair.size:
        return float("nan"), float("nan")
    values = repair.astype(np.float64) - harm.astype(np.float64)
    rng = np.random.RandomState(seed)
    means = np.empty(repeats, dtype=np.float64)
    for start in range(0, repeats, 128):
        end = min(start + 128, repeats)
        indices = rng.randint(0, len(values), (end - start, len(values)))
        means[start:end] = values[indices].mean(axis=1) * 100.0
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def _paired(rows: list[dict[str, Any]], winner: str, loser: str,
            total: int, repeats: int, seed: int, label: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"group": label, "n": 0}
    correct_w = np.asarray([_truth(row, f"correct_{winner}") for row in rows], dtype=bool)
    correct_l = np.asarray([_truth(row, f"correct_{loser}") for row in rows], dtype=bool)
    pred_w = np.asarray([int(row[f"pred_{winner}"]) for row in rows])
    pred_l = np.asarray([int(row[f"pred_{loser}"]) for row in rows])
    repair = correct_w & ~correct_l
    harm = ~correct_w & correct_l
    accuracy_w = 100.0 * correct_w.mean()
    accuracy_l = 100.0 * correct_l.mean()
    repair_pct = 100.0 * repair.mean()
    harm_pct = 100.0 * harm.mean()
    net_pct = repair_pct - harm_pct
    if not np.isclose(net_pct, accuracy_w - accuracy_l, rtol=0, atol=1e-10):
        raise RuntimeError(f"{label}: Repair-Harm != paired accuracy difference")
    low, high = _bootstrap_interval(repair, harm, repeats, seed)
    labels = np.asarray([int(row["true_label"]) for row in rows])
    class_deltas = [100.0 * (repair[labels == cls].mean() - harm[labels == cls].mean())
                    for cls in np.unique(labels)]
    return {
        "group": label, "n": n, "accuracy_winner": accuracy_w,
        "accuracy_loser": accuracy_l, "accuracy_delta_pp": accuracy_w - accuracy_l,
        "repair_count": int(repair.sum()), "harm_count": int(harm.sum()),
        "repair_pct": repair_pct, "harm_pct": harm_pct, "net_gain_pp": net_pct,
        "same_correct_pct": 100.0 * np.mean(correct_w & correct_l),
        "same_wrong_pct": 100.0 * np.mean(~correct_w & ~correct_l),
        "prediction_disagreement_pct": 100.0 * np.mean(pred_w != pred_l),
        "class_macro_delta_pp": float(np.mean(class_deltas)),
        "contribution_to_all_pp": net_pct * n / total,
        "net_gain_bootstrap_ci_low_pp": low,
        "net_gain_bootstrap_ci_high_pp": high,
        "regression_check": "PASS",
    }


def _subset(rows: list[dict[str, Any]], predicate) -> list[dict[str, Any]]:
    return [row for row in rows if predicate(row)]


def _support_alignment(rows: list[dict[str, Any]], winner: str, loser: str) -> list[dict[str, Any]]:
    names = {
        "repair": lambda r: _truth(r, f"correct_{winner}") and not _truth(r, f"correct_{loser}"),
        "harm": lambda r: not _truth(r, f"correct_{winner}") and _truth(r, f"correct_{loser}"),
        "both_correct": lambda r: _truth(r, f"correct_{winner}") and _truth(r, f"correct_{loser}"),
        "both_wrong": lambda r: not _truth(r, f"correct_{winner}") and not _truth(r, f"correct_{loser}"),
    }
    result = []
    for name, predicate in names.items():
        members = _subset(rows, predicate)
        item: dict[str, Any] = {"comparison": winner + loser, "prediction_group": name, "n": len(members)}
        for metric, key in (("support_loss", f"delta_{loser}{winner}_support"),
                            ("support_winner", f"support_{winner}"),
                            ("support_loser", f"support_{loser}")):
            values = np.asarray([_float(row, key) for row in members], dtype=float)
            if metric == "support_loss":
                values = -values
            item.update({
                f"{metric}_mean": float(np.mean(values)) if len(values) else float("nan"),
                f"{metric}_median": float(np.median(values)) if len(values) else float("nan"),
                f"{metric}_q25": float(np.quantile(values, .25)) if len(values) else float("nan"),
                f"{metric}_q75": float(np.quantile(values, .75)) if len(values) else float("nan"),
            })
        result.append(item)
    return result


def _comparison(rows: list[dict[str, Any]], winner: str, loser: str,
                repeats: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    comp = winner + loser
    n_all = len(rows)
    all_row = _paired(rows, winner, loser, n_all, repeats, seed, "all")
    old = _paired(_subset(rows, lambda r: r["old_or_new"] == "old"), winner, loser,
                  n_all, repeats, seed + 1, "old")
    new = _paired(_subset(rows, lambda r: r["old_or_new"] == "new"), winner, loser,
                  n_all, repeats, seed + 2, "new")
    flip_col = "signflip_AB" if comp == "AB" else "signflip_BC"
    flip = _paired(_subset(rows, lambda r: _truth(r, flip_col)), winner, loser,
                   n_all, repeats, seed + 3, "support_signflip")
    loss_col = f"delta_{loser}{winner}_support"
    losses = np.asarray([-_float(row, loss_col) for row in rows], dtype=float)
    threshold = float(np.quantile(losses, .8))
    high_rows = [row for row, loss in zip(rows, losses) if loss >= threshold]
    high = _paired(high_rows, winner, loser, n_all, repeats, seed + 4, "support_loss_top20")
    high["threshold_support_loss"] = threshold
    high["threshold_source"] = "test support distribution q80; correctness never used"
    high_old = _paired(_subset(high_rows, lambda r: r["old_or_new"] == "old"), winner, loser,
                       n_all, repeats, seed + 5, "support_loss_top20_old")
    high_new = _paired(_subset(high_rows, lambda r: r["old_or_new"] == "new"), winner, loser,
                       n_all, repeats, seed + 6, "support_loss_top20_new")
    quintile_edges = np.quantile(losses, [0, .2, .4, .6, .8, 1.0])
    quantile_rows = []
    for q in range(5):
        mask = losses >= quintile_edges[q]
        mask &= losses <= quintile_edges[q + 1] if q == 4 else losses < quintile_edges[q + 1]
        subset = [row for row, selected in zip(rows, mask) if selected]
        item = _paired(subset, winner, loser, n_all, repeats, seed + 10 + q, f"Q{q + 1}")
        item.update({"support_loss_lower": float(quintile_edges[q]),
                     "support_loss_upper": float(quintile_edges[q + 1])})
        quantile_rows.append(item)
    class_rows = []
    for class_id in sorted({int(row["class_id"]) for row in rows}):
        members = _subset(rows, lambda r, cls=class_id: int(r["class_id"]) == cls)
        item = _paired(members, winner, loser, n_all, repeats, seed + 100 + class_id,
                       f"class_{class_id}")
        item.update({"class_id": class_id, "old_or_new": members[0]["old_or_new"]})
        class_rows.append(item)
    for collection in ([all_row, old, new, flip, high, high_old, high_new],
                       quantile_rows, class_rows):
        for item in collection:
            item["comparison"] = comp
            item["winner"] = winner
            item["loser"] = loser
    return {
        "main": [all_row], "old_new": [old, new], "signflip": [all_row, flip],
        "high_loss": [high, high_old, high_new], "quantiles": quantile_rows,
        "class": class_rows, "alignment": _support_alignment(rows, winner, loser),
    }


def _plot_bars(rows: list[dict[str, Any]], path: Path, title: str, label_key: str = "group") -> None:
    labels = [str(row[label_key]) for row in rows if row.get("n", 0)]
    values = [row for row in rows if row.get("n", 0)]
    if not values:
        return
    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(max(6, len(values) * 1.25), 4.2))
    ax.bar(x - .24, [r["repair_pct"] for r in values], .24, label="Repair")
    ax.bar(x, [r["harm_pct"] for r in values], .24, label="Harm")
    ax.bar(x + .24, [r["net_gain_pp"] for r in values], .24, label="Net gain")
    ax.axhline(0, linewidth=.8, color="black")
    ax.set_xticks(x, labels, rotation=20)
    ax.set(ylabel="percent / percentage points", title=title)
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _protocol_analysis(root: Path, protocol: str, repeats: int, seed: int) -> dict[str, Any]:
    output = root / protocol
    rows = read_csv(output / "paired_source_prediction_table.csv")
    if not rows:
        raise RuntimeError(f"{protocol}: no sample-level paired rows")
    if len(set(row["sample_id"] for row in rows)) != len(rows):
        raise RuntimeError(f"{protocol}: duplicate sample IDs")
    ab = _comparison(rows, "A", "B", repeats, seed)
    bc = _comparison(rows, "B", "C", repeats, seed + 10000)
    for name, result in (("ab", ab), ("bc", bc)):
        write_csv(output / f"{name}_summary.csv", result["main"])
    write_csv(output / "signflip_summary.csv", ab["signflip"] + bc["signflip"])
    write_csv(output / "support_loss_quantile_summary.csv", ab["quantiles"] + bc["quantiles"])
    write_csv(output / "class_macro_summary.csv", ab["class"] + bc["class"])
    write_csv(output / "old_new_summary.csv", ab["old_new"] + bc["old_new"])
    write_csv(output / "high_loss_summary.csv", ab["high_loss"] + bc["high_loss"])
    write_csv(output / "support_prediction_alignment.csv", ab["alignment"] + bc["alignment"])
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_bars(ab["main"], plot_dir / "ab_repair_harm_net.png", f"{protocol}: A versus B")
    _plot_bars(bc["main"], plot_dir / "bc_repair_harm_net.png", f"{protocol}: B versus C")
    _plot_bars(ab["signflip"], plot_dir / "ab_signflip_vs_all.png", f"{protocol}: A/B support sign-flip")
    _plot_bars(bc["signflip"], plot_dir / "bc_signflip_vs_all.png", f"{protocol}: B/C support sign-flip")
    _plot_bars(ab["quantiles"], plot_dir / "ab_support_loss_quantiles.png", f"{protocol}: A→B support-loss quintiles")
    _plot_bars(bc["quantiles"], plot_dir / "bc_support_loss_quantiles.png", f"{protocol}: B→C support-loss quintiles")
    return {"AB": ab, "BC": bc}


def _fmt(value: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:+.{digits}f}"


def _decision(analyses: dict[str, dict[str, Any]], protocols: tuple[str, ...]) -> tuple[list[dict[str, str]], str]:
    rows = []
    for comparison in ("AB", "BC"):
        states = []
        for protocol in protocols:
            result = analyses[protocol][comparison]
            all_row = result["main"][0]
            flip = result["signflip"][1]
            high = result["high_loss"][0]
            old, new = result["old_new"]
            states.append({
                "protocol": protocol,
                "net": all_row["net_gain_pp"],
                "repair": all_row["repair_pct"],
                "harm": all_row["harm_pct"],
                "macro": all_row["class_macro_delta_pp"],
                "flip_contribution": flip.get("contribution_to_all_pp", 0.0),
                "high_contribution": high.get("contribution_to_all_pp", 0.0),
                "old_net": old["net_gain_pp"],
                "new_net": new["net_gain_pp"],
            })
        strong = all(
            s["net"] > .5 and s["repair"] > s["harm"]
            and s["macro"] > 0 and s["old_net"] >= 0 and s["new_net"] >= 0
            and s["high_contribution"] > .1 for s in states
        )
        rows.append({
            "comparison": comparison,
            "dataset_consistent_prediction_gain": strong,
            "prototype_ready": "NO",
            "method_ready": "NO",
            "reason": (
                "Consistent paired gain observed, but no oracle-free EF-CIL intervention has recovered it."
                if strong else
                "Paired prediction gain is not strong and consistent across datasets, classes, old/new queries and high-loss samples."
            ),
            "protocol_evidence": "; ".join(
                f"{s['protocol']}: net={s['net']:+.2f}pp, macro={s['macro']:+.2f}pp, "
                f"old/new={s['old_net']:+.2f}/{s['new_net']:+.2f}pp, "
                f"high-loss contribution={s['high_contribution']:+.2f}pp"
                for s in states
            ),
        })
    return rows, "PROTOTYPE READY = NO; METHOD READY = NO. No oracle-free prototype or independent-seed/transition validation was performed."


def generate_report(root: Path, protocols: tuple[str, ...], repeats: int, seed: int) -> Path:
    analyses = {protocol: _protocol_analysis(root, protocol, repeats, seed) for protocol in protocols}
    cross = root / "cross_protocol"
    cross.mkdir(parents=True, exist_ok=True)
    final_rows = []
    for protocol in protocols:
        for comparison in ("AB", "BC"):
            final_rows.append({"protocol": protocol, "comparison": comparison,
                               **analyses[protocol][comparison]["main"][0]})
    write_csv(cross / "final_summary.csv", final_rows)
    decisions, verdict = _decision(analyses, protocols)
    write_csv(cross / "decision_table.csv", decisions)

    path = root / "paired_source_prediction_final_report.md"
    lines = [
        "# Paired Source Prediction Diagnostic", "",
        "## 1. 설계와 재현 검사", "",
        "- IA10/CAR10, seed 1993, final T10; 동일 test sample ID에 세 matched head를 평가했다.",
        "- h_A/h_B는 기존 R256/G256 설정으로 재현했고 기존 집계 정확도와 차이 1e-6pp 이내를 확인한 후 paired 분석을 시작했다.",
        "- h_C는 old source만 stored/APR Gaussian으로 바꾸고 256/class, new bank, pre-CA 초기 head, label 순서, batch 경계, optimizer와 step을 동일하게 했다.",
        "- Coverage support는 별도 unique-reference bank 실험의 B/C 3회 draw 평균이다. Head 학습의 256/class Gaussian draw와 동일 sample을 뜻하지 않는다.",
        "- Support-loss 상위 20% 기준은 test support-loss의 80th percentile을 정답 여부와 무관하게 계산했다.",
        "- Confidence interval은 동일 sample 단위 paired bootstrap이다. 단일 seed 결과이며 causal mediation 검증이 아니다.",
        "", "## 2. 전체 paired prediction 결과", "",
        "| Protocol | 비교 | Winner acc | Loser acc | Repair | Harm | Net gain | Disagreement | 95% CI for net | Class-macro net |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in final_rows:
        lines.append(
            f"| {row['protocol']} | {row['comparison']} | {row['accuracy_winner']:.2f}% | "
            f"{row['accuracy_loser']:.2f}% | {row['repair_pct']:.2f}% | {row['harm_pct']:.2f}% | "
            f"{row['net_gain_pp']:+.2f}pp | {row['prediction_disagreement_pct']:.2f}% | "
            f"[{row['net_gain_bootstrap_ci_low_pp']:+.2f}, {row['net_gain_bootstrap_ci_high_pp']:+.2f}]pp | "
            f"{row['class_macro_delta_pp']:+.2f}pp |"
        )
    lines.extend(["", "AB는 B wrong→A correct가 Repair, BC는 C wrong→B correct가 Repair다. 모든 행에서 `Net gain = Repair − Harm = Winner accuracy − Loser accuracy` 검사를 통과했다.", ""])
    for section_index, protocol in enumerate(protocols, start=3):
        result = analyses[protocol]
        lines.extend([f"## {section_index}. {protocol}: 집단별 분석", "",
                      "| 비교 | 집단 | N | Repair | Harm | Net | 전체 정확도 기여 | Disagreement | Class-macro |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
        for comparison in ("AB", "BC"):
            for row in (result[comparison]["main"] + result[comparison]["old_new"] +
                        [result[comparison]["signflip"][1]] + result[comparison]["high_loss"]):
                if not row.get("n"):
                    continue
                lines.append(
                    f"| {comparison} | {row['group']} | {row['n']} | {row['repair_pct']:.2f}% | "
                    f"{row['harm_pct']:.2f}% | {row['net_gain_pp']:+.2f}pp | "
                    f"{row['contribution_to_all_pp']:+.2f}pp | {row['prediction_disagreement_pct']:.2f}% | "
                    f"{row['class_macro_delta_pp']:+.2f}pp |"
                )
        lines.extend(["", "### Support loss와 예측 변화의 정렬", "",
                      "| 비교 | 예측 그룹 | N | Support loss mean | Median | q25 | q75 |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for comparison in ("AB", "BC"):
            for row in result[comparison]["alignment"]:
                lines.append(
                    f"| {comparison} | {row['prediction_group']} | {row['n']} | "
                    f"{_fmt(row['support_loss_mean'],4)} | {_fmt(row['support_loss_median'],4)} | "
                    f"{_fmt(row['support_loss_q25'],4)} | {_fmt(row['support_loss_q75'],4)} |"
                )
        lines.extend(["", "Support loss는 AB에서 `support_A−support_B`, BC에서 `support_B−support_C`다. 이 집단 비교는 상관 설명이며 인과 매개효과가 아니다.", "",
                      "### 시각화", ""])
        for filename, caption in (
            ("ab_repair_harm_net.png", "A/B Repair, Harm, Net"),
            ("bc_repair_harm_net.png", "B/C Repair, Harm, Net"),
            ("ab_signflip_vs_all.png", "A/B sign-flip vs all"),
            ("bc_signflip_vs_all.png", "B/C sign-flip vs all"),
            ("ab_support_loss_quantiles.png", "A→B support-loss quintiles"),
            ("bc_support_loss_quantiles.png", "B→C support-loss quintiles"),
        ):
            lines.extend([f"![{protocol} {caption}]({protocol}/plots/{filename})", ""])
        lines.extend([
            f"{protocol}의 class별 결과는 [`class_macro_summary.csv`]({protocol}/class_macro_summary.csv), "
            f"quintile별 정확한 값은 [`support_loss_quantile_summary.csv`]({protocol}/support_loss_quantile_summary.csv)에 있다.", "",
        ])
    decision_section = len(protocols) + 3
    lines.extend([f"## {decision_section}. 최종 판정", "",
                  "| 비교 | 공통 예측 이득 | 근거 | Prototype ready | Method ready |",
                  "|---|---|---|---|---|"])
    for row in decisions:
        lines.append(f"| {row['comparison']} | {row['dataset_consistent_prediction_gain']} | {row['protocol_evidence']} | {row['prototype_ready']} | {row['method_ready']} |")
    lines.extend(["", f"**{verdict}**", "",
                  "판정 기준은 repair/harm의 실제 순이득, class-macro, old/new, high-loss 집단의 전체 기여도와 두 dataset의 일관성이다. Geometry 차이만으로 prototype 또는 method 단계로 넘어가지 않는다.", "",
                  f"## {decision_section + 1}. 질문별 답변", ""])
    for protocol in protocols:
        ab = analyses[protocol]["AB"]
        bc = analyses[protocol]["BC"]
        ab_all, bc_all = ab["main"][0], bc["main"][0]
        ab_flip, bc_flip = ab["signflip"][1], bc["signflip"][1]
        ab_high, bc_high = ab["high_loss"][0], bc["high_loss"][0]
        ab_old, ab_new = ab["old_new"]
        bc_old, bc_new = bc["old_new"]
        ab_q1, ab_q5 = ab["quantiles"][0], ab["quantiles"][-1]
        bc_q1, bc_q5 = bc["quantiles"][0], bc["quantiles"][-1]
        lines.extend([
            f"### {protocol}", "",
            f"1. A/B prediction disagreement는 **{ab_all['prediction_disagreement_pct']:.2f}%**이고, B→A repair는 **{ab_all['repair_pct']:.2f}%** ({ab_all['repair_count']}개), harm은 **{ab_all['harm_pct']:.2f}%** ({ab_all['harm_count']}개)다.",
            f"2. A−B 순이득은 **{ab_all['net_gain_pp']:+.2f}pp**다. Repair와 harm의 절대 크기, disagreement를 보면 예측 변화가 전혀 없는 상황인지 상쇄되는 상황인지 구분된다.",
            f"3. A/B sign-flip {ab_flip['n']}개에서 순이득은 **{ab_flip.get('net_gain_pp', float('nan')):+.2f}pp**, 전체 기여는 **{ab_flip.get('contribution_to_all_pp', float('nan')):+.2f}pp**다. Support-loss Q1→Q5 순이득은 **{ab_q1.get('net_gain_pp', float('nan')):+.2f}→{ab_q5.get('net_gain_pp', float('nan')):+.2f}pp**다.",
            f"4. A/B high-loss {ab_high['n']}개의 순이득은 **{ab_high['net_gain_pp']:+.2f}pp**, 전체 기여는 **{ab_high['contribution_to_all_pp']:+.2f}pp**다. Old/New은 **{ab_old['net_gain_pp']:+.2f}/{ab_new['net_gain_pp']:+.2f}pp**, class-macro는 **{ab_all['class_macro_delta_pp']:+.2f}pp**다.",
            f"5. B/C prediction disagreement는 **{bc_all['prediction_disagreement_pct']:.2f}%**이고, C→B repair/harm은 **{bc_all['repair_pct']:.2f}/{bc_all['harm_pct']:.2f}%**다. B−C 순이득은 **{bc_all['net_gain_pp']:+.2f}pp**다.",
            f"6. B/C sign-flip {bc_flip['n']}개의 전체 기여는 **{bc_flip.get('contribution_to_all_pp', float('nan')):+.2f}pp**, high-loss {bc_high['n']}개의 기여는 **{bc_high['contribution_to_all_pp']:+.2f}pp**다. Q1→Q5 순이득은 **{bc_q1.get('net_gain_pp', float('nan')):+.2f}→{bc_q5.get('net_gain_pp', float('nan')):+.2f}pp**이며, Old/New은 **{bc_old['net_gain_pp']:+.2f}/{bc_new['net_gain_pp']:+.2f}pp**, class-macro는 **{bc_all['class_macro_delta_pp']:+.2f}pp**다.",
            "",
        ])
    lines.extend([
        "두 dataset을 함께 보면 A/B의 support-loss가 큰 sample에서 A가 더 잘 맞히는 패턴은 유지되지 않는다. B/C는 CAR10에서 support-loss와 prediction recovery가 정렬되지만 IA10에서는 전체·class-macro 이득이 작다. 따라서 source 효과와 현재 support metric의 설명력을 동일시하지 않는다.",
        "",
        f"Q10 판정: **{verdict}**", "",
        f"## {decision_section + 2}. 산출물과 한계", "",
                  "- [`audit.md`](audit.md)", "- [`cross_protocol/final_summary.csv`](cross_protocol/final_summary.csv)",
                  "- [`cross_protocol/decision_table.csv`](cross_protocol/decision_table.csv)",
                  "- 각 protocol의 `paired_source_prediction_table.csv`와 `matched_scores.npz`에 동일 sample ID 순서의 예측 및 score를 보존했다.",
                  "- A의 256/class exposure에는 actual feature replacement가 포함된다. Coverage A의 unique-reference bank와 역할이 다르다.",
                  "- B/C support 평균은 분류 head 학습의 Gaussian draw와 독립적이다. Support 정렬은 설명적이다.",
                  "- 단일 final task와 seed 1993에 한정된다. Method-ready에는 oracle 없는 개입과 독립 검증이 추가로 필요하다.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
