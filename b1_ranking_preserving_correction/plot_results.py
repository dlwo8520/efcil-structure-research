"""Read-only result audit and static figures; does not run model fitting."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ("IA10", "CAR10")
STATES = ("correct", "same_task_wrong", "cross_task_wrong")
BRANCHES = ("C0", "C1", "C2")


def load_and_audit():
    metrics, matrices, audit = {}, {}, {}
    for protocol in PROTOCOLS:
        metrics[protocol] = json.loads((ROOT / protocol / "metrics.json").read_text())
        metric = metrics[protocol]
        with (ROOT / protocol / "sample_predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with (ROOT / protocol / "transition_tables.csv").open(newline="") as stream:
            transitions = list(csv.DictReader(stream))
        assert len(rows) == metric["n"]
        assert len({row["sample_id"] for row in rows}) == len(rows)
        labels = np.array([int(row["y"]) for row in rows])
        base = np.array([int(row["pred_C0"]) for row in rows])
        b1 = np.array([row["fixed_error_type"] == "B1" for row in rows])
        assert b1.sum() == metric["fixed_B1_n"]
        assert not (base[b1] == labels[b1]).any()
        birth = np.repeat(np.arange(1, 11), metric["task_partition"])
        errors = {}
        for branch in BRANCHES:
            pred = np.array([int(row[f"pred_{branch}"]) for row in rows])
            correct = pred == labels
            repaired = (base != labels) & correct
            harmed = (base == labels) & ~correct
            summary = metric["branches"][branch]["subsets"]["overall"]
            assert repaired.sum() == summary["repair"]
            assert harmed.sum() == summary["harm"]
            assert correct.sum() == summary["correct"]
            assert (repaired & b1).sum() == metric["branches"][branch]["subsets"]["B1"]["repair"]
            assert np.isclose(100 * (repaired.sum() - harmed.sum()) / len(rows), summary["delta_pp"])
            actual_states = np.where(correct, "correct", np.where(
                birth[pred] == birth[labels], "same_task_wrong", "cross_task_wrong"))
            saved_states = np.array([row[f"error_state_{branch}"] for row in rows])
            assert np.array_equal(actual_states, saved_states)
            same_harm = int((harmed & (birth[pred] == birth[labels])).sum())
            assert same_harm == 0
            matrix = np.array([[sum(row["error_state_C0"] == source and
                                      row[f"error_state_{branch}"] == target for row in rows)
                                for target in STATES] for source in STATES])
            saved_matrix = np.zeros((3, 3), dtype=int)
            for row in transitions:
                if row["branch"] == branch:
                    saved_matrix[STATES.index(row["from_C0"]), STATES.index(row["to_branch"])] = int(row["n"])
            assert np.array_equal(matrix, saved_matrix)
            errors[branch] = {"repair": int(repaired.sum()), "harm": int(harmed.sum()),
                              "correct_to_same_task_wrong": same_harm,
                              "transition_sum": int(matrix.sum())}
            if branch == "C2":
                matrices[protocol] = matrix
        audit[protocol] = {"sample_rows": len(rows), "fixed_B1_n": int(b1.sum()),
                           "sample_vs_metrics_and_transition_tables": "PASS", "branches": errors}
    return metrics, matrices, audit


def make_plots(metrics, matrices):
    output = ROOT / "plots"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        x = np.arange(2)
        summary = [metrics[protocol]["branches"][b]["subsets"]["overall"] for b in ("C1", "C2")]
        for shift, field, label, color in ((-0.18, "repair", "Repair", "#277a59"),
                                            (0.18, "harm", "Harm", "#ba4545")):
            bars = ax.bar(x + shift, [r[field] for r in summary], width=0.36, label=label, color=color)
            ax.bar_label(bars, padding=3)
        ax.set_xticks(x, [f"{b}\nNet {r['net']:+d}, {r['delta_pp']:+.3f} pp"
                          for b, r in zip(("C1", "C2"), summary)])
        ax.set(title=protocol, ylabel="Real-test samples (vs C0)")
        ax.margins(y=0.15)
        ax.legend()
    fig.savefig(output / "01_paired_repair_harm.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = ("#777777", "#719ac8", "#df8a32")
    for col, protocol in enumerate(PROTOCOLS):
        metric = metrics[protocol]
        values = [metric["gaussian_CE"][b]["check_Gaussian_CE"] for b in BRANCHES]
        bars = axes[0, col].bar(BRANCHES, values, color=colors)
        axes[0, col].bar_label(bars, fmt="%.4f", padding=3)
        axes[0, col].set(title=f"{protocol}: independent Gaussian check", ylabel="CE (lower is better)")
        axes[0, col].margins(y=0.2)
        base = metric["branches"]["C0"]
        x = np.arange(2)
        micro = [metric["branches"][b]["subsets"]["overall"]["delta_pp"] for b in ("C1", "C2")]
        macro = [100 * (metric["branches"][b]["macro_accuracy"] - base["macro_accuracy"])
                 for b in ("C1", "C2")]
        for shift, vals, label, color in ((-0.18, micro, "Micro", "#5c88b3"),
                                          (0.18, macro, "Class macro", "#b88954")):
            bars = axes[1, col].bar(x + shift, vals, width=0.36, label=label, color=color)
            axes[1, col].bar_label(bars, fmt="%+.3f", padding=3)
        axes[1, col].axhline(0, color="black", linewidth=0.8)
        axes[1, col].set(xticks=x, xticklabels=("C1", "C2"),
                         title=f"{protocol}: real test", ylabel="Accuracy change vs C0 (pp)", ylim=(-0.7, 0.3))
        axes[1, col].legend()
    fig.savefig(output / "02_gaussian_vs_real.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, protocol in zip(axes, PROTOCOLS):
        matrix = matrices[protocol]
        ax.imshow(np.log1p(matrix), cmap="Blues")
        for (i, j), n in np.ndenumerate(matrix):
            ax.text(j, i, str(n), ha="center", va="center",
                    color="white" if np.log1p(n) > np.log1p(matrix.max()) * 0.6 else "black")
        ax.set(xticks=range(3), yticks=range(3),
               xticklabels=("Correct", "Same wrong", "Cross wrong"),
               yticklabels=("Correct", "Same wrong", "Cross wrong"),
               xlabel="C2 state", ylabel="C0 state", title=protocol)
    fig.suptitle("C0 to C2 sample transitions (counts; color uses log scale)")
    fig.savefig(output / "03_c0_c2_transitions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    metrics, matrices, audit = load_and_audit()
    make_plots(metrics, matrices)
    (ROOT / "report_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Sample-level audit PASS; 3 result figures generated.")
