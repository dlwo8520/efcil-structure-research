"""Validate paired controls, aggregate T9/T10 outcomes, and draw pilot plots."""
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
BRANCHES = ("control", "js")


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _cohort(rows: list[dict], start: int, stop: int) -> float:
    selected = [row for row in rows if start <= int(row["mapped_class_id"]) < stop]
    if not selected:
        raise RuntimeError(f"empty cohort {start}:{stop}")
    count = sum(int(row["num_samples"]) for row in selected)
    correct = sum(int(row["num_correct"]) for row in selected)
    return 100 * correct / count


def _task_rows(protocol: str, branch: str) -> tuple[dict, dict, list[dict], list[dict]]:
    run = ROOT / protocol / branch / "seed_1993"
    summary = _read_csv(run / "task_summary.csv")
    classes = _read_csv(run / "class_accuracy.csv")
    t9 = next(row for row in summary if int(row["task_id"]) == 9)
    t10 = next(row for row in summary if int(row["task_id"]) == 10)
    c9 = [row for row in classes if int(row["trained_task"]) == 9]
    c10 = [row for row in classes if int(row["trained_task"]) == 10]
    return t9, t10, c9, c10


def _plot_bars(path: Path, labels: list[str], control: list[float], js: list[float], title: str):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    ax.bar(x - .18, control, .36, label="Two-view control")
    ax.bar(x + .18, js, .36, label="Two-view + JS")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def phase_a_summary() -> dict:
    folder = ROOT / "phaseA_ceos_error_destination"
    data = {p: _read_json(folder / f"{p}_summary.json") for p in PROTOCOLS}
    lines = ["# Phase A — CEOS New-error destination", "",
             "저장되지 않았던 C1 logits는 사용자가 승인한 동일 설정 결정론적 재실행으로 복원했다. 기존 C1 accuracy·train loss와 정확히 일치함을 확인한 뒤 paired New test sample에만 적용했다.", "",
             "| Protocol | N New | Gaussian correct→CEOS wrong | Gaussian wrong→CEOS correct | Type 1 Old competition | Type 2 New internal | Type 3 Both |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for p in PROTOCOLS:
        row = data[p]
        types = row["type_counts"]["loss"]
        lines.append(f"| {p} | {row['N_new_test']} | {row['gaussian_correct_ceos_wrong_count']} ({row['gaussian_correct_ceos_wrong_rate']*100:.2f}%) | {row['gaussian_wrong_ceos_correct_count']} ({row['gaussian_wrong_ceos_correct_rate']*100:.2f}%) | {types['type1_old_competition']} | {types['type2_new_internal']} | {types['type3_both']} |")
    lines += ["", "손실 사례는 모두 true New class가 New 내부에서는 1위지만 Old class에 밀린 Type 1이었다. Recovery는 없었다. 즉 CEOS의 New 손실은 이 실행에서 Old/New 경쟁 악화로 국소화되며 New-New 단독 판별 악화 근거는 없다.", "",
              "`delta_g`/`delta_w`의 전체·손실·회복 부분집합별 mean/median/q10/q25/q75/q90은 각 protocol의 `*_summary.json`에, sample별 값은 CSV에 있다.", ""]
    (folder / "summary.md").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["IA10", "CAR10"]
    bottoms = np.zeros(2)
    for key, label in (("type1_old_competition", "Old competition"),
                       ("type2_new_internal", "New internal"), ("type3_both", "Both")):
        values = np.array([data[p]["type_counts"]["loss"][key] for p in PROTOCOLS])
        ax.bar(labels, values, bottom=bottoms, label=label)
        bottoms += values
    ax.set_ylabel("New errors introduced by CEOS (count)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "new_error_destination_stacked.png", dpi=150)
    plt.close(fig)
    return data


def analyze() -> dict:
    phase_a = phase_a_summary()
    output = ROOT / "cross_protocol"
    output.mkdir(parents=True, exist_ok=True)
    summaries, deltas, decisions = [], [], []
    controls = {}
    for protocol in PROTOCOLS:
        manifests = {}
        metrics = {}
        for branch in BRANCHES:
            run = ROOT / protocol / branch / "seed_1993"
            if not (run / "task_checkpoints" / "task_10.pt").is_file():
                raise RuntimeError(f"pilot incomplete: {run}")
            augmentation_config = {
                "source": "structure_multicenter_separation/data/data.py::build_transform(True,None)",
                "views_per_image": 2,
                "per_view": [
                    "RandomResizedCrop(224, scale=(0.05,1.0), ratio=(0.75,1.3333333333333333))",
                    "RandomHorizontalFlip(p=0.5)", "ToTensor()",
                ],
                "new_augmentation_added": False,
                "held_out_test_transform": "Resize(256, interpolation=3), CenterCrop(224), ToTensor()",
            }
            (run / "augmentation_config.json").write_text(json.dumps(augmentation_config, indent=2) + "\n")
            start = _read_json(run / "pilot_start_manifest.json")
            t9_manifest = _read_json(run / "pilot_task_09_manifest.json")
            t10_manifest = _read_json(run / "pilot_task_10_manifest.json")
            ca_rows = _read_csv(run / "epoch_metrics.csv")
            ca = {}
            for task_id in (9, 10):
                selected = [row for row in ca_rows if row["stage"] == "classifier_retraining" and int(row["task_id"]) == task_id]
                if len(selected) != int(start["ca_epochs"]):
                    raise AssertionError(f"{protocol} {branch} T{task_id}: CA epoch count mismatch")
                expected_classes = 180 if protocol == "IA10" and task_id == 9 else 200 if protocol == "IA10" else 177 if task_id == 9 else 196
                if any(int(row["optimizer_steps"]) != expected_classes or
                       int(row["processed_samples"]) != expected_classes * int(start["ca_exposure_per_class"])
                       for row in selected):
                    raise AssertionError(f"{protocol} {branch} T{task_id}: CA step/exposure mismatch")
                ca[task_id] = {"epochs": len(selected), "optimizer_steps": sum(int(row["optimizer_steps"]) for row in selected),
                               "processed_samples": sum(int(row["processed_samples"]) for row in selected)}
            manifests[branch] = (start, t9_manifest, t10_manifest, ca)
            t9, t10, c9, c10 = _task_rows(protocol, branch)
            total = len(c10)
            # Canonical partitions are 20x10 (IA) and 20x6+19x4 (CAR).
            birth9 = 160 if protocol == "IA10" else 158
            birth10 = 180 if protocol == "IA10" else 177
            if len(c9) != birth10 or total != (200 if protocol == "IA10" else 196):
                raise AssertionError("class partition mismatch")
            row = {
                "protocol": protocol, "branch": branch,
                "t9_seen": float(t9["accuracy"]),
                "t9_prior_old": _cohort(c9, 0, birth9),
                "t9_new": _cohort(c9, birth9, birth10),
                "t9_macro": float(np.mean([float(x["accuracy"]) for x in c9])),
                "t9_old_to_new": float(t9["old_to_new_error_rate"]),
                "t9_new_to_old": float(t9["new_to_old_error_rate"]),
                "t10_seen": float(t10["accuracy"]),
                "t10_prior_old": _cohort(c10, 0, birth9),
                "t10_newly_old_t9": _cohort(c10, birth9, birth10),
                "t10_new": _cohort(c10, birth10, total),
                "t10_macro": float(np.mean([float(x["accuracy"]) for x in c10])),
            }
            row["retention_t9"] = row["t10_newly_old_t9"] - row["t9_new"]
            metrics[branch] = row
            summaries.append(row)
            _write_csv(ROOT / protocol / "after_T9.csv", [
                {key: metrics[b][key] for key in ("protocol", "branch", "t9_seen", "t9_prior_old", "t9_new", "t9_macro", "t9_old_to_new", "t9_new_to_old")}
                for b in metrics])
        for offset in (1, 2):
            a, b = manifests["control"][offset], manifests["js"][offset]
            strict = ("view_sha256", "train_samples_per_epoch", "epochs", "optimizer_steps",
                      "stats_update_count", "apr_correction_count", "ca_call_count",
                      "ca_epochs", "ca_exposure_per_class", "num_views_per_image")
            for key in strict:
                if a[key] != b[key]:
                    raise AssertionError(f"{protocol} T{a['task']} unmatched {key}: {a[key]} != {b[key]}")
        if manifests["control"][3] != manifests["js"][3]:
            raise AssertionError(f"{protocol}: unmatched CA optimizer steps or exposure")
        for key in ("source_checkpoint_sha256", "seed", "optimizer", "init_lr", "inc_epochs",
                    "batch_size", "ca_epochs", "ca_exposure_per_class", "scc", "apr", "ca"):
            if manifests["control"][0][key] != manifests["js"][0][key]:
                raise AssertionError(f"{protocol}: unmatched start configuration {key}")
        if metrics["control"].keys() != metrics["js"].keys():
            raise AssertionError("metric keys differ")
        delta = {"protocol": protocol, **{key: metrics["js"][key] - metrics["control"][key]
                                       for key in metrics["control"] if key not in ("protocol", "branch")}}
        deltas.append(delta)
        c9, j9 = metrics["control"], metrics["js"]
        promising = (delta["t9_new"] > 0
                     and delta["t10_newly_old_t9"] > 0
                     and delta["t10_newly_old_t9"] >= .25 * delta["t9_new"]
                     and delta["t10_prior_old"] >= -delta["t10_newly_old_t9"]
                     and delta["t10_new"] > -2 and delta["t10_seen"] > 0)
        decisions.append({"protocol": protocol, "promising": promising,
                          "reason": "all primary directions positive without larger prior-old/new cost" if promising
                          else "primary gain, retention, or overall cost condition failed"})
        controls[protocol] = {"control": c9, "js": j9, "delta": delta,
                              "view_sha256_t9": manifests["control"][1]["view_sha256"],
                              "view_sha256_t10": manifests["control"][2]["view_sha256"],
                              "ca_exposure_and_steps": manifests["control"][3]}
        _write_csv(ROOT / protocol / "after_T10.csv", [
            {key: metrics[b][key] for key in ("protocol", "branch", "t10_seen", "t10_prior_old", "t10_newly_old_t9", "t10_new", "t10_macro")}
            for b in BRANCHES])
        _write_csv(ROOT / protocol / "retention.csv", [
            {key: metrics[b][key] for key in ("protocol", "branch", "t9_new", "t10_newly_old_t9", "retention_t9")}
            for b in BRANCHES])
        plots = ROOT / protocol / "plots"
        _plot_bars(plots / "t9_new_control_vs_js.png", ["T9 New"],
                   [c9["t9_new"]], [j9["t9_new"]], f"{protocol}: T9 held-out New")
        _plot_bars(plots / "t9_retention.png", ["After T9", "After T10"],
                   [c9["t9_new"], c9["t10_newly_old_t9"]],
                   [j9["t9_new"], j9["t10_newly_old_t9"]], f"{protocol}: T9 cohort")
        categories = ["Seen", "Prior Old", "T9 Newly Old", "T10 New"]
        keys = ("t10_seen", "t10_prior_old", "t10_newly_old_t9", "t10_new")
        _plot_bars(plots / "t10_cohorts.png", categories,
                   [c9[k] for k in keys], [j9[k] for k in keys], f"{protocol}: after T10")
        c10_classes = {int(row["mapped_class_id"]): float(row["accuracy"]) for row in _task_rows(protocol, "control")[3]}
        j10_classes = {int(row["mapped_class_id"]): float(row["accuracy"]) for row in _task_rows(protocol, "js")[3]}
        cohort_ids = list(range(birth9, birth10))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([str(x) for x in cohort_ids], [j10_classes[x] - c10_classes[x] for x in cohort_ids])
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("JS − Control accuracy (pp)")
        ax.set_title(f"{protocol}: T9 class delta after T10")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        plots.mkdir(parents=True, exist_ok=True)
        fig.savefig(plots / "t9_per_class_delta.png", dpi=150)
        plt.close(fig)
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "delta_table.csv", deltas)
    _write_csv(output / "decision_table.csv", decisions)
    result = {"phase_a": phase_a, "phase_b": controls, "decisions": decisions,
              "boundary_axis_closed": True,
              "consistency_prototype_promising": all(x["promising"] for x in decisions),
              "full_method_ready": False}
    (ROOT / "run_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


if __name__ == "__main__":
    analyze()
