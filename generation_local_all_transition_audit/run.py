#!/usr/bin/env python3
"""Read-only prediction analysis: no torch, model inference, training or generation."""
from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[1])))

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ("IA10", "CAR10")
STATES = ("W", "C", "K")
COHORTS = ("T9_all", "T9_A_within", "T9_A_cross", "T9_non_A")
EXPECTED_N = {"IA10": 147, "CAR10": 758}
EXPECTED_SUBTYPES = {"IA10": (27, 34), "CAR10": (42, 77)}
EXPECTED_DIRECT_ALL_CORRECT = {
    "IA10": {9: {"S": 84, "G": 84}, 10: {"S": 84, "G": 84}},
    "CAR10": {9: {"S": 621, "G": 617}, 10: {"S": 614, "G": 608}},
}


def require(condition, message):
    """Unlike bare assert, validation stays enabled under python -O."""
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), f"Empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state(label, local_pred, all_pred):
    """K implies local correct: stored argmax must use consistent tie policy."""
    label, local_pred, all_pred = int(label), int(local_pred), int(all_pred)
    require(all_pred != label or local_pred == label,
            "All-seen correct but local wrong: inconsistent candidate set/prediction")
    if local_pred != label:
        return "W"
    return "K" if all_pred == label else "C"


def birth_for_label(label, partition):
    upper = 0
    for task, size in enumerate(partition, 1):
        upper += size
        if 0 <= label < upper:
            return task
    raise ValueError(f"Label {label} outside partition")


def unique_index(rows, source):
    result = {row["sample_id"]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate sample ID in {source}")
    return result


def trajectory_type(row):
    kind = row["error_type"]
    if kind.startswith("A_"):
        return "A"
    if kind.startswith("C_"):
        return "C"
    if kind.startswith("B_"):
        return "B1" if row["birth_restriction_restored"] == "1" else "B2"
    require(kind == "final_correct", f"Unknown fixed type {kind}")
    return kind


def frozen_subtypes(trajectory, audit_rows):
    """Only existing vanilla birth audit defines A-within/A-cross; no S/G use."""
    locations = {}
    repeated = defaultdict(set)
    for row in audit_rows:
        if row["group"] != "A":
            continue
        sid = row["sample_id"]
        require(sid in trajectory, "Vanilla subtype sample missing from trajectory")
        original = trajectory[sid]
        require(trajectory_type(original) == "A", "Subtype is not vanilla persistent A")
        label, local, all_pred = int(row["y"]), int(row["local_pred"]), int(row["all_pred"])
        require(label == int(original["true_label"]) and int(row["task"]) == int(original["birth_task"]),
                "Vanilla subtype label/birth mismatch")
        require(all_pred == int(original["pred_Tbirth"]) and all_pred != label,
                "Vanilla A birth prediction mismatch")
        location = "A-cross" if local == label else "A-within"
        require(row["A_location"] == location, "Existing subtype disagrees with its vanilla predictions")
        repeated[sid].add((location, label, local, all_pred))
        require(len(repeated[sid]) == 1, "Subtype differs across support subsampling seeds")
        locations[sid] = location
    for sid, row in trajectory.items():
        if trajectory_type(row) == "A" and int(row["birth_task"]) <= 9:
            require(sid in locations, "Existing vanilla subtype audit does not cover fixed A")
    return {sid: locations[sid] if trajectory_type(row) == "A" and int(row["birth_task"]) <= 9
            else trajectory_type(row) for sid, row in trajectory.items()}


def cohort_name(subtype):
    if subtype == "A-within":
        return "T9_A_within"
    if subtype == "A-cross":
        return "T9_A_cross"
    return "T9_non_A"


def summarize(pairs, protocol, task, cohort, comparison):
    """pairs: sample_id, y, reference_state, target_state. Percentages are 0..100."""
    counts = Counter((row["reference_state"], row["target_state"]) for row in pairs)
    require(all(a in STATES and b in STATES for a, b in counts), "Invalid W/C/K state")
    n = len(pairs)
    require(sum(counts.values()) == n, "Transition sum mismatch")
    local_repair = counts["W", "C"] + counts["W", "K"]
    local_harm = counts["C", "W"] + counts["K", "W"]
    all_repair = counts["W", "K"] + counts["C", "K"]
    all_harm = counts["K", "W"] + counts["K", "C"]
    ref_local = sum(row["reference_state"] != "W" for row in pairs)
    target_local = sum(row["target_state"] != "W" for row in pairs)
    ref_all = sum(row["reference_state"] == "K" for row in pairs)
    target_all = sum(row["target_state"] == "K" for row in pairs)
    require(local_repair - local_harm == target_local - ref_local, "Local net identity mismatch")
    require(all_repair - all_harm == target_all - ref_all, "All-seen net identity mismatch")
    by_class = defaultdict(list)
    for row in pairs:
        by_class[int(row["y"])].append(row)
    percent = lambda count: 100 * count / n if n else None
    macro = lambda branch, local: sum(
        sum((row[branch] != "W") if local else (row[branch] == "K") for row in xs) / len(xs)
        for xs in by_class.values()) * 100 / len(by_class) if by_class else None
    reference, target = comparison.split("-")[1], comparison.split("-")[0]
    result = {"protocol": protocol, "time": f"T{task}", "cohort": cohort, "comparison": comparison,
              "reference_branch": reference, "target_branch": target, "N": n,
              "reference_local_accuracy_pct": percent(ref_local), "target_local_accuracy_pct": percent(target_local),
              "local_repair": local_repair, "local_harm": local_harm, "local_net": local_repair-local_harm,
              "local_delta_pp": percent(local_repair-local_harm),
              "reference_all_accuracy_pct": percent(ref_all), "target_all_accuracy_pct": percent(target_all),
              "all_repair": all_repair, "all_harm": all_harm, "all_net": all_repair-all_harm,
              "all_delta_pp": percent(all_repair-all_harm), "class_count": len(by_class),
              "reference_macro_local_accuracy_pct": macro("reference_state", True),
              "target_macro_local_accuracy_pct": macro("target_state", True),
              "reference_macro_all_accuracy_pct": macro("reference_state", False),
              "target_macro_all_accuracy_pct": macro("target_state", False)}
    for a in STATES:
        for b in STATES:
            result[f"n{a}{b}"] = counts[a, b]
    return result


def binary_summary(pairs, protocol, task, scope, comparison):
    n = len(pairs)
    repair = sum(not row["reference_correct"] and row["target_correct"] for row in pairs)
    harm = sum(row["reference_correct"] and not row["target_correct"] for row in pairs)
    ref = sum(row["reference_correct"] for row in pairs)
    target = sum(row["target_correct"] for row in pairs)
    require(target-ref == repair-harm, "Collateral paired identity mismatch")
    return {"protocol": protocol, "time": f"T{task}", "scope": scope, "comparison": comparison,
            "N": n, "reference_all_accuracy_pct": 100*ref/n if n else None,
            "target_all_accuracy_pct": 100*target/n if n else None,
            "repair": repair, "harm": harm, "net": repair-harm,
            "delta_pp": 100*(repair-harm)/n if n else None}


def load_protocol(source, trajectory_root, protocol, register):
    metrics = read_json(register(source / "metrics.json"))[protocol]
    partition = read_json(register(source / f"training/{protocol}/T9_T10/S/resolved_config.json"))["task_partition"]
    require(len(partition) == 10, "Expected ten-task protocol")
    trajectory = unique_index(read_csv(register(trajectory_root / protocol / "sample_trajectory.csv")), "trajectory")
    subtype = frozen_subtypes(trajectory, read_csv(register(source / f"audit/{protocol}/birth_support_audit.csv")))
    predictions = {}
    for location, tasks in (("T1", (1,)), ("T9_T10", (9, 10))):
        for branch in ("S", "G", "R"):
            config = read_json(register(source / f"training/{protocol}/{location}/{branch}/resolved_config.json"))
            require(config["seed"] == 1993 and config["protocol_id"] == protocol,
                    "Expected IA10/CAR10 seed 1993")
            require(config["task_partition"] == partition, "Branch task partition mismatch")
        for task in tasks:
            for branch in ("S", "G", "R"):
                path = register(source / f"training/{protocol}/{location}/{branch}/predictions_T{task:02d}.csv")
                rows = unique_index(read_csv(path), path)
                for sid, row in rows.items():
                    require(sid in trajectory, "Prediction not present in vanilla trajectory")
                    vanilla = trajectory[sid]
                    label, birth = int(row["y"]), int(row["birth_task"])
                    require(label == int(vanilla["true_label"]) and birth == int(vanilla["birth_task"]),
                            "Paired class mapping/birth mismatch")
                    require(birth_for_label(label, partition) == birth, "Class label inconsistent with birth partition")
                    require(row["branch"] == branch and int(row["evaluated_after_task"]) == task,
                            "Wrong branch/evaluation time in predictions")
                    require(row["fixed_type"] == trajectory_type(vanilla), "Fixed trajectory subset changed")
                    local, all_pred = int(row["local_pred"]), int(row["pred"])
                    lower, upper = sum(partition[:birth-1]), sum(partition[:birth])
                    require(lower <= local < upper, "Local candidate is not ORIGINAL BIRTH TASK (T9 at T10)")
                    require(0 <= all_pred < sum(partition[:task]), "All-seen prediction outside seen candidate set")
                    state(label, local, all_pred)
                predictions[task, branch] = rows
            require(set(predictions[task, "S"]) == set(predictions[task, "G"]) == set(predictions[task, "R"]),
                    "Branches evaluated different sample identities")
            for branch in ("S", "G", "R"):
                rows = list(predictions[task, branch].values())
                stored = metrics["conditions"][f"{location}_T{task}_{branch}"]
                require(len(rows) == stored["n"], "Whole-seen N differs from previous report")
                require(math.isclose(sum(int(r["pred"])==int(r["y"]) for r in rows)/len(rows), stored["accuracy"], abs_tol=1e-12),
                        "Whole-seen accuracy differs from previous report")
    return partition, trajectory, subtype, predictions, metrics


def analyze_protocol(source, trajectory_root, output, protocol, register):
    partition, trajectory, subtype, predictions, metrics = load_protocol(source, trajectory_root, protocol, register)
    ids = sorted(s for s, row in predictions[9, "S"].items() if int(row["birth_task"]) == 9)
    require(len(ids) == EXPECTED_N[protocol], "A1: T9 direct cohort count mismatch")
    require(set(ids) == {s for s, row in predictions[10, "S"].items() if int(row["birth_task"]) == 9},
            "T9 cohort identities changed at T10")
    expected_within, expected_cross = EXPECTED_SUBTYPES[protocol]
    require(sum(subtype[s] == "A-within" for s in ids) == expected_within, "Fixed T9 A-within N mismatch")
    require(sum(subtype[s] == "A-cross" for s in ids) == expected_cross, "Fixed T9 A-cross N mismatch")
    wide = []
    for sid in ids:
        row = {"protocol": protocol, "sample_id": sid, "true_label": int(trajectory[sid]["true_label"]),
               "birth_task": 9, "vanilla_subtype": subtype[sid],
               "vanilla_birth_all_pred": int(trajectory[sid]["pred_Tbirth"]),
               "vanilla_birth_all_correct": int(trajectory[sid]["correct_Tbirth"])}
        for task in (9, 10):
            for branch in ("S", "G", "R"):
                pred = predictions[task, branch][sid]
                row.update({f"{branch}_T{task}_local_pred": int(pred["local_pred"]),
                            f"{branch}_T{task}_all_pred": int(pred["pred"]),
                            f"{branch}_T{task}_state": state(pred["y"], pred["local_pred"], pred["pred"]),
                            f"{branch}_T{task}_margin_all": float(pred["margin_all"]),
                            f"{branch}_T{task}_margin_within": float(pred["margin_within"])})
            row[f"transition_T{task}"] = f"{row[f'S_T{task}_state']}->{row[f'G_T{task}_state']}"
            row[f"transition_G_R_T{task}"] = f"{row[f'R_T{task}_state']}->{row[f'G_T{task}_state']}"
        wide.append(row)
    target = output / protocol
    write_csv(target / "t9_sample_state_transitions.csv", wide)
    summaries, class_rows, collateral, collateral_samples, t1 = [], [], [], [], []
    for comparison, reference in (("G-S", "S"), ("G-R", "R")):
        for task in (9, 10):
            pairs = [{"sample_id": row["sample_id"], "y": row["true_label"],
                      "cohort": cohort_name(row["vanilla_subtype"]),
                      "reference_state": row[f"{reference}_T{task}_state"], "target_state": row[f"G_T{task}_state"]}
                     for row in wide]
            for cohort in COHORTS:
                own = pairs if cohort == "T9_all" else [r for r in pairs if r["cohort"] == cohort]
                summary = summarize(own, protocol, task, cohort, comparison)
                summaries.append(summary)
                if cohort == "T9_all":
                    previous = metrics["comparisons"][f"T9_T10_T{task}_{comparison}"]["T9"]
                    require((summary["N"], summary["all_repair"], summary["all_harm"]) ==
                            (previous["n"], previous["repair"], previous["harm"]), "All net differs from original paired report")
                    for branch, field in ((reference, "reference"), ("G", "target")):
                        previous_cond = metrics["conditions"][f"T9_T10_T{task}_{branch}"]
                        accuracy = previous_cond["new_accuracy"] if task == 9 else previous_cond["T9_cohort_accuracy"]
                        require(math.isclose(summary[f"{field}_all_accuracy_pct"], 100*accuracy, abs_tol=1e-10),
                                "A2/A3: cohort accuracy differs from original report")
                        if branch in ("S", "G"):
                            correct = sum(predictions[task, branch][sid]["pred"] == predictions[task, branch][sid]["y"] for sid in ids)
                            require(correct == EXPECTED_DIRECT_ALL_CORRECT[protocol][task][branch],
                                    "A2/A3: fixed experiment's direct raw correct count changed")
                        if task == 9:
                            require(math.isclose(summary[f"{field}_local_accuracy_pct"], 100*previous_cond["new_local_accuracy"], abs_tol=1e-10),
                                    "T9 local accuracy differs from original report")
                for label in sorted({r["y"] for r in own}):
                    class_summary = summarize([r for r in own if r["y"] == label], protocol, task, cohort, comparison)
                    class_rows.append({"true_label": label, **class_summary})
            for scope in ("historical_birth_le_T8", "T9_direct", "T10_new", "whole_seen"):
                raw = []
                for sid, pred in predictions[task, reference].items():
                    birth = int(pred["birth_task"])
                    included = scope == "whole_seen" or (scope == "historical_birth_le_T8" and birth <= 8) or (
                        scope == "T9_direct" and birth == 9) or (scope == "T10_new" and birth == 10)
                    if included:
                        gen = predictions[task, "G"][sid]
                        ref_good = int(pred["pred"]) == int(pred["y"])
                        target_good = int(gen["pred"]) == int(gen["y"])
                        raw.append({"reference_correct": ref_good, "target_correct": target_good})
                        if scope == "historical_birth_le_T8":
                            collateral_samples.append({"protocol": protocol, "time": f"T{task}", "comparison": comparison,
                                "sample_id": sid, "true_label": int(pred["y"]), "birth_task": birth,
                                "vanilla_subtype": subtype[sid], "reference_all_pred": int(pred["pred"]),
                                "G_all_pred": int(gen["pred"]), "repair": int(not ref_good and target_good),
                                "harm": int(ref_good and not target_good)})
                collateral.append(binary_summary(raw, protocol, task, scope, comparison))
            scopes = collateral[-4:]
            require(sum(row["N"] for row in scopes[:3]) == scopes[3]["N"], "Whole-seen scope denominator identity failed")
            for field in ("repair", "harm", "net"):
                require(sum(row[field] for row in scopes[:3]) == scopes[3][field], "Whole-seen scope paired count identity failed")
            row = scopes[3]
            prev = metrics["comparisons"][f"T9_T10_T{task}_{comparison}"]["overall"]
            require((row["N"], row["repair"], row["harm"]) == (prev["n"], prev["repair"], prev["harm"]),
                    "Whole-seen paired comparison differs from original report")
        initial = predictions[1, reference]
        pairs = [{"sample_id": sid, "y": int(row["y"]),
                  "reference_state": state(row["y"], row["local_pred"], row["pred"]),
                  "target_state": state(predictions[1, "G"][sid]["y"], predictions[1, "G"][sid]["local_pred"],
                                        predictions[1, "G"][sid]["pred"])} for sid, row in initial.items()]
        require(all(r["reference_state"] != "C" and r["target_state"] != "C" for r in pairs), "T1 local must equal all")
        result = summarize(pairs, protocol, 1, "T1_separate_acquisition", comparison)
        previous = metrics["comparisons"][f"T1_T1_{comparison}"]["overall"]
        require((result["N"], result["all_repair"], result["all_harm"]) == (previous["n"], previous["repair"], previous["harm"]),
                "T1 comparison differs from original report")
        t1.append(result)
    for task in (9, 10):
        rows = [row for row in summaries if row["time"] == f"T{task}"]
        write_csv(target / f"t{task}_transition_summary.csv", rows)
    write_csv(target / "per_class_summary.csv", class_rows)
    write_csv(target / "collateral_summary.csv", collateral)
    write_csv(target / "historical_sample_pairs.csv", collateral_samples)
    write_csv(target / "t1_summary.csv", t1)
    write_json(target / "transition_matrices.json", [{"protocol": r["protocol"], "time": r["time"],
        "cohort": r["cohort"], "comparison": r["comparison"], "state_order": list(STATES),
        "rows_reference_columns_G": [[r[f"n{a}{b}"] for b in STATES] for a in STATES], "N": r["N"]} for r in summaries])
    return summaries, collateral, t1


def decisions(summaries):
    rows = []
    for protocol in PROTOCOLS:
        own = {row["time"]: row for row in summaries if row["protocol"] == protocol and
               row["comparison"] == "G-S" and row["cohort"] == "T9_all"}
        birth, later = own["T9"], own["T10"]
        rows.append({"protocol": protocol, "T9_local_net": birth["local_net"], "T9_local_delta_pp": birth["local_delta_pp"],
                     "T10_local_net": later["local_net"], "T10_local_delta_pp": later["local_delta_pp"],
                     "T9_all_net": birth["all_net"], "T10_all_net": later["all_net"],
                     "LOCAL_SIGNAL": "YES" if birth["local_net"] > 0 else "NO",
                     "STRICT_POSITIVE_GAIN_AT_T10": "YES" if later["local_net"] > 0 else "NO",
                     "SPEC_NONNEGATIVE_T10_CONDITION": "YES" if birth["local_net"] > 0 and later["local_net"] >= 0 else "NO",
                     "GENERATION_DIRECTION": "STOP", "NEXT_METHOD": "REVIEW_REAL_ONLY_CE_SUPCON_MATCHED_PILOT"})
    common = all(row["T9_local_net"] > 0 for row in rows)
    retained = common and all(row["T10_local_net"] >= 0 for row in rows)
    all_translation = all(row["T9_all_net"] > 0 and row["T10_all_net"] >= 0 for row in rows)
    return rows, {"IA10_LOCAL_SIGNAL": rows[0]["LOCAL_SIGNAL"], "CAR10_LOCAL_SIGNAL": rows[1]["LOCAL_SIGNAL"],
                  "LOCAL_SIGNAL_PERSISTS_TO_T10": "YES" if retained else "NO",
                  "DATASET_COMMON_EVIDENCE": "YES" if common else "NO",
                  "ALL_SEEN_TRANSLATION": "YES" if all_translation else "NO",
                  "GENERATION_DIRECTION": "STOP", "NEXT_METHOD": "REVIEW_REAL_ONLY_CE_SUPCON_MATCHED_PILOT",
                  "persistence_definition": "Both datasets: T9 local net > 0 AND T10 local net >= 0. IA10 zero at T10 satisfies only its nonnegative bound, not a strictly positive advantage.",
                  "positive_net_is_statistical_significance": False, "new_method_authorized_or_run": False}


def draw_plots(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [r for r in summaries if r["comparison"] == "G-S"]
    for protocol in PROTOCOLS:
        folder = output / protocol / "plots"
        folder.mkdir(parents=True, exist_ok=True)
        for task in (9, 10):
            own = [r for r in primary if r["protocol"] == protocol and r["time"] == f"T{task}"]
            for row in own:
                matrix = [[row[f"n{a}{b}"] for b in STATES] for a in STATES]
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                im = ax.imshow(matrix, cmap="Blues", vmin=0)
                for i in range(3):
                    for j in range(3):
                        ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=13,
                                color="white" if matrix[i][j] > max(max(x) for x in matrix)/2 else "black")
                ax.set(xticks=range(3), yticks=range(3), xticklabels=STATES, yticklabels=STATES,
                       xlabel="G state (columns)", ylabel="S state (rows)",
                       title=f"{protocol} T{task}: {row['cohort']} (N={row['N']})")
                fig.colorbar(im, ax=ax, label="Raw sample count")
                fig.tight_layout()
                fig.savefig(folder / f"t{task}_{row['cohort']}_heatmap.png", dpi=170)
                plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, task in zip(axes, (9, 10)):
            rows = [r for r in primary if r["protocol"] == protocol and r["time"] == f"T{task}" and r["cohort"] != "T9_all"]
            x = list(range(len(rows)))
            ax.bar([v-.18 for v in x], [r["local_delta_pp"] for r in rows], .36, label="Local", color="#2a9d8f")
            ax.bar([v+.18 for v in x], [r["all_delta_pp"] for r in rows], .36, label="All-seen", color="#e76f51")
            ax.axhline(0, color="black", linewidth=.7)
            ax.set(xticks=x, xticklabels=[f"{r['cohort'].replace('T9_', '')}\nN={r['N']}" for r in rows],
                   title=f"{protocol} T{task} G-S", ylabel="Paired accuracy difference (pp)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(folder / "subtype_local_all_gain.png", dpi=170)
        plt.close(fig)
    folder = output / "cross_protocol/plots"
    folder.mkdir(parents=True, exist_ok=True)
    rows = [r for r in primary if r["cohort"] == "T9_all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, local_field, all_field, ylabel in ((axes[0], "local_net", "all_net", "Raw net sample count"),
                                             (axes[1], "local_delta_pp", "all_delta_pp", "Paired accuracy difference (pp)")):
        x = list(range(len(rows)))
        ax.bar([v-.18 for v in x], [r[local_field] for r in rows], .36, label="Local", color="#2a9d8f")
        ax.bar([v+.18 for v in x], [r[all_field] for r in rows], .36, label="All-seen", color="#e76f51")
        ax.axhline(0, color="black", linewidth=.7)
        ax.set(xticks=x, xticklabels=[f"{r['protocol']} {r['time']}\nN={r['N']}" for r in rows], ylabel=ylabel)
        ax.legend()
    fig.suptitle("G-S: direct T9 cohort only; no new training / generation")
    fig.tight_layout()
    fig.savefig(folder / "local_all_net_gain.png", dpi=170)
    plt.close(fig)


def fmt(value, signed=False):
    return "—" if value is None else format(value, "+.3f" if signed else ".3f")


def matrix_markdown(row):
    lines = ["| S \\ G | W | C | K |", "|---|---:|---:|---:|"]
    lines.extend(f"| {a} | {row[f'n{a}W']} | {row[f'n{a}C']} | {row[f'n{a}K']} |" for a in STATES)
    return lines


def build_report(output, summaries, collateral, t1, decision):
    primary = [r for r in summaries if r["comparison"] == "G-S"]
    main = {(r["protocol"], r["time"]): r for r in primary if r["cohort"] == "T9_all"}
    lines = ["# T9 Local/All Paired Transition Audit — 결과 보고서", "",
        "작성일: " + datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S KST"), "",
        "## 1. 결론", "",
        "**두 데이터셋에서 공통으로 local acquisition이 개선됐으나 all-seen 경쟁에 가려졌다는 해석은 지지되지 않는다. 현재 generation 방향은 STOP을 유지한다.**", "",
        "IA10 T9의 G−S local 순이득은 **1개(+0.680%p)**지만 T10에는 0개로 바뀐다. CAR10은 T9 local **−5개(−0.660%p)**, T10 local **−8개(−1.055%p)**로 악화됐다. IA10 T9에는 W→C가 없고, local repair 1개는 W→K다. 따라서 그 local repair 자체가 all-seen 경쟁에 가려진 사례는 아니다. 대신 별도의 K→C harm 2개가 전체 net을 상쇄한다.", "",
        "```text"]
    lines.extend(f"{key} = {value}" for key, value in decision.items() if key in (
        "IA10_LOCAL_SIGNAL", "CAR10_LOCAL_SIGNAL", "LOCAL_SIGNAL_PERSISTS_TO_T10", "ALL_SEEN_TRANSLATION",
        "DATASET_COMMON_EVIDENCE", "GENERATION_DIRECTION", "NEXT_METHOD"))
    lines += ["```", "", "YES/NO는 이번 seed의 정미 count 방향이며 통계적 유의성이나 인과 증명이 아니다. Persistence는 명세 Case 1의 ‘양 dataset에서 T9 local net>0, T10 local net≥0’로 판정했다. IA10만의 T10=0은 비음수 조건은 만족하지만 양의 우위가 유지된 것은 아니다. 새 SupCon 코드는 구현·실행하지 않았으며 다음 real-only matched pilot 검토 대상으로만 기록한다.", "",
        "## 2. 분석 범위와 고정 정의", "",
        "새 학습·이미지 생성·증강·inference 없이 기존 S/G/R prediction CSV만 재사용했다. T9/T10에서 local은 sample의 **원 birth task class set**으로 제한된 저장 예측이다. 특히 이 분석의 direct cohort는 birth==9이므로 T10에도 G9를 사용하며, T10 신규 class set으로 바꾸지 않는다.", "",
        "- W: local 오답.\n- C: local 정답, all-seen 오답.\n- K: all-seen 정답(local도 정답).", "",
        "A-within/A-cross는 기존 vanilla persistent Type A 중 birth checkpoint의 local/all prediction으로 고정한 subtype을 사용했다. S/G prediction은 subtype 정의에 사용하지 않았다. T9 non-A는 B1/B2/C/final_correct를 포함하므로 전부 vanilla birth-correct라고 가정하지 않는다. 각 sample의 vanilla_birth_all_correct도 별도 저장했다.", "",
        "| Protocol | T9 direct 전체 | T9 A-within | T9 A-cross | T9 non-A | 과거 birth≤T8 | T10 new |", "|---|---:|---:|---:|---:|---:|---:|",
        "| IA10 | 147 | 27 | 34 | 86 | 1,248 | 124 |", "| CAR10 | 758 | 42 | 77 | 639 | 6,504 | 779 |", "",
        "327/907의 전체 A를 direct acquisition 분모로 쓰지 않는다. Historical cohort는 collateral effect이며 direct repair와 구분한다. Whole seen은 T9에서는 historical+direct, T10에서는 historical+direct+T10 new이다. T1 branch는 별도 acquisition이며 T10까지 이어지지 않았다.", "",
        "기존 prediction은 birth-task local argmax와 all argmax를 저장하지만 true logit은 저장하지 않았다. Optional true-logit column을 임의 복원하지 않았으며 추가 inference가 결론에 필요하지 않아 실행하지 않았다. Margin은 기존 값을 sample CSV에 참고용으로 포함했다.", "",
        "## 3. 필수 summary — G−S", "",
        "Accuracy는 %, 차이는 %p. Local repair=W→C+W→K, local harm=C→W+K→W. All repair=W→K+C→K, all harm=K→W+K→C. `net=repair−harm`, `Δ=100×net/N`.", "",
        "| Protocol | Time | Cohort | N | S local | G local | Local repair/harm/net | Local Δ pp | S all | G all | All repair/harm/net | All Δ pp |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|"]
    for r in primary:
        lines.append(f"| {r['protocol']} | {r['time']} | {r['cohort']} | {r['N']} | {fmt(r['reference_local_accuracy_pct'])} | {fmt(r['target_local_accuracy_pct'])} | {r['local_repair']}/{r['local_harm']}/{r['local_net']:+d} | {fmt(r['local_delta_pp'],True)} | {fmt(r['reference_all_accuracy_pct'])} | {fmt(r['target_all_accuracy_pct'])} | {r['all_repair']}/{r['all_harm']}/{r['all_net']:+d} | {fmt(r['all_delta_pp'],True)} |")
    lines += ["", "![Direct cohort local/all gain](cross_protocol/plots/local_all_net_gain.png)", "",
        "## 4. Class macro 및 class별 net", "",
        "Class macro는 각 cohort에서 sample이 존재하는 class를 동일 가중치로 평균한다. 특정 subtype에 sample이 없는 class를 0으로 넣지 않는다. 모든 cohort의 macro와 per-class raw net은 CSV에 저장했다.", "",
        "| Protocol | Time | Class 수 | S macro local | G macro local | Macro local Δ pp | S macro all | G macro all | Macro all Δ pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        if r["cohort"] == "T9_all":
            lines.append(f"| {r['protocol']} | {r['time']} | {r['class_count']} | {fmt(r['reference_macro_local_accuracy_pct'])} | {fmt(r['target_macro_local_accuracy_pct'])} | {fmt(r['target_macro_local_accuracy_pct']-r['reference_macro_local_accuracy_pct'],True)} | {fmt(r['reference_macro_all_accuracy_pct'])} | {fmt(r['target_macro_all_accuracy_pct'])} | {fmt(r['target_macro_all_accuracy_pct']-r['reference_macro_all_accuracy_pct'],True)} |")
    for protocol in PROTOCOLS:
        class_rows = read_csv(output / protocol / "per_class_summary.csv")
        for task in (9, 10):
            rows = [r for r in class_rows if r["comparison"] == "G-S" and r["cohort"] == "T9_all" and r["time"] == f"T{task}"]
            changed = [f"c{r['true_label']}: local {int(r['local_net']):+d}, all {int(r['all_net']):+d} (N={r['N']})"
                       for r in rows if int(r["local_net"]) or int(r["all_net"])]
            lines += ["", f"{protocol} T{task}의 net이 0이 아닌 class: " + ("; ".join(changed) if changed else "없음") + "."]
    lines += ["", "IA10 T9 micro local +0.680%p는 c178(N=7)의 정미 1개에 해당한다. All micro는 0이지만 class macro all은 +1.825%p이며, 작은 class의 이득과 다른 class의 손실에 동일 class 가중치를 준 차이다. 이 macro 개선도 T10에는 −0.317%p로 바뀐다. Macro를 누락하거나 micro와 혼동하지 않으며 주 판정은 명세의 direct sample-level net 기준을 따른다.", "",
        "Class ID는 기존 remapped label이며 새로운 sample/class selector에 사용하지 않는다.", "",
        "## 5. 모든 고정 cohort의 3×3 transition", "", "행=S, 열=G. 각 표의 9개 cell 합은 해당 cohort N이다. 아래 그림은 direct 전체이고 subtype 그림·표도 함께 제공한다."]
    for protocol in PROTOCOLS:
        for task in (9, 10):
            lines += ["", f"### {protocol} T{task}", "", f"![{protocol} T{task} heatmap]({protocol}/plots/t{task}_T9_all_heatmap.png)"]
            for r in primary:
                if r["protocol"] == protocol and r["time"] == f"T{task}":
                    lines += ["", f"{r['cohort']} (N={r['N']})", ""] + matrix_markdown(r)
        lines += ["", f"![{protocol} fixed subtype gain]({protocol}/plots/subtype_local_all_gain.png)"]
    lines += ["", "## 6. 명세의 핵심 질문에 대한 답", "",
        "1. **T9 전체 cohort local 개선?** IA10은 repair/harm=1/0으로 +1개. CAR10은 2/7로 −5개다.",
        "2. **두 dataset 공통?** 아니다. IA10의 1개 신호만으로 dataset-common acquisition improvement를 주장할 수 없다.",
        "3. **T10까지 유지?** IA10은 1/1, net=0; CAR10은 1/9, net=−8. 공통 양의 이득은 없다. IA10은 명세의 비음수 경계만 만족한다.",
        "4. **W→C가 많았는가?** IA10 T9 0개, T10 1개. CAR10 T9 2개, T10 1개다. 분모는 각각 147/758이며 주된 개선 경로라고 할 수 없다.",
        "5. **Local repair가 all 정답이 되지 않은 sample?** CAR10 T9 W→C 2개는 실제 존재하지만 C→W 4개와 K→W 3개로 상쇄된다. IA10 T9의 유일한 local repair는 W→K라 all 정답까지 복구됐다. ‘local aggregate Δ > all Δ’만으로 해당 repair가 competition에 가려졌다고 해석하면 안 된다.",
        "6. **A-within에 local repair가 집중?** 아니다. IA10 T9 A-within local net=0, CAR10=−1이다. IA10의 local +1은 non-A의 W→K 1개에서 나온다.",
        "7. **A-cross 변화?** IA10 T9/T10에서 local net=0, all net=+1(C→K). CAR10은 T9/T10 모두 local net=−1, all net=−1이다.",
        "8. **A 이득이 다른 harm으로 상쇄?** IA10 T9의 A 전체 all net=+1이 non-A −1과 상쇄된다. 이는 A-within acquisition repair가 아니라 A-cross의 C→K다. CAR10 T9는 A 전체 all net=−1, non-A −3으로 둘 다 손실이다. Non-A 전체를 originally-correct로 부르지는 않는다.",
        "9. **Historical collateral harm?** 아래 별도 표처럼 T9와 T10에서 두 protocol 모두 G−S historical net이 음수다. Direct acquisition 분모와 섞지 않는다.",
        "10. **다음 질문?** 현재 recipe 중단을 유지하고 real-only CE+SupCon matched pilot을 검토한다. 공통 local-positive/all-negative evidence가 없으므로 이번 결과만으로 acquisition→competition interface를 공통 다음 병목으로 확정하지 않는다. SupCon의 효능 역시 아직 검증한 것이 아니다.", "",
        "## 7. Historical collateral 및 whole-seen 영향", "",
        "| Protocol | Time | Scope | N | S all | G all | Repair | Harm | Net | Δ pp |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in collateral:
        if r["comparison"] == "G-S" and r["N"]:
            lines.append(f"| {r['protocol']} | {r['time']} | {r['scope']} | {r['N']} | {fmt(r['reference_all_accuracy_pct'])} | {fmt(r['target_all_accuracy_pct'])} | {r['repair']} | {r['harm']} | {r['net']:+d} | {fmt(r['delta_pp'],True)} |")
    lines += ["", "Scope별 raw repair/harm/net 합이 whole seen의 값과 일치함을 검사했다. Scope별 %p는 분모가 달라 단순 합산하지 않는다. Historical과 T10 new의 local을 G9로 강제 제한하지 않았으며 이 표에서는 all-seen collateral만 보고한다.", "",
        "## 8. T1 별도 재확인", "",
        "| Protocol | 비교 | N | Reference acc | G acc | W→K repair | K→W harm | Net | Δ pp |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in t1:
        lines.append(f"| {r['protocol']} | {r['comparison']} | {r['N']} | {fmt(r['reference_all_accuracy_pct'])} | {fmt(r['target_all_accuracy_pct'])} | {r['all_repair']} | {r['all_harm']} | {r['all_net']:+d} | {fmt(r['all_delta_pp'],True)} |")
    lines += ["", "T1에는 C가 없고 local=all임을 검사했다. T1→T10 retention은 이번 분석으로 추정하지 않는다.", "",
        "## 9. Optional G−R — 주 결론을 대체하지 않음", "",
        "| Protocol | Time | N | Local repair/harm/net | Local Δ pp | All repair/harm/net | All Δ pp |",
        "|---|---|---:|---|---:|---|---:|"]
    for r in summaries:
        if r["comparison"] == "G-R" and r["cohort"] == "T9_all":
            lines.append(f"| {r['protocol']} | {r['time']} | {r['N']} | {r['local_repair']}/{r['local_harm']}/{r['local_net']:+d} | {fmt(r['local_delta_pp'],True)} | {r['all_repair']}/{r['all_harm']}/{r['all_net']:+d} | {fmt(r['all_delta_pp'],True)} |")
    lines += ["", "R을 포함한 모든 고정 cohort의 W/C/K summary·행렬은 CSV/JSON에 저장했다. G−R의 부분적 이득이 있더라도 G−S의 공통 우위 부재를 뒤집지 않는다.", "",
        "## 10. 검증·재현·한계", "",
        "A1: 147/758 direct sample 수, A2: T9 all accuracy 및 정답수(IA S/G=84/84, CAR=621/617), A3: T10 T9 cohort accuracy 및 정답수(IA=84/84, CAR=614/608), A4: 유일 W/C/K state, A5: transition 9-cell 합을 검사했다. 추가로 model seed 1993/protocol/partition, prediction label/birth/task/branch/range, T10 G9 local candidate, 기존 paired repair/harm, whole seen scope 합, vanilla subtype audit의 반복 seed 일치, input SHA-256 불변을 검사했다.", "",
        "이 분석은 prediction 상태의 전이이며 head/representation/label fidelity 원인을 분리하는 새로운 causal intervention이 아니다. 기존 generation pilot의 단일 seed·CAR fine-grained label fidelity 미검증 한계도 그대로 적용된다. Threshold/parent/prompt/subtype을 바꾸지 않았고 generator/GM에 접근하거나 학습하지 않았다.", "",
        "명령어(CPU 분석; GPU 번호 불필요):", "", "```bash", ("cd ${RESEARCH_WORKSPACE}/generation_local_all_transition_audit").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)),
        ("source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT)), "conda activate cds", "CUDA_VISIBLE_DEVICES= python run.py", "```", "",
        "[실행 manifest](run_manifest.json) · [검증·분석 명세](audit.md) · [통합 summary](cross_protocol/summary.csv) · [decision table](cross_protocol/decision_table.csv) · [decision JSON](cross_protocol/decision.json)", "",
        "[IA10 sample transitions](IA10/t9_sample_state_transitions.csv) · [CAR10 sample transitions](CAR10/t9_sample_state_transitions.csv)", "",
        "[IA10 per-class](IA10/per_class_summary.csv) · [CAR10 per-class](CAR10/per_class_summary.csv)", "",
        "[IA10 전체 transition matrices](IA10/transition_matrices.json) · [CAR10 전체 transition matrices](CAR10/transition_matrices.json)", "",
        "[기존 generation pilot 결과](../type_a_generative_augmentation/final_report.md)"]
    (output / "final_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "type_a_generative_augmentation")
    parser.add_argument("--trajectory-root", type=Path, default=ROOT.parent / "error_trajectory_diagnostic")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    source, trajectory_root, output = args.source_root.resolve(), args.trajectory_root.resolve(), args.output_root.resolve()
    for protected in (source, trajectory_root):
        require(output != protected and protected not in output.parents and output not in protected.parents,
                "Output directory must not overlap original experiment/source directories")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    inputs = {}
    def register(path):
        path = path.resolve()
        require(path.is_file(), f"Required saved prediction/mapping missing: {path}; no automatic training/inference fallback")
        if str(path) not in inputs:
            inputs[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return path
    manifest = {"status": "running", "mode": "analysis_only_saved_predictions", "physical_gpu": None,
                "training_performed": False, "generation_performed": False, "inference_performed": False,
                "model_seed": 1993, "source_root": str(source), "trajectory_root": str(trajectory_root),
                "output_root": str(output), "started_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                "code_sha256": sha256(Path(__file__)), "local_candidate_definition": "original birth task; G9 at both T9 and T10",
                "primary_comparison": "G-S", "optional_secondary": "G-R"}
    write_json(output / "run_manifest.json", manifest)
    try:
        all_summaries, all_collateral, all_t1 = [], [], []
        for protocol in PROTOCOLS:
            print(f"[{protocol}] reuse stored S/G/R predictions; no inference", flush=True)
            summaries, collateral, initial = analyze_protocol(source, trajectory_root, output, protocol, register)
            all_summaries.extend(summaries)
            all_collateral.extend(collateral)
            all_t1.extend(initial)
        rows, decision = decisions(all_summaries)
        write_csv(output / "cross_protocol/summary.csv", all_summaries)
        write_csv(output / "cross_protocol/decision_table.csv", rows)
        write_json(output / "cross_protocol/decision.json", decision)
        draw_plots(output, all_summaries)
        build_report(output, all_summaries, all_collateral, all_t1, decision)
        require(all(sha256(path) == info["sha256"] for path, info in inputs.items()), "Original input changed during analysis")
        manifest.update({"status": "complete", "finished_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                         "elapsed_seconds": time.perf_counter()-started, "source_files": inputs,
                         "source_files_unchanged": True, "validation": {"A1_A5": "passed",
                            "fixed_vanilla_subtypes": "passed", "G9_candidate_at_T10": "passed",
                            "previous_accuracy_and_paired_count_parity": "passed", "scope_count_additivity": "passed"},
                         "decision": decision, "plots_count": len(list(output.glob("**/plots/*.png")))})
        write_json(output / "run_manifest.json", manifest)
        print(f"Complete: {output / 'final_report.md'}", flush=True)
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}", "source_files": inputs})
        write_json(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
