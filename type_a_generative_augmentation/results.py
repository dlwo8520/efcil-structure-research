"""Matched sample comparisons and predeclared expansion decision."""
from __future__ import annotations

import numpy as np

from .common import (ROOT, PROTOCOLS, run_dir, source_config, checkpoint_path,
                     read_json, read_csv, write_json, write_csv)


def paired(left, right, mask):
    a = np.array([int(r["pred"]) == int(r["y"]) for r in left])[mask]
    b = np.array([int(r["pred"]) == int(r["y"]) for r in right])[mask]
    n = int(mask.sum())
    if not n:
        return {"n": 0, "repair": 0, "harm": 0, "net": 0, "delta_pp": None}
    repair, harm = int((~a & b).sum()), int((a & ~b).sum())
    delta = 100 * (repair - harm) / n
    if not np.isclose(delta, 100 * (b.mean() - a.mean()), atol=1e-10):
        raise AssertionError("paired accuracy identity failed")
    return {"n": n, "repair": repair, "harm": harm, "net": repair - harm, "delta_pp": delta}


def summary(rows, known, current_task):
    labels = np.array([int(r["y"]) for r in rows])
    correct = np.array([int(r["pred"]) == int(r["y"]) for r in rows])
    new, old = labels >= known, labels < known
    result = {"n": len(rows), "correct": int(correct.sum()), "accuracy": float(correct.mean()),
            "macro_accuracy": float(np.mean([correct[labels == c].mean() for c in np.unique(labels)])),
            "old_accuracy": float(correct[old].mean()) if old.any() else None,
            "new_accuracy": float(correct[new].mean()), "new_n": int(new.sum()),
            "new_local_accuracy": float(np.mean([int(r["local_pred"]) == int(r["y"]) for r in rows if int(r["y"]) >= known]))
            if all(str(r.get("local_pred", "")) for r in rows) else None,
            "T9_cohort_accuracy": float(correct[np.array([int(r["birth_task"]) == 9 for r in rows])].mean())
            if current_task >= 9 else None}
    for field in ("head_CE", "margin_all", "margin_within"):
        result[field + "_mean"] = float(np.mean([float(r[field]) for r in rows])) if all(str(r.get(field, "")) for r in rows) else None
        result["new_" + field + "_mean"] = float(np.mean([float(r[field]) for r in rows if int(r["y"]) >= known])) if result[field + "_mean"] is not None else None
    error_directions = {"new_to_old": 0, "new_to_new": 0, "old_to_new": 0, "old_to_old": 0}
    for row, good in zip(rows, correct):
        if not good:
            key = ("new" if int(row["y"]) >= known else "old") + "_to_" + ("new" if int(row["pred"]) >= known else "old")
            error_directions[key] += 1
    result["error_directions"] = error_directions
    return result


def retention_transitions(birth_rows, final_rows):
    """T9 cohort only; fixed vanilla error labels never redefine the subset."""
    birth = {r["sample_id"]: r for r in birth_rows if int(r["birth_task"]) == 9}
    final = {r["sample_id"]: r for r in final_rows if int(r["birth_task"]) == 9}
    if set(birth) != set(final):
        raise AssertionError("T9 retention identities differ")
    mask = np.ones(len(birth), bool)
    transition = paired(list(birth.values()), [final[key] for key in birth], mask)
    return {"n": transition["n"], "birth_correct_to_T10_wrong": transition["harm"],
            "birth_wrong_to_T10_correct": transition["repair"], "net": transition["net"],
            "T10_minus_T9_accuracy_pp": transition["delta_pp"]}


def baseline_rows(protocol, task):
    partition = source_config(protocol)["task_partition"]
    trajectory = read_csv(ROOT.parent / "error_trajectory_diagnostic" / protocol / "sample_trajectory.csv")
    rows = []
    for row in trajectory:
        if int(row["true_label"]) >= sum(partition[:task]):
            continue
        kind = row["error_type"]
        if kind.startswith("A_"):
            kind = "A"
        elif kind.startswith("C_"):
            kind = "C"
        elif kind.startswith("B_"):
            kind = "B1" if row["birth_restriction_restored"] == "1" else "B2"
        rows.append({"sample_id": row["sample_id"], "y": int(row["true_label"]),
                     "birth_task": int(row["birth_task"]), "branch": "V", "fixed_type": kind,
                     "evaluated_after_task": task, "pred": int(row[f"pred_T{task:02d}"]), "local_pred": ""})
    return rows


def verify_stats_fork(protocol, location):
    import torch
    from oracle_audit.runtime import torch_load
    task = 1 if location == "T1" else 9
    g_dir, gm_dir = run_dir(protocol, location, "G"), run_dir(protocol, location, "GM")
    g = torch_load(g_dir / "task_checkpoints" / f"task_{task:02d}.pt")
    gm = torch_load(gm_dir / "task_checkpoints" / f"task_{task:02d}.pt")
    representation_equal = all(torch.equal(value, gm["network_state_dict"][key])
                               for key, value in g["network_state_dict"].items() if key.startswith("convnet."))
    if not representation_equal:
        raise AssertionError("G/GM representation differs at statistics fork")
    lower = sum(g["task_sizes"][:-1])
    if not np.array_equal(g["class_means"][:lower], gm["class_means"][:lower]):
        raise AssertionError("G/GM changed old APR means at fork")
    if not torch.equal(g["class_covs"][:lower], gm["class_covs"][:lower]):
        raise AssertionError("G/GM changed old covariance at fork")
    if task == 9:
        side_g = torch_load(g_dir / "pre_ca_classifier/task_09.pt")
        side_gm = torch_load(gm_dir / "pre_ca_classifier/task_09.pt")
        if not all(torch.equal(v, side_gm["classifier_state_dict"][k]) for k, v in side_g["classifier_state_dict"].items()):
            raise AssertionError("G/GM pre-CA initialization differs")
    return {"same_representation_at_fork": True, "same_old_APR_moments_at_fork": True,
            "statistics_changes": read_json(g_dir / "statistics_comparison.json")}


def build_report():
    all_rows, metrics, stats, resources = [], {}, {}, {}
    for protocol in PROTOCOLS:
        partition = source_config(protocol)["task_partition"]
        metrics[protocol] = {"conditions": {}, "comparisons": {}, "T9_retention": {}}
        resources[protocol], stats[protocol] = {}, {}
        for location in ("T1", "T9_T10"):
            tasks = (1,) if location == "T1" else (9, 10)
            stats[protocol][location] = verify_stats_fork(protocol, location)
            starts = [read_json(run_dir(protocol, location, b) / "start_manifest.json") for b in ("R", "S", "G")]
            if len({r["initial_network_sha256"] for r in starts}) != 1:
                raise AssertionError("R/S/G starting model differs")
            for task in tasks:
                traces = [read_json(run_dir(protocol, location, b) / f"matched_training_T{task:02d}.json") for b in ("R", "S", "G")]
                for field in ("anchor_sha256", "parent_schedule_sha256", "epochs", "optimizer", "batch_size_anchor"):
                    if field == "epochs":
                        exposures = [[(e["anchor_samples"], e["forward_backward_images"], e["optimizer_steps"], e["learning_rates_after_step"])
                                      for e in trace[field]] for trace in traces]
                        if any(item != exposures[0] for item in exposures[1:]):
                            raise AssertionError("R/S/G optimizer budget differs")
                    elif any(trace[field] != traces[0][field] for trace in traces[1:]):
                        raise AssertionError(f"R/S/G fairness mismatch: {field}")
                known = sum(partition[:task - 1])
                predictions = {b: read_csv(run_dir(protocol, location, b) / f"predictions_T{task:02d}.csv")
                               for b in ("R", "S", "G", "GM")}
                predictions["V"] = baseline_rows(protocol, task)
                reference = predictions["R"]
                for branch, rows in predictions.items():
                    by_id = {r["sample_id"]: r for r in rows}
                    if len(by_id) != len(rows) or set(by_id) != {r["sample_id"] for r in reference}:
                        raise AssertionError("paired evaluation sample identities differ")
                    predictions[branch] = [by_id[r["sample_id"]] for r in reference]
                    if any(int(a["y"]) != int(b["y"]) for a, b in zip(reference, predictions[branch])):
                        raise AssertionError("paired evaluation class remapping differs")
                    metrics[protocol]["conditions"][f"{location}_T{task}_{branch}"] = summary(predictions[branch], known, task)
                    all_rows.extend({"protocol": protocol, "location": location, **r} for r in predictions[branch])
                labels = np.array([int(r["y"]) for r in reference])
                birth = np.array([int(r["birth_task"]) for r in reference])
                fixed = np.array([r["fixed_type"] for r in reference])
                masks = {"overall": np.ones(len(labels), bool), "new": labels >= known,
                         "old": labels < known, "T9": birth == 9}
                for kind in ("A", "B1", "B2", "C", "final_correct"):
                    masks[kind] = (birth < 10) & (fixed == kind)
                masks["B"] = masks["B1"] | masks["B2"]
                for comparison, left, right in (("G-S", "S", "G"), ("G-R", "R", "G"),
                                                ("GM-G", "G", "GM"), ("S-R", "R", "S"), ("G-V", "V", "G")):
                    metrics[protocol]["comparisons"][f"{location}_T{task}_{comparison}"] = {
                        key: paired(predictions[left], predictions[right], mask) for key, mask in masks.items()}
            for branch in ("R", "S", "G", "GM"):
                path = run_dir(protocol, location, branch) / "resource_usage.json"
                resources[protocol][f"{location}_{branch}"] = read_json(path) if path.exists() else {"shared_G_training": True}
            if location == "T9_T10":
                for branch in ("V", "R", "S", "G", "GM"):
                    rows9 = baseline_rows(protocol, 9) if branch == "V" else read_csv(run_dir(protocol, location, branch) / "predictions_T09.csv")
                    rows10 = baseline_rows(protocol, 10) if branch == "V" else read_csv(run_dir(protocol, location, branch) / "predictions_T10.csv")
                    metrics[protocol]["T9_retention"][branch] = retention_transitions(rows9, rows10)
        c = metrics[protocol]["comparisons"]
        task1 = c["T1_T1_G-S"]["new"]["delta_pp"]
        task9 = c["T9_T10_T9_G-S"]["new"]["delta_pp"]
        suffix = c["T9_T10_T10_G-S"]
        gr_birth = float(np.mean([c["T1_T1_G-R"]["new"]["delta_pp"], c["T9_T10_T9_G-R"]["new"]["delta_pp"]]))
        checks = {"birth_task_mean_G_minus_S_at_least_1pp": (task1 + task9) / 2 >= 1,
                  "no_birth_loss_below_minus_0p5pp": min(task1, task9) >= -0.5,
                  "T9_seen_not_below_minus_0p5pp": c["T9_T10_T9_G-S"]["overall"]["delta_pp"] >= -0.5,
                  "T9_cohort_after_T10_at_least_plus_0p5pp": suffix["T9"]["delta_pp"] >= 0.5,
                  "T10_seen_does_not_drop": suffix["overall"]["delta_pp"] >= 0,
                  "old_new_not_below_minus_1pp": all(c[key][subset]["delta_pp"] >= -1 for key in
                       ("T9_T10_T9_G-S", "T9_T10_T10_G-S") for subset in ("old", "new")),
                  "better_than_R_birth_mean_and_T10_seen": gr_birth > 0 and c["T9_T10_T10_G-R"]["overall"]["delta_pp"] > 0}
        metrics[protocol]["decision"] = {"status": "PASS" if all(checks.values()) else "STOP_CURRENT_RECIPE", "checks": checks}
    write_csv(ROOT / "per_sample_predictions.csv", all_rows)
    exploratory = (ROOT / "exploratory_config.json").exists()
    if exploratory:
        from .automatic_sanity import load_accepted, LIMITATION
        load_accepted()
        metrics["experiment_scope"] = {"mode": read_json(ROOT / "exploratory_config.json"), "limitation": LIMITATION,
                                       "automatic_sanity": read_json(ROOT / "automatic_sanity_summary.json")}
    write_json(ROOT / "metrics.json", metrics)
    write_json(ROOT / "statistics_comparison.json", stats)
    write_json(ROOT / "resource_usage.json", {"generation": read_json(ROOT / "generation_resource_usage.json"), "training": resources,
               "automatic_sanity": read_json(ROOT / "automatic_sanity_summary.json")["resource_usage"] if exploratory else None})
    lines = ["# Type A current generative augmentation pilot", "",
             "Current train parents only; fixed vanilla error subsets are diagnostic/evaluation labels. "
             "Primary comparison G−S, secondary G−R, statistics-only fork GM−G. "
             "T1 acquisition and T9→T10 suffix do not constitute full T1→T10 augmented training.", "",
             "## Accuracy", "", "| Protocol | Location/task/branch | N | Seen | Macro | New all-seen | New local | T9 cohort |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    if exploratory:
        lines[2:2] = ["## Exploratory scope and label-fidelity limitation", "", LIMITATION,
                      "Contact sheets are inspection artifacts only; no human per-image choices affect inclusion. "
                      "All fixed-rule survivors are used in the common R/S/G parent bank. "
                      "Promising pilot results recommend stricter label-preservation/human QC and additional seeds before any formal efficacy claim.", "",
                      "[Automatic sanity report](automatic_sanity_report.md) · [Fixed rules](sanity_policy.json) · "
                      "[Similarity/counts](automatic_sanity_summary.json) · [Exploratory configuration](exploratory_config.json)", ""]
    fmt = lambda x: "—" if x is None else f"{100*x:.3f}%"
    for protocol in PROTOCOLS:
        for key, row in metrics[protocol]["conditions"].items():
            lines.append(f"| {protocol} | {key} | {row['n']} | {fmt(row['accuracy'])} | {fmt(row['macro_accuracy'])} | "
                         f"{fmt(row['new_accuracy'])} | {fmt(row['new_local_accuracy'])} | {fmt(row['T9_cohort_accuracy'])} |")
    lines += ["", "## Paired comparisons", "", "| Protocol | Comparison | Subset | N | Repair | Harm | Net | ΔAcc pp |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for key, comparison in metrics[protocol]["comparisons"].items():
            if not key.endswith(("G-S", "G-R", "GM-G")):
                continue
            for subset, row in comparison.items():
                if row["n"]:
                    lines.append(f"| {protocol} | {key} | {subset} | {row['n']} | {row['repair']} | {row['harm']} | {row['net']} | {row['delta_pp']:+.3f} |")
    lines += ["", "## Decision", ""]
    for protocol in PROTOCOLS:
        decision = metrics[protocol]["decision"]
        scope = " Exploratory numerical screen only; not formal label/QC validation." if exploratory else ""
        lines.append(f"- {protocol}: **{decision['status']}**; checks: `{decision['checks']}`.{scope}")
    lines += ["", "## T9 birth → T10 transitions", "", "| Protocol | Branch | N | Birth correct → wrong | Birth wrong → correct | Net |",
              "|---|---|---:|---:|---:|---:|"]
    for protocol in PROTOCOLS:
        for branch, row in metrics[protocol]["T9_retention"].items():
            lines.append(f"| {protocol} | {branch} | {row['n']} | {row['birth_correct_to_T10_wrong']} | {row['birth_wrong_to_T10_correct']} | {row['net']} |")
    lines += ["", "## Head diagnostics", "", "CE uses scaled cosine logits (no training margin at evaluation); local scores are diagnostic, not oracle-task inference. V has no cached per-sample logits: CE/margins are unavailable, not zero.", "",
              "| Protocol | Location/task/branch | Mean CE | Mean all margin | Mean local margin | New→old | New→new | Old→new | Old→old |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    scalar = lambda x: "—" if x is None else f"{x:.4f}"
    for protocol in PROTOCOLS:
        for key, row in metrics[protocol]["conditions"].items():
            direction = row["error_directions"]
            lines.append(f"| {protocol} | {key} | {scalar(row['head_CE_mean'])} | {scalar(row['margin_all_mean'])} | {scalar(row['margin_within_mean'])} | "
                         f"{direction['new_to_old']} | {direction['new_to_new']} | {direction['old_to_new']} | {direction['old_to_old']} |")
    lines += ["", "Both datasets must pass before extending to full trajectory. A failed recipe does not refute all coverage/generative augmentation hypotheses. "
              "Improved support is an association, and G−V includes added exposure/compute. "
              "G/GM representation equality applies at the statistics fork only; later T10 adaptation may diverge.", "",
              "## Evidence", "", "[IA10 birth audit](audit/IA10/report.md) · [CAR10 birth audit](audit/CAR10/report.md) · "
              "[Parent distribution](parent_distribution.csv) · " +
              ("[Automatic sanity](automatic_sanity_report.md) · " if exploratory else "[Quality](generation_quality.md) · ") +
              "[Configuration](config.json) · [Metrics](metrics.json) · [Sample predictions](per_sample_predictions.csv) · "
              "[Statistics fork](statistics_comparison.json) · [Resources](resource_usage.json)"]
    (ROOT / "final_report.md").write_text("\n".join(lines) + "\n")
    print(f"Report: {ROOT / 'final_report.md'}", flush=True)
