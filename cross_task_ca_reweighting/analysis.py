"""Paired real-test outcomes for the fixed two-branch CA pilot."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        result = list(csv.DictReader(handle))
    if not result:
        raise ValueError(f"empty input: {path}")
    return result


def birth_by_label(partition: list[int]) -> np.ndarray:
    return np.repeat(np.arange(1, len(partition) + 1), partition).astype(np.int64)


def _class_macro_accuracy(correct: np.ndarray, labels: np.ndarray, classes: int) -> float:
    rates = []
    for class_id in range(classes):
        mask = labels == class_id
        if not mask.any():
            raise ValueError(f"class {class_id} has no real test sample")
        rates.append(float(correct[mask].mean()))
    return float(np.mean(rates))


def _branch_summary(pred: np.ndarray, labels: np.ndarray, partition: list[int]) -> dict:
    correct = pred == labels
    group = birth_by_label(partition)
    same = (~correct) & (group[pred] == group[labels])
    cross = (~correct) & (group[pred] != group[labels])
    if np.any(same & cross) or int(same.sum() + cross.sum()) != int((~correct).sum()):
        raise AssertionError("same/cross errors do not partition all errors")
    known_before = sum(partition[:-1])
    old = labels < known_before
    new = ~old
    if not old.any() or not new.any():
        raise AssertionError("final-task old/new groups are missing")
    old_to_new = old & (pred >= known_before)
    new_to_old = new & (pred < known_before)
    return {
        "accuracy": float(correct.mean()),
        "correct_n": int(correct.sum()),
        "class_macro_accuracy": _class_macro_accuracy(correct, labels, sum(partition)),
        "old_accuracy": float(correct[old].mean()),
        "new_accuracy": float(correct[new].mean()),
        "old_n": int(old.sum()), "new_n": int(new.sum()),
        "old_to_new_n": int(old_to_new.sum()),
        "new_to_old_n": int(new_to_old.sum()),
        "old_to_new_rate": float(old_to_new.sum() / old.sum()),
        "new_to_old_rate": float(new_to_old.sum() / new.sum()),
        "same_task_error_n": int(same.sum()),
        "cross_task_error_n": int(cross.sum()),
        "same_task_error_rate": float(same.mean()),
        "cross_task_error_rate": float(cross.mean()),
    }


def paired_summary(control: np.ndarray, intervention: np.ndarray, labels: np.ndarray,
                   selected: np.ndarray) -> dict:
    if not (control.shape == intervention.shape == labels.shape == selected.shape):
        raise ValueError("paired arrays have different lengths")
    c = control[selected] == labels[selected]
    i = intervention[selected] == labels[selected]
    n = int(selected.sum())
    if not n:
        raise ValueError("paired subset is empty")
    repair = int((~c & i).sum())
    harm = int((c & ~i).sum())
    delta = (repair - harm) / n
    observed = float(i.mean() - c.mean())
    if not np.isclose(delta, observed, atol=1e-12, rtol=0):
        raise AssertionError("DeltaAcc != repair-minus-harm divided by N")
    return {
        "n": n, "control_correct_n": int(c.sum()), "intervention_correct_n": int(i.sum()),
        "control_accuracy": float(c.mean()), "intervention_accuracy": float(i.mean()),
        "repair_n": repair, "harm_n": harm, "net_gain_n": repair - harm,
        "delta_accuracy": delta,
    }


def make_tables(protocol: str, ids: np.ndarray, labels: np.ndarray,
                control_scores: np.ndarray, intervention_scores: np.ndarray,
                baseline_scores: np.ndarray, b1_mask: np.ndarray,
                partition: list[int], output: Path) -> dict:
    n, classes = control_scores.shape
    if intervention_scores.shape != (n, classes) or baseline_scores.shape != (n, classes):
        raise ValueError("test score matrices differ in shape")
    if len(ids) != n or len(labels) != n or len(b1_mask) != n or classes != sum(partition):
        raise ValueError("test IDs, labels, B1 mask or class count differ")
    if len(set(map(str, ids))) != n:
        raise AssertionError("duplicate test sample IDs")
    control = np.argmax(control_scores, axis=1).astype(np.int64)
    intervention = np.argmax(intervention_scores, axis=1).astype(np.int64)
    baseline = np.argmax(baseline_scores, axis=1).astype(np.int64)
    group = birth_by_label(partition)
    baseline_b1 = baseline[b1_mask]
    baseline_b1_correct_n = int((baseline_b1 == labels[b1_mask]).sum())
    if baseline_b1_correct_n > 2:
        raise AssertionError("more than two fixed B1 IDs disagree with cached baseline inference")
    branch = {"control": _branch_summary(control, labels, partition),
              "intervention": _branch_summary(intervention, labels, partition),
              "vanilla_cached": _branch_summary(baseline, labels, partition)}
    overall = paired_summary(control, intervention, labels, np.ones(n, dtype=bool))
    b1 = paired_summary(control, intervention, labels, b1_mask)
    result = {"protocol": protocol, "overall": overall, "b1": b1, "branch": branch,
              "b1_expected_n": int(b1_mask.sum()),
              "baseline_b1_correct_under_cached_features_n": baseline_b1_correct_n}
    for name, scores, pred in (("control", control_scores, control),
                               ("intervention", intervention_scores, intervention)):
        rows = []
        for j in range(n):
            rows.append({
                "protocol": protocol, "sample_id": str(ids[j]), "true_label": int(labels[j]),
                "true_task_group": int(group[labels[j]]), "pred_class": int(pred[j]),
                "pred_task_group": int(group[pred[j]]), "correct": int(pred[j] == labels[j]),
                "same_task_error": int(pred[j] != labels[j] and group[pred[j]] == group[labels[j]]),
                "cross_task_error": int(pred[j] != labels[j] and group[pred[j]] != group[labels[j]]),
                "true_class_logit": float(scores[j, labels[j]]),
                "top1_logit": float(scores[j, pred[j]]),
            })
        write_csv(output / f"{name}_predictions.csv", rows)
    pairs = []
    for j in range(n):
        pairs.append({
            "protocol": protocol, "sample_id": str(ids[j]), "true_label": int(labels[j]),
            "birth_task": int(group[labels[j]]), "baseline_b1": int(b1_mask[j]),
            "control_pred": int(control[j]), "intervention_pred": int(intervention[j]),
            "control_correct": int(control[j] == labels[j]),
            "intervention_correct": int(intervention[j] == labels[j]),
            "repair": int(control[j] != labels[j] and intervention[j] == labels[j]),
            "harm": int(control[j] == labels[j] and intervention[j] != labels[j]),
        })
    write_csv(output / "paired_repair_harm.csv", pairs)
    b1_rows = []
    for j in np.flatnonzero(b1_mask):
        b1_rows.append({
            "protocol": protocol, "sample_id": str(ids[j]), "true_label": int(labels[j]),
            "true_task_group": int(group[labels[j]]),
            "baseline_top1_class": int(baseline[j]),
            "control_top1_class": int(control[j]),
            "control_pred_task_group": int(group[control[j]]),
            "intervention_top1_class": int(intervention[j]),
            "intervention_pred_task_group": int(group[intervention[j]]),
            "control_correct": int(control[j] == labels[j]),
            "intervention_correct": int(intervention[j] == labels[j]),
            "repair": int(control[j] != labels[j] and intervention[j] == labels[j]),
            "harm": int(control[j] == labels[j] and intervention[j] != labels[j]),
            "control_true_logit": float(control_scores[j, labels[j]]),
            "intervention_true_logit": float(intervention_scores[j, labels[j]]),
        })
    write_csv(output / "b1_results.csv", b1_rows)
    group_rows = []
    for name in ("control", "intervention"):
        stats = branch[name]
        group_rows.append({
            "protocol": protocol, "branch": name, "n_test": n,
            "same_task_error_n": stats["same_task_error_n"],
            "same_task_error_rate": stats["same_task_error_rate"],
            "cross_task_error_n": stats["cross_task_error_n"],
            "cross_task_error_rate": stats["cross_task_error_rate"],
            "old_to_new_n": stats["old_to_new_n"], "old_to_new_rate": stats["old_to_new_rate"],
            "new_to_old_n": stats["new_to_old_n"], "new_to_old_rate": stats["new_to_old_rate"],
        })
    write_csv(output / "error_group_summary.csv", group_rows)
    return result
