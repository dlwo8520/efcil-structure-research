"""Paired, fixed-subset analysis of frozen-Structure score corrections."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


BRANCHES = ("C0", "C1", "C2")
STATES = ("correct", "same_task_wrong", "cross_task_wrong")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def birth_groups(partition: list[int]) -> np.ndarray:
    return np.repeat(np.arange(1, len(partition) + 1), partition)


def states(pred: np.ndarray, labels: np.ndarray, birth: np.ndarray) -> np.ndarray:
    answer = np.full(len(labels), "cross_task_wrong", dtype=object)
    answer[birth[pred] == birth[labels]] = "same_task_wrong"
    answer[pred == labels] = "correct"
    return answer


def paired(base: np.ndarray, pred: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if not n:
        return {"n": 0, "base_correct": 0, "correct": 0, "repair": 0,
                "harm": 0, "net": 0, "delta_pp": 0.0, "accuracy": None}
    old = base[mask] == labels[mask]
    new = pred[mask] == labels[mask]
    repair, harm = int((~old & new).sum()), int((old & ~new).sum())
    delta = 100 * (repair - harm) / n
    if not np.isclose(delta, 100 * (new.mean() - old.mean()), atol=1e-10):
        raise AssertionError("paired repair/harm identity failed")
    return {"n": n, "base_correct": int(old.sum()), "correct": int(new.sum()),
            "repair": repair, "harm": harm, "net": repair - harm,
            "delta_pp": delta, "accuracy": float(new.mean())}


def macro(pred: np.ndarray, labels: np.ndarray, classes: int) -> float:
    correct = pred == labels
    rates = []
    for class_id in range(classes):
        selected = labels == class_id
        if not selected.any():
            raise ValueError(f"missing test class {class_id}")
        rates.append(float(correct[selected].mean()))
    return float(np.mean(rates))


def fixed_types(trajectory: list[dict], ids: np.ndarray, labels: np.ndarray,
                expected_b1: int) -> tuple[np.ndarray, np.ndarray]:
    by_id = {row["sample_id"]: row for row in trajectory}
    if len(by_id) != len(trajectory) or set(by_id) != set(map(str, ids)):
        raise ValueError("trajectory sample IDs and final-feature cache differ")
    types = []
    b1 = []
    for sample_id, label in zip(ids, labels):
        row = by_id[str(sample_id)]
        if int(row["true_label"]) != int(label):
            raise ValueError(f"trajectory label mismatch: {sample_id}")
        group = row["error_type"]
        is_b1 = group == "B_birth_correct_final_wrong" and row["birth_restriction_restored"] == "1"
        if group == "B_birth_correct_final_wrong":
            group = "B1" if is_b1 else "B2"
        elif group == "A_persistent_wrong":
            group = "A"
        elif group == "C_birth_wrong_recovered_final_wrong":
            group = "C"
        elif group != "final_correct":
            raise ValueError(f"unknown fixed trajectory group: {group}")
        types.append(group)
        b1.append(is_b1)
    b1 = np.asarray(b1, dtype=bool)
    if int(b1.sum()) != expected_b1:
        raise AssertionError(f"fixed B1 count {b1.sum()} != {expected_b1}")
    return np.asarray(types), b1


def analyze(protocol: str, ids: np.ndarray, labels: np.ndarray, partition: list[int],
            scores: dict[str, np.ndarray], trajectory: list[dict], output: Path,
            expected_b1: int) -> dict:
    classes = sum(partition)
    n = len(labels)
    if any(matrix.shape != (n, classes) for matrix in scores.values()):
        raise ValueError("score matrix shape mismatch")
    if len(set(map(str, ids))) != n:
        raise ValueError("duplicate test sample ID")
    birth = birth_groups(partition)
    fixed, b1 = fixed_types(trajectory, ids, labels, expected_b1)
    preds = {name: np.asarray(np.argmax(scores[name], axis=1), dtype=np.int64) for name in BRANCHES}
    if (preds["C0"][b1] == labels[b1]).any():
        raise AssertionError("vanilla C0 must be wrong on all fixed B1 samples")
    state = {name: states(preds[name], labels, birth) for name in BRANCHES}
    mask_old = labels < sum(partition[:-1])
    masks = {"overall": np.ones(n, dtype=bool), "old": mask_old, "new": ~mask_old,
             "B1": b1, "retention": birth[labels] < len(partition)}
    for name in ("A", "B1", "B2", "C", "final_correct"):
        masks[name] = masks["retention"] & (fixed == name)
    metrics = {"protocol": protocol, "n": n, "fixed_B1_n": int(b1.sum()), "branches": {}}
    prediction_rows = []
    for j, sample_id in enumerate(ids):
        row = {"sample_id": str(sample_id), "y": int(labels[j]),
               "birth_task": int(birth[labels[j]]), "fixed_error_type": str(fixed[j]),
               "pred_C0": int(preds["C0"][j]), "pred_C1": int(preds["C1"][j]),
               "pred_C2": int(preds["C2"][j]),
               "error_state_C0": state["C0"][j], "error_state_C1": state["C1"][j],
               "error_state_C2": state["C2"][j],
               "delta_true_task_C1": float((scores["C1"][j] - scores["C0"][j])[labels[j]]),
               "delta_true_task_C2": float((scores["C2"][j] - scores["C0"][j])[labels[j]]),
               "delta_pred_task_C1": float((scores["C1"][j] - scores["C0"][j])[preds["C0"][j]]),
               "delta_pred_task_C2": float((scores["C2"][j] - scores["C0"][j])[preds["C0"][j]])}
        prediction_rows.append(row)
    write_csv(output / "sample_predictions.csv", prediction_rows)
    transition_rows, destination_rows, true_task_rows = [], [], []
    for name in BRANCHES:
        pred = preds[name]
        branch = {"accuracy": float((pred == labels).mean()),
                  "macro_accuracy": macro(pred, labels, classes),
                  "subsets": {key: paired(preds["C0"], pred, labels, mask) for key, mask in masks.items()},
                  "same_task_errors": int((state[name] == "same_task_wrong").sum()),
                  "cross_task_errors": int((state[name] == "cross_task_wrong").sum()),
                  "old_to_new": int((mask_old & (pred >= sum(partition[:-1]))).sum()),
                  "new_to_old": int((~mask_old & (pred < sum(partition[:-1]))).sum())}
        metrics["branches"][name] = branch
        for source in STATES:
            for target in STATES:
                transition_rows.append({"protocol": protocol, "branch": name,
                                        "from_C0": source, "to_branch": target,
                                        "n": int(((state["C0"] == source) & (state[name] == target)).sum())})
        for true_task in range(1, len(partition) + 1):
            own = birth[labels] == true_task
            result = paired(preds["C0"], pred, labels, own)
            true_task_rows.append({"protocol": protocol, "branch": name, "true_task": true_task, **result})
            for destination in range(1, len(partition) + 1):
                destination_rows.append({"protocol": protocol, "branch": name,
                                         "true_task": true_task, "pred_task": destination,
                                         "wrong_n": int((own & (pred != labels) &
                                                         (birth[pred] == destination)).sum())})
        if name != "C0":
            harmful = (state["C0"] == "correct") & (state[name] == "same_task_wrong")
            if harmful.any():
                raise AssertionError(f"{name}: baseline correct -> same-task wrong harm")
    write_csv(output / "transition_tables.csv", transition_rows)
    write_csv(output / "error_destination_task.csv", destination_rows)
    write_csv(output / "true_task_repair_harm.csv", true_task_rows)
    return metrics
