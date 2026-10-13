"""Minimal, mutually exclusive correct/wrong trajectory analysis."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


TYPE_A = "A_persistent_wrong"
TYPE_B = "B_birth_correct_final_wrong"
TYPE_C = "C_birth_wrong_recovered_final_wrong"
ERROR_TYPES = (TYPE_A, TYPE_B, TYPE_C)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def birth_tasks(labels: np.ndarray, partition: list[int]) -> np.ndarray:
    boundaries = np.cumsum(np.asarray(partition, dtype=np.int64))
    if labels.min() < 0 or labels.max() >= boundaries[-1]:
        raise ValueError("label outside task partition")
    return np.searchsorted(boundaries, labels, side="right").astype(np.int64) + 1


def trajectories(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    partition: list[int],
    predictions: np.ndarray,
    correctness: np.ndarray,
    final_scores: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Rows are canonical T10 test samples; pre-birth prediction cells are blank."""
    n, task_count = predictions.shape
    if task_count != 10 or correctness.shape != predictions.shape or len(labels) != n:
        raise ValueError("trajectory dimensions disagree")
    if final_scores.shape != (n, sum(partition)):
        raise ValueError("T10 score matrix dimensions disagree")
    birth = birth_tasks(labels, partition)
    for i in range(n):
        before = birth[i] - 1
        if np.any(predictions[i, :before] != -1) or np.any(correctness[i, :before] != -1):
            raise AssertionError("pre-birth evaluation leaked into trajectory")
        if np.any(predictions[i, before:] < 0) or not np.isin(correctness[i, before:], (0, 1)).all():
            raise AssertionError("missing post-birth prediction")
    birth_correct = correctness[np.arange(n), birth - 1] == 1
    final_correct = correctness[:, -1] == 1
    final_wrong = ~final_correct
    persistent = np.zeros(n, dtype=bool)
    intermediate_recovery = np.zeros(n, dtype=bool)
    for i in range(n):
        persistent[i] = not correctness[i, birth[i] - 1 :].any()
        intermediate_recovery[i] = bool(np.any(correctness[i, birth[i] : -1] == 1))
    type_a = final_wrong & ~birth_correct & persistent
    type_b = final_wrong & birth_correct
    type_c = final_wrong & ~birth_correct & intermediate_recovery
    if not np.array_equal(type_a | type_b | type_c, final_wrong):
        raise AssertionError("A/B/C do not cover all final errors")
    if np.any((type_a.astype(int) + type_b.astype(int) + type_c.astype(int)) > 1):
        raise AssertionError("A/B/C are not mutually exclusive")
    if int(type_a.sum() + type_b.sum() + type_c.sum()) != int(final_wrong.sum()):
        raise AssertionError("N_A + N_B + N_C != N_final_wrong")
    if np.any((birth == 10) & (type_b | type_c)):
        raise AssertionError("T10-born class cannot be B/C")

    restricted_prediction = np.full(n, -1, dtype=np.int64)
    restored = np.zeros(n, dtype=bool)
    boundaries = np.cumsum(partition)
    for i in np.flatnonzero(type_b):
        restricted_prediction[i] = int(np.argmax(final_scores[i, : boundaries[birth[i] - 1]]))
        restored[i] = restricted_prediction[i] == labels[i]
    if int((type_b & restored).sum() + (type_b & ~restored).sum()) != int(type_b.sum()):
        raise AssertionError("B1/B2 do not partition Type B")

    rows: list[dict] = []
    for i in range(n):
        kind = TYPE_A if type_a[i] else TYPE_B if type_b[i] else TYPE_C if type_c[i] else "final_correct"
        row = {
            "protocol": "", "sample_id": str(sample_ids[i]), "true_label": int(labels[i]),
            "birth_task": int(birth[i]),
            "pred_Tbirth": int(predictions[i, birth[i] - 1]),
            "correct_Tbirth": int(birth_correct[i]),
            "error_type": kind,
            "birth_restricted_pred_T10": int(restricted_prediction[i]) if type_b[i] else "",
            "birth_restriction_restored": int(restored[i]) if type_b[i] else "",
        }
        for task in range(1, 11):
            row[f"pred_T{task:02d}"] = int(predictions[i, task - 1]) if task >= birth[i] else ""
            row[f"correct_T{task:02d}"] = int(correctness[i, task - 1]) if task >= birth[i] else ""
        row["true_class_score_T10"] = float(final_scores[i, labels[i]])
        row["top1_score_T10"] = float(final_scores[i].max())
        rows.append(row)
    state = {
        "birth": birth, "birth_correct": birth_correct, "final_correct": final_correct,
        "final_wrong": final_wrong, "type_a": type_a, "type_b": type_b,
        "type_c": type_c, "recovered_at_final": ~birth_correct & final_correct,
        "b1": type_b & restored, "b2": type_b & ~restored,
    }
    return rows, state


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _macro_rate(numer: np.ndarray, denom: np.ndarray, labels: np.ndarray,
                selected: np.ndarray) -> tuple[float | None, int]:
    values = []
    for class_id in np.unique(labels[selected]):
        mask = selected & (labels == class_id)
        count = int(denom[mask].sum())
        if count:
            values.append(float(numer[mask].sum()) / count)
    return (float(np.mean(values)) if values else None, len(values))


def _macro_burden(numer: np.ndarray, labels: np.ndarray, selected: np.ndarray) -> float:
    values = []
    for class_id in np.unique(labels[selected]):
        mask = selected & (labels == class_id)
        values.append(float(numer[mask].sum()) / int(mask.sum()))
    return 100 * float(np.mean(values)) if values else 0.0


def aggregate(protocol: str, labels: np.ndarray, state: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    birth = state["birth"]
    selections = []
    for scope, base in (("all_final_classes", np.ones(len(labels), bool)),
                        ("retention_birth_le9", birth <= 9)):
        selections.append((scope, "micro", "all", base, False))
        selections.append((scope, "class_macro", "all", base, True))
        for task in range(1, 11 if scope == "all_final_classes" else 10):
            group = base & (birth == task)
            selections.append((scope, "birth_task_micro", f"T{task}", group, False))
            selections.append((scope, "birth_task_class_macro", f"T{task}", group, True))
    composition: list[dict] = []
    loss_recovery: list[dict] = []
    restriction: list[dict] = []
    for scope, grouping, group_id, selected, macro in selections:
        n = int(selected.sum())
        wrong = int((selected & state["final_wrong"]).sum())
        a = int((selected & state["type_a"]).sum())
        b = int((selected & state["type_b"]).sum())
        c = int((selected & state["type_c"]).sum())
        if a + b + c != wrong:
            raise AssertionError(f"{protocol} {scope} {grouping} {group_id}: A+B+C mismatch")
        comp = {"protocol": protocol, "scope": scope, "grouping": grouping,
                "group_id": group_id, "n_test": n, "n_final_wrong": wrong,
                "n_A": a, "n_B": b, "n_C": c}
        for letter, key in (("A", "type_a"), ("B", "type_b"), ("C", "type_c")):
            numerator = state[key]
            if macro:
                comp[f"{letter}_final_error_share"], valid = _macro_rate(
                    numerator, state["final_wrong"], labels, selected)
                comp[f"{letter}_valid_error_classes"] = valid
                comp[f"{letter}_all_test_burden_pp"] = _macro_burden(numerator, labels, selected)
            else:
                comp[f"{letter}_final_error_share"] = _ratio(int((selected & numerator).sum()), wrong)
                comp[f"{letter}_valid_error_classes"] = ""
                comp[f"{letter}_all_test_burden_pp"] = 100 * int((selected & numerator).sum()) / n if n else None
        composition.append(comp)

        bc = state["birth_correct"]
        bw = ~bc
        lost = state["type_b"]
        recovered = state["recovered_at_final"]
        lr = {"protocol": protocol, "scope": scope, "grouping": grouping, "group_id": group_id,
              "n_test": n, "birth_correct_n": int((selected & bc).sum()),
              "lost_at_final_n": int((selected & lost).sum()),
              "birth_wrong_n": int((selected & bw).sum()),
              "recovered_at_final_n": int((selected & recovered).sum())}
        if macro:
            lr["loss_rate"], lr["valid_loss_classes"] = _macro_rate(lost, bc, labels, selected)
            lr["recovery_rate"], lr["valid_recovery_classes"] = _macro_rate(recovered, bw, labels, selected)
        else:
            lr["loss_rate"] = _ratio(lr["lost_at_final_n"], lr["birth_correct_n"])
            lr["recovery_rate"] = _ratio(lr["recovered_at_final_n"], lr["birth_wrong_n"])
            lr["valid_loss_classes"] = lr["valid_recovery_classes"] = ""
        loss_recovery.append(lr)

        b1 = int((selected & state["b1"]).sum())
        b2 = int((selected & state["b2"]).sum())
        if b1 + b2 != b:
            raise AssertionError(f"{protocol} {scope} {grouping} {group_id}: B1+B2 mismatch")
        br = {"protocol": protocol, "scope": scope, "grouping": grouping, "group_id": group_id,
              "n_test": n, "type_b_n": b, "restored_b1_n": b1, "still_wrong_b2_n": b2}
        if macro:
            br["restored_b1_share"], valid = _macro_rate(state["b1"], state["type_b"], labels, selected)
            br["valid_type_b_classes"] = valid
            br["restored_b1_all_test_burden_pp"] = _macro_burden(state["b1"], labels, selected)
            br["still_wrong_b2_all_test_burden_pp"] = _macro_burden(state["b2"], labels, selected)
        else:
            br["restored_b1_share"] = _ratio(b1, b)
            br["valid_type_b_classes"] = ""
            br["restored_b1_all_test_burden_pp"] = 100 * b1 / n if n else None
            br["still_wrong_b2_all_test_burden_pp"] = 100 * b2 / n if n else None
        br["still_wrong_b2_share"] = (1 - br["restored_b1_share"]
                                       if br["restored_b1_share"] is not None else None)
        restriction.append(br)
    return {"composition": composition, "loss_recovery": loss_recovery,
            "restriction": restriction}
