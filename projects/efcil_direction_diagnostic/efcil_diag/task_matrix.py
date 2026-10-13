from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CILMatrixSummary:
    transfer: None
    average: float
    last: float
    row_averages: list[float]


def parse_lower_triangular_matrix(
    rows: list[dict[str, str]],
    num_tasks: int,
) -> list[list[float]]:
    """Parse the saved wide matrix and reject invented future-task values."""
    if len(rows) != num_tasks:
        raise ValueError(f"Expected {num_tasks} matrix rows, found {len(rows)}")
    matrix: list[list[float]] = []
    for expected_stage, row in enumerate(rows):
        stage = int(row["eval_task"])
        if stage != expected_stage:
            raise ValueError(f"Expected stage {expected_stage}, found {stage}")
        current: list[float] = []
        for task_id in range(num_tasks):
            value = row.get(f"task_{task_id}", "").strip()
            if task_id <= stage:
                if value == "":
                    raise ValueError(f"Missing A[{stage},{task_id}]")
                accuracy = float(value)
                if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                    raise ValueError(f"Invalid A[{stage},{task_id}]={accuracy}")
                current.append(accuracy)
            elif value != "":
                raise ValueError(
                    f"Future-task A[{stage},{task_id}] must be empty, found {value}"
                )
        matrix.append(current)
    return matrix


def summarize_cil_matrix(matrix: list[list[float]]) -> CILMatrixSummary:
    if not matrix:
        raise ValueError("Accuracy matrix is empty")
    for stage, row in enumerate(matrix):
        if len(row) != stage + 1:
            raise ValueError("Accuracy matrix must be lower triangular")
    row_averages = [sum(row) / len(row) for row in matrix]
    return CILMatrixSummary(
        transfer=None,
        average=sum(row_averages) / len(row_averages),
        last=row_averages[-1],
        row_averages=row_averages,
    )


def long_format_rows(
    method: str,
    seed: int,
    matrix: list[list[float]],
) -> list[dict]:
    """Return learned cells in percentage units, matching paper-style tables."""
    summary = summarize_cil_matrix(matrix)
    result: list[dict] = []
    for stage, row in enumerate(matrix):
        for evaluated_task, accuracy in enumerate(row):
            result.append({
                "method": method,
                "seed": seed,
                "trained_through_task": stage,
                "evaluated_task": evaluated_task,
                "accuracy": round(100.0 * accuracy, 10),
                "seen_average": round(100.0 * summary.row_averages[stage], 10),
            })
    return result
