"""Small, dependency-free helpers for reproducible experiment state and logs."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: os.PathLike[str] | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def append_jsonl(path: os.PathLike[str] | str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: os.PathLike[str] | str, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        write_json(target.with_suffix(".json"), [])
        return
    fieldnames = list(materialized[0].keys())
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: json_ready(value) for key, value in row.items()} for row in materialized])
    os.replace(temporary, target)


def read_csv(path: os.PathLike[str] | str) -> list[dict[str, str]]:
    target = Path(path)
    if not target.is_file():
        return []
    with target.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def upsert_csv_rows(
    path: os.PathLike[str] | str,
    rows: Iterable[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> None:
    """Atomically insert/replace rows by a stable stringified compound key."""
    incoming = [dict(row) for row in rows]
    if not incoming:
        return
    existing = read_csv(path)

    def key(row):
        return tuple(str(row.get(field, "")) for field in key_fields)

    def sort_value(value):
        text = str(value)
        try:
            return (0, int(text))
        except ValueError:
            return (1, text)

    merged = {key(row): row for row in existing}
    for row in incoming:
        merged[key(row)] = row
    fieldnames = []
    for row in [*existing, *incoming]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    ordered = sorted(
        merged.values(),
        key=lambda row: tuple(sort_value(row.get(field, "")) for field in key_fields),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    json.dumps(json_ready(row.get(field)), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else json_ready(row.get(field))
                )
                for field in fieldnames
            }
            for row in ordered
        )
    os.replace(temporary, target)


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def update_json(path: os.PathLike[str] | str, updates: Mapping[str, Any]) -> Dict[str, Any]:
    target = Path(path)
    current: Dict[str, Any] = {}
    if target.is_file():
        with target.open("r", encoding="utf-8") as handle:
            current = json.load(handle)
    current.update(json_ready(dict(updates)))
    write_json(target, current)
    return current
