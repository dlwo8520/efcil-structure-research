#!/usr/bin/env python3
"""Freeze top-2 A/B/C settings per protocol after a complete Stage 1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(ROOT / "outputs" / "metrics_per_seed.csv"))
    parser.add_argument("--output", default=str(ROOT / "stage2_selection.json"))
    cli = parser.parse_args()
    data = [row for row in rows(cli.metrics) if int(row["seed"]) == 1993]
    selection, rationale = {}, []
    for protocol in ("IA5", "IA10", "CAR5", "CAR10"):
        protocol_rows = [row for row in data if row["protocol"] == protocol]
        if len(protocol_rows) != 27:
            raise RuntimeError(f"{protocol}: Stage 1 must have 27 complete rows, found {len(protocol_rows)}")
        selection[protocol] = {}
        for family in ("A", "B", "C"):
            candidates = [row for row in protocol_rows if row["method"] == family]
            guarded = [row for row in candidates if row.get("guardrail_pass") == "True"]
            pool = guarded if len(guarded) >= 2 else candidates
            ranked = sorted(
                pool,
                key=lambda row: (
                    number(row, "Inc-Acc") or -1e9,
                    number(row, "Last-Acc") or -1e9,
                    -(number(row, "Forgetting") or 1e9),
                ),
                reverse=True,
            )
            chosen = [row["run_id"] for row in ranked[:2]]
            selection[protocol][family] = chosen
            rationale.append(
                f"{protocol}/{family}: {', '.join(chosen)} "
                f"({'guardrail candidates' if len(guarded) >= 2 else 'fallback: fewer than two guardrail candidates'})"
            )
    Path(cli.output).write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "stage2_selection.md").write_text(
        "# Frozen Stage-2 selection\n\n" + "\n".join(f"- {line}" for line in rationale) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {cli.output}")


if __name__ == "__main__":
    main()
