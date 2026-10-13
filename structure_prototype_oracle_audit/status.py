#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from oracle_audit.config import PROTOCOLS, ROOT


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    print("protocol  training  snapshots  audit_tasks  report")
    for protocol in PROTOCOLS:
        train = ROOT / "training_outputs" / protocol / "Structure_AUDIT" / "seed_1993"
        summary = read(train / "summary.json")
        snapshots = len(list((train / "task_checkpoints").glob("task_*.pt")))
        audit = ROOT / "outputs" / protocol / "seed_1993"
        manifest = read(audit / "audit_manifest.json")
        report = (audit / "structure_prototype_oracle_audit.md").is_file()
        print(
            f"{protocol:<8} {summary.get('status', 'missing'):<9} {snapshots:<9} "
            f"{str(manifest.get('audited_tasks', [])):<20} {'yes' if report else 'no'}"
        )


if __name__ == "__main__":
    main()
