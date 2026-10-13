#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

from oracle_audit.config import ROOT


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    root = ROOT / "scientific_audit"
    status = read(root / "queue_status.json")
    pid_path = ROOT / "logs" / "scientific_gpu1.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    alive = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            pass
    print(
        f"queue={status.get('status', 'NOT_STARTED')} pid={pid or '-'} "
        f"alive={alive} gpu=1 protocol={status.get('current_protocol')} "
        f"stage={status.get('current_stage')} task={status.get('current_task')}"
    )
    print("protocol  structure          audit              A-D       visualization  report    status")
    for protocol in ("IA5", "CAR5", "IA10", "CAR10"):
        row = status.get("protocols", {}).get(protocol, {})
        print(
            f"{protocol:<8} {str(row.get('structure', 'PENDING')):<18} "
            f"{str(row.get('audit', 'PENDING')):<18} {str(row.get('a_d', 'PENDING')):<9} "
            f"{str(row.get('visualization', 'PENDING')):<14} "
            f"{str(row.get('report', 'PENDING')):<9} {row.get('status', 'PENDING')}"
        )
    if status.get("failures"):
        print("failures:")
        for failure in status["failures"]:
            print(
                f"- {failure.get('protocol')} {failure.get('stage')}: "
                f"{failure.get('type')}: {failure.get('reason')}"
            )


if __name__ == "__main__":
    main()
