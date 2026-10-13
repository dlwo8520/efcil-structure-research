#!/usr/bin/env python3
"""Small CPU-only RP loss/diagnostic smoke test; it does not train a model."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_tests.py")], cwd=ROOT, check=True
    )
    assert result.returncode == 0
    validations = []
    for config in sorted((ROOT / "configs").glob("RP_*.json")):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "run.py"), "--config", str(config), "--validate-only"],
            cwd=ROOT,
            text=True,
        ).strip()
        validations.append(output)
    with tempfile.TemporaryDirectory(prefix="rp_protoclad_smoke_") as temporary:
        record = Path(temporary) / "smoke.json"
        record.write_text(json.dumps({"validations": validations}, indent=2) + "\n", encoding="utf-8")
        assert len(json.loads(record.read_text(encoding="utf-8"))["validations"]) == 4
    print("SMOKE PASS: 3 loss/projection tests + 4 fixed config validations; no training launched")


if __name__ == "__main__":
    main()
