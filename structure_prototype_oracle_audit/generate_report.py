#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from oracle_audit.config import ROOT
from oracle_audit.report import generate_combined_report, generate_protocol_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    root = Path(args.audit_root).expanduser().resolve()
    for manifest in sorted(root.glob("*/seed_*/audit_manifest.json")):
        report = generate_protocol_report(manifest.parent)
        protocol = manifest.parent.parent.name
        named = manifest.parent / f"{protocol}_structure_oracle_report.md"
        named.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
        print(named)
    combined = generate_combined_report(root)
    named_combined = root / "structure_oracle_cross_protocol_report.md"
    named_combined.write_text(combined.read_text(encoding="utf-8"), encoding="utf-8")
    print(named_combined)


if __name__ == "__main__":
    main()
