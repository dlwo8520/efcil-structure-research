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
        print(generate_protocol_report(manifest.parent))
    print(generate_combined_report(root))


if __name__ == "__main__":
    main()
