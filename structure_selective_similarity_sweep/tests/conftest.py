from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT.parent / "structure_multicenter_separation",
    ROOT.parent / "structure_e3_followup",
    ROOT.parent / "structure_similarity_aware_followup",
    ROOT,
):
    sys.path.insert(0, str(path))
