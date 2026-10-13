#!/usr/bin/env python3
"""Dependency-free assertion test runner for the RP extension."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT.parent / "structure_multicenter_separation",
    ROOT.parent / "structure_e3_followup",
    ROOT.parent / "structure_similarity_aware_followup",
    ROOT.parent / "structure_selective_similarity_sweep",
    ROOT,
):
    sys.path.insert(0, str(path))

from tests.test_rp import (
    test_projection_is_frozen_reproducible_and_seed_sensitive,
    test_relu_is_before_normalization_and_has_no_bias,
    test_rp_proto_clad_uses_original_selection_and_rp_loss_only,
)


def main():
    tests = (
        test_projection_is_frozen_reproducible_and_seed_sensitive,
        test_relu_is_before_normalization_and_has_no_bias,
        test_rp_proto_clad_uses_original_selection_and_rp_loss_only,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"UNIT PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()

