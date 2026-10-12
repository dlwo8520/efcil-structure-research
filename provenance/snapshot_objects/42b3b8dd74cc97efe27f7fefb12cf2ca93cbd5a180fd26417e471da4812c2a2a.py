#!/usr/bin/env python3
"""Dependency-free runner for the small assertion-based unit tests."""

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

from tests.test_losses import (
    test_all_three_losses_are_finite_and_differentiable,
    test_c0_is_exact_inherited_am_drift,
    test_top_p_is_per_class_exact_and_tie_stable,
)
from tests.test_registry_protocol import (
    test_exact_partitions_cover_every_class,
    test_partition_hash_depends_on_boundaries,
    test_stage1_registry_is_exactly_27,
)


def main():
    tests = (
        test_stage1_registry_is_exactly_27,
        test_exact_partitions_cover_every_class,
        test_partition_hash_depends_on_boundaries,
        test_top_p_is_per_class_exact_and_tie_stable,
        test_all_three_losses_are_finite_and_differentiable,
        test_c0_is_exact_inherited_am_drift,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"UNIT PASS ({len(tests)} tests)")


if __name__ == "__main__":
    main()
