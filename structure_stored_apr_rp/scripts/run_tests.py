#!/usr/bin/env python3
"""Small CPU-only sanity plus unchanged parent RP regression suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for name in (
    "structure_multicenter_separation", "structure_e3_followup",
    "structure_similarity_aware_followup", "structure_selective_similarity_sweep",
    "structure_rp_protoclad_task5", "structure_stored_apr_rp",
):
    sys.path.insert(0, str(ROOT.parent / name))

from tests.test_stored import (
    test_birth_is_samplewise_projected_mean_not_projected_mean,
    test_zero_and_nonzero_apr_drift_use_exact_formula,
    test_original_r_checksum_and_frozen_regression,
    test_loss_matches_old_rp_when_given_oldstyle_and_has_same_selection,
    test_first_task_birth_hook_stores_class_level_statistic,
    test_checkpoint_restores_identical_birth_class_statistic,
)


def main():
    tests = (
        test_birth_is_samplewise_projected_mean_not_projected_mean,
        test_zero_and_nonzero_apr_drift_use_exact_formula,
        test_original_r_checksum_and_frozen_regression,
        test_loss_matches_old_rp_when_given_oldstyle_and_has_same_selection,
        test_first_task_birth_hook_stores_class_level_statistic,
        test_checkpoint_restores_identical_birth_class_statistic,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}", flush=True)
    parent = ROOT.parent / "structure_rp_protoclad_task5"
    subprocess.run([sys.executable, str(parent / "scripts" / "run_tests.py")], cwd=parent, check=True)
    for protocol in ("IA5", "CAR5"):
        subprocess.run(
            [sys.executable, str(ROOT / "run.py"), "--protocol", protocol,
             "--gpu", "2", "--validate-only"],
            cwd=ROOT, check=True,
        )
    print("STOREDAPR SANITY PASS: 6 new + 3 unchanged RP + 2 config checks", flush=True)


if __name__ == "__main__":
    main()
