"""Synthetic CPU-only report smoke test; no real experiment is run."""
import json
import tempfile
from pathlib import Path

import numpy as np

from analysis import make_tables
from report import create_report


def _scores(predicted):
    values = np.full((len(predicted), 4), -1.0, np.float32)
    values[np.arange(len(predicted)), predicted] = 1.0
    return values


def test_five_plots_and_report_render():
    labels = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], np.int64)
    ids = np.asarray([f"toy:{i}" for i in range(len(labels))])
    baseline = _scores([2, 0, 1, 1, 2, 2, 3, 3])
    intervention = _scores([0, 0, 1, 0, 2, 2, 3, 3])
    b1 = np.asarray([True, False, False, False, False, False, False, False])
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = {"physical_gpu": 0, "protocols": {}}
        for protocol in ("IA10", "CAR10"):
            folder = root / protocol
            data = make_tables(protocol, ids, labels, baseline, intervention, baseline,
                               b1, [2, 2], folder)
            data["audit"] = {
                "original_training_task_summary_correct_n": 7,
                "original_training_task_summary_accuracy": 0.875,
                "cached_vanilla_vs_original_training_accuracy_delta_pp": 0.0,
                "trajectory_baseline_vs_cached_final_head_prediction_mismatch_n": 0,
                "matched_control_vs_vanilla_cached_accuracy_delta_pp": 0.0,
                "matched_control_vs_original_final_head_weight_max_abs": 0.0,
                "ce_identity_synthetic_max_abs": 0.0,
                "ce_identity_first_real_max_abs": 0.0,
                "optimizer_steps": {"control": 20, "intervention": 20},
                "original_ca_mean_scale_task_size_argument": 2,
            }
            (folder / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            manifest["protocols"][protocol] = {
                "status": "complete", "source_experiment_id": "Structure_AUDIT",
                "source_training_git_commit": None,
                "source_snapshot_hash_at_training": "toy",
                "task_partition": [2, 2],
                "source_paths": {"checkpoint": "toy.pt"},
                "source_sha256": {"checkpoint": "toyhash"},
            }
        create_report(root, manifest)
        assert (root / "final_report.md").is_file()
        assert len(list((root / "cross_protocol/plots").glob("*.png"))) == 5
        assert (root / "cross_protocol/summary.csv").is_file()
        assert (root / "cross_protocol/decision_table.csv").is_file()
