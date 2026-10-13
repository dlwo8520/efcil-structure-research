#!/usr/bin/env python3
"""Fast preflight checks; does not train the ViT."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.data_manager import DataManager


BASE_CONFIG = ROOT / "exps_jsons/cars196_structure_pcds_seed0.json"
SGDS_CONFIG = ROOT / "exps_jsons/cars196_structure_pcds_sgds_seed0.json"


def load(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    baseline = load(BASE_CONFIG)
    sgds = load(SGDS_CONFIG)
    allowed_differences = {
        "prefix", "method_label", "use_sgds", "sgds_target_layer", "sgds_k",
        "sgds_k_interpretation", "sgds_beta", "sgds_gamma",
        "sgds_exploration_ratio", "sgds_scope",
        "sgds_mask_seed", "sgds_rng_isolated_from_training",
    }
    all_keys = set(baseline) | set(sgds)
    unexpected = {
        key: (baseline.get(key), sgds.get(key))
        for key in all_keys
        if key not in allowed_differences and baseline.get(key) != sgds.get(key)
    }
    if unexpected:
        raise AssertionError("Unfair config differences: {}".format(unexpected))
    if baseline["use_sgds"] is not False or sgds["use_sgds"] is not True:
        raise AssertionError("The comparison must differ in SGDS activation guidance")
    if sgds["sgds_target_layer"] != 11 or sgds["sgds_k"] != 0.6:
        raise AssertionError("Expected final block and 60% Top-K retention")

    manager = DataManager(
        baseline["dataset"], baseline["shuffle"], baseline["seed"][0],
        baseline["init_cls"], baseline["increment"],
        data_root=baseline["data_root"],
        class_order_seed=baseline["class_order_seed"],
    )
    if manager._increments != [20] * 9 + [16]:
        raise AssertionError("Unexpected task split: {}".format(manager._increments))
    if len(manager._train_data) != 8144 or len(manager._test_data) != 8041:
        raise AssertionError("Unexpected official split counts")
    result = {
        "status": "PASS",
        "new_repository": str(ROOT),
        "official_structure_learner": "models/CDS_adapter.py",
        "dataset_root": baseline["data_root"],
        "official_split_images": {"train": len(manager._train_data), "test": len(manager._test_data)},
        "task_sizes": manager._increments,
        "classes": manager.get_total_classnum(),
        "training_seed": baseline["seed"][0],
        "shared_class_order_seed": baseline["class_order_seed"],
        "baseline_and_sgds_configs_match_except_guidance": True,
        "sgds_target": "final ViT block attention-adapter input only",
        "sgds_scope": "Eq. (4)-(10) only",
        "raw_replay": False,
        "single_expanding_head": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
