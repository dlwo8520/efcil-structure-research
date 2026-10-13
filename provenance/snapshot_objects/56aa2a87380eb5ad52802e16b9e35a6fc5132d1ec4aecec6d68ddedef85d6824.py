#!/usr/bin/env python3
"""Fast, non-training smoke checks for all configs, losses and real data manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "structure_multicenter_separation"
E3 = ROOT.parent / "structure_e3_followup"
SIMILARITY = ROOT.parent / "structure_similarity_aware_followup"
for path in (BASE, E3, SIMILARITY, ROOT):
    sys.path.insert(0, str(path))

from run import load_config, merge, validate_config
from selective.protocol import SweepDataManager
from selective.registry import PROTOCOL_ORDER, method_registry, protocol_config_path
from similarity.run_compat import validate_followup_compat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imageneta-root", default="${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a")
    parser.add_argument("--cars-root", default="${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196")
    parser.add_argument("--skip-data", action="store_true")
    cli = parser.parse_args()
    resolved = {}
    for protocol in PROTOCOL_ORDER:
        base = load_config(protocol_config_path(protocol))
        for run_id, method in method_registry().items():
            config = merge(base, method)
            validate_config(config)
            validate_followup_compat(config)
        resolved[protocol] = base
    print("PASS config validation: 4 protocols x 27 settings = 108 runs")

    if not cli.skip_data:
        roots = {
            "IA5": cli.imageneta_root, "IA10": cli.imageneta_root,
            "CAR5": cli.cars_root, "CAR10": cli.cars_root,
        }
        hashes = {}
        for protocol in PROTOCOL_ORDER:
            config = dict(resolved[protocol])
            config.update({"data_root": roots[protocol], "seed": 1993, "verify_split_content": False})
            manager = SweepDataManager(
                protocol, config["shuffle"], 1993, config["init_cls"], config["increment"], args=config
            )
            assert manager.nb_tasks == config["expected_tasks"]
            assert sum(manager._increments) == config["expected_num_classes"]
            hashes[protocol] = manager.protocol_manifest["class_order_hash"]
            print(
                f"PASS {protocol}: classes={manager.get_total_classnum()} tasks={manager._increments} "
                f"train={len(manager._train_targets)} test={len(manager._test_targets)}"
            )
        assert hashes["IA5"] == hashes["IA10"]
        assert hashes["CAR5"] == hashes["CAR10"]
        print("PASS shared class order: IA5=IA10 and CAR5=CAR10")
        prior_protocol = (
            ROOT.parent
            / "pcds_structure_sgds_official/outputs/cars196_10task_official_structure/seed_0/pcds_structure/protocol.json"
        )
        if prior_protocol.is_file():
            prior = json.loads(prior_protocol.read_text(encoding="utf-8"))["class_order"]
            car_config = dict(resolved["CAR10"])
            car_config.update({"data_root": roots["CAR10"], "seed": 1993, "verify_split_content": False})
            car_manager = SweepDataManager(
                "CAR10", car_config["shuffle"], 1993,
                car_config["init_cls"], car_config["increment"], args=car_config,
            )
            assert car_manager._class_order == prior
            print("PASS Cars-196 order matches the existing fixed RandomState(1993) protocol")
    print("SMOKE PASS (no model training was launched)")


if __name__ == "__main__":
    main()
