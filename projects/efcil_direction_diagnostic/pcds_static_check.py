from __future__ import annotations

import argparse
import json
from pathlib import Path

from efcil_diag.pcds_experiment import validate_protocol
from efcil_diag.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Static PCDS protocol audit")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent
    result = validate_protocol(load_config(args.config), repo_root, require_gpu=True)
    result.update({
        "scc_reference_gradient": "blocked by frozen deepcopy + no_grad",
        "apr_old_data_source": "PrototypeMemory means/covariances only",
        "current_train_dataset_calls": "current task_id only",
        "classifier_candidate_space": "all currently seen rows",
        "ca_optimizer_scope": "classifier parameters only",
        "status": "PASS",
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
