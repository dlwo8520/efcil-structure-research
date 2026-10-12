#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oracle_audit.config import PROTOCOLS, protocol_config, validate_protocol_config
from oracle_audit.metrics import covariance_metrics, mahalanobis_squared_diagonal, vector_error


def main() -> None:
    value = np.asarray([1.0, 2.0, 3.0])
    assert np.isclose(vector_error(value, value)["relative_l2"], 0.0)
    covariance = np.diag([1.0, 4.0])
    assert np.isclose(covariance_metrics(covariance, covariance)["frobenius_relative_error"], 0.0)
    assert np.allclose(
        mahalanobis_squared_diagonal(np.asarray([[1.0, 2.0]]), np.zeros(2), covariance),
        [2.0],
    )
    for protocol in PROTOCOLS:
        config = protocol_config(protocol)
        validate_protocol_config(config)
        assert sum(config["task_partition"]) == config["expected_num_classes"]
    print("PASS metric primitives")
    print("PASS IA5/IA10/CAR5/CAR10 vanilla Structure config validation")


if __name__ == "__main__":
    main()
