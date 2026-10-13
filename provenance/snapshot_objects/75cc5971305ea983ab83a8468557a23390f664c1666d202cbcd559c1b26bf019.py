"""Canonical protocol and Stage-1 method registry.

Run IDs and grids are kept here so the launcher, queue generator, tests and
report code cannot silently disagree.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ORDER = ("IA5", "IA10", "CAR5", "CAR10")


def protocol_config_path(protocol: str) -> Path:
    protocol = protocol.upper()
    if protocol not in PROTOCOL_ORDER:
        raise ValueError(f"unknown protocol {protocol!r}; choose {PROTOCOL_ORDER}")
    return ROOT / "configs" / "protocols" / f"{protocol}.json"


def _pcode(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def method_registry() -> dict[str, dict]:
    """Return all 27 Stage-1 settings in their stable execution order."""
    rows: list[tuple[str, dict]] = [
        (
            "C0",
            {
                "method": "C0",
                "method_name": "AM-Drift baseline",
                "selective_method": "baseline_am_drift",
                "top_p": None,
                "alpha": None,
                "lambda_clad": None,
                "sep_lambda": 0.3,
                "similarity_relation_mode": "legacy_absolute",
            },
        )
    ]
    for p in (0.05, 0.10, 0.15, 0.20, 0.25):
        rows.append(
            (
                f"A_P{_pcode(p)}",
                {
                    "method": "A",
                    "method_name": "Selective AM-Drift",
                    "selective_method": "selective_am_drift",
                    "top_p": p,
                    "alpha": None,
                    "lambda_clad": None,
                    "sep_lambda": 0.3,
                    "similarity_relation_mode": "selective_am",
                },
            )
        )
    for p in (0.05, 0.10, 0.20):
        for alpha in (0.25, 0.50, 1.00):
            rows.append(
                (
                    f"B_P{_pcode(p)}_A{int(round(alpha * 100)):03d}",
                    {
                        "method": "B",
                        "method_name": "Similarity-Enlarged Boundary",
                        "selective_method": "enlarged_boundary",
                        "top_p": p,
                        "alpha": alpha,
                        "lambda_clad": None,
                        "sep_lambda": 0.3,
                        "similarity_relation_mode": "enlarged_boundary",
                    },
                )
            )
    for p in (0.05, 0.10, 0.20):
        for coefficient in (0.10, 0.30, 1.00, 3.00):
            rows.append(
                (
                    f"C_P{_pcode(p)}_L{int(round(coefficient * 100)):03d}",
                    {
                        "method": "C",
                        "method_name": "Proto-CLAD",
                        "selective_method": "proto_clad",
                        "top_p": p,
                        "alpha": None,
                        "lambda_clad": coefficient,
                        # The inherited training loop multiplies the custom
                        # relation loss by sep_lambda. For Method C this field
                        # is the Proto-CLAD coefficient, not AM separation.
                        "sep_lambda": coefficient,
                        "similarity_relation_mode": "proto_clad",
                    },
                )
            )
    registry = {}
    for run_id, values in rows:
        registry[run_id] = {"experiment_id": run_id, **values}
    if len(registry) != 27:
        raise AssertionError(f"Stage-1 registry must contain 27 settings, got {len(registry)}")
    return registry


def resolve_method(run_id: str) -> dict:
    try:
        return dict(method_registry()[run_id])
    except KeyError as error:
        raise ValueError(f"unknown run ID {run_id!r}") from error
