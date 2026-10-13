"""The intentionally fixed four-run RP-ProtoCLAD registry."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("IA5", "CAR5")
RUN_IDS = ("RP_C_P020_L030", "RP_C_P020_L100")
SEED = 1993
RP_DIM = 2048
RP_SEED = 1993
RP_INPUT_DIM = 768


def definitions() -> tuple[dict, ...]:
    rows = []
    for protocol in PROTOCOLS:
        for run_id, parent, coefficient in (
            ("RP_C_P020_L030", "C_P020_L030", 0.30),
            ("RP_C_P020_L100", "C_P020_L100", 1.00),
        ):
            rows.append(
                {
                    "protocol": protocol,
                    "run_id": run_id,
                    "parent_run_id": parent,
                    "lambda_clad": coefficient,
                    "config": ROOT / "configs" / f"{run_id}_{protocol}.json",
                }
            )
    return tuple(rows)


def definition(protocol: str, run_id: str) -> dict:
    for row in definitions():
        if row["protocol"] == protocol and row["run_id"] == run_id:
            return dict(row)
    raise ValueError(f"unsupported RP run: {protocol}/{run_id}")

