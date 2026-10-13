"""Filesystem-backed audit for RP runs and their fixed parent baselines."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import csv
import json
from pathlib import Path

from .registry import ROOT, SEED, definitions


RP_OUTPUT_ROOT = ROOT / "outputs"
SELECTIVE_ROOT = ROOT.parent / "structure_selective_similarity_sweep"
SELECTIVE_OUTPUT_ROOT = SELECTIVE_ROOT / "outputs"
IA5_STRUCTURE_DIR = (
    ROOT.parent / "structure_e3_followup" / "outputs" / "imageneta" / "B0_STRUCTURE" / "seed_1993"
)
DATA_ROOTS = {
    "IA5": Path(("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
    "CAR5": Path(("${RESEARCH_WORKSPACE}/cds-main/data/cars196").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
}


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _task_final_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["task_id"])
            for row in rows
            if row.get("eval_stage") == "task_final" and row.get("task_id", "").isdigit()
        }


def classify(run_dir: Path, expected_tasks: int = 5) -> tuple[str, str]:
    run_dir = Path(run_dir)
    summary = _json(run_dir / "summary.json")
    metadata = _json(run_dir / "run_metadata.json")
    resolved = _json(run_dir / "resolved_config.json")
    finals = _task_final_ids(run_dir / "task_summary.csv")
    checkpoint = run_dir / "checkpoint_latest.pt"
    if (
        summary.get("status") == "complete"
        and int(summary.get("completed_tasks", 0)) == expected_tasks
        and metadata.get("status") == "complete"
        and int(metadata.get("completed_tasks", 0)) == expected_tasks
        and finals == set(range(1, expected_tasks + 1))
        and bool(resolved)
        and checkpoint.is_file()
    ):
        return "COMPLETE", "summary+metadata complete; task_final 5/5; checkpoint present"
    if metadata.get("status") == "failed" or (run_dir / "FAILED.json").is_file():
        return (
            "FAILED",
            f"failure marker; last task={metadata.get('last_recorded_task', '-')} "
            f"epoch={metadata.get('last_recorded_epoch', '-')}",
        )
    try:
        has_artifacts = run_dir.is_dir() and any(run_dir.iterdir())
    except OSError:
        has_artifacts = False
    if has_artifacts:
        manifest = _json(run_dir / "checkpoint_manifest.json")
        state = manifest.get("checkpoint_state_id", metadata.get("last_checkpoint_state_id", "unknown"))
        return "PARTIAL", f"artifacts preserved; latest checkpoint state={state}"
    return "NOT_STARTED", "no run artifacts"


def baseline_dir(protocol: str, run_id: str, selective_output_root=SELECTIVE_OUTPUT_ROOT) -> Path:
    if protocol == "IA5" and run_id == "Structure":
        return IA5_STRUCTURE_DIR
    return Path(selective_output_root) / protocol / run_id / f"seed_{SEED}"


def audit_baselines(selective_output_root=SELECTIVE_OUTPUT_ROOT) -> list[dict]:
    rows = []
    for protocol in ("IA5", "CAR5"):
        for run_id in ("Structure", "C_P020_L030", "C_P020_L100"):
            run_dir = baseline_dir(protocol, run_id, selective_output_root)
            status, evidence = classify(run_dir)
            rows.append(
                {
                    "protocol": protocol,
                    "run_id": run_id,
                    "status": status,
                    "evidence": evidence,
                    "output_dir": str(run_dir.resolve()),
                }
            )
    return rows


def audit_rp(output_root=RP_OUTPUT_ROOT, selective_output_root=SELECTIVE_OUTPUT_ROOT) -> list[dict]:
    output_root = Path(output_root).expanduser().resolve()
    baselines = {
        (row["protocol"], row["run_id"]): row for row in audit_baselines(selective_output_root)
    }
    rows = []
    for spec in definitions():
        protocol, run_id = spec["protocol"], spec["run_id"]
        run_dir = output_root / protocol / run_id / f"seed_{SEED}"
        status, evidence = classify(run_dir)
        required = [
            baselines[(protocol, "Structure")],
            baselines[(protocol, spec["parent_run_id"])],
        ]
        missing = [row for row in required if row["status"] != "COMPLETE"]
        resume = ""
        if status == "COMPLETE":
            action = "KEEP_COMPLETE"
            reason = evidence
        elif status == "FAILED":
            action = "SKIP_FAILED"
            reason = f"manual diagnosis required; {evidence}"
        elif missing:
            action = "WAIT_PARENT"
            reason = "; ".join(f"{row['run_id']}={row['status']}" for row in missing)
        elif status == "PARTIAL" and (run_dir / "checkpoint_latest.pt").is_file():
            action = "RESUME"
            resume = str((run_dir / "checkpoint_latest.pt").resolve())
            reason = evidence
        else:
            action = "RUN_NEW"
            reason = evidence
        command = (
            f"python run.py --config {spec['config']} --gpu ${{GPU_ID}} "
            f"--data-root {DATA_ROOTS[protocol]} --output-root {output_root} --seed {SEED}"
        )
        if resume:
            command += f" --resume {resume}"
        rows.append(
            {
                **spec,
                "status": status,
                "action": action,
                "resume_from": resume,
                "reason": reason,
                "command": command,
                "output_dir": str(run_dir),
                "parent_status": baselines[(protocol, spec["parent_run_id"])]["status"],
                "structure_status": baselines[(protocol, "Structure")]["status"],
            }
        )
    return rows

