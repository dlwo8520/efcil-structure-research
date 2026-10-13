"""Filesystem-backed audit rules for the reduced five-task experiment."""

from __future__ import annotations
from pathlib import Path as _ResearchPath
import os as _research_os
_RESEARCH_ROOT = _ResearchPath(_research_os.environ.get("RESEARCH_WORKSPACE", str(_ResearchPath(__file__).resolve().parents[2])))

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .registry import (
    PROTOCOL_ORDER,
    REDUCED_PROTOCOLS,
    REDUCED_RUN_IDS,
    method_registry,
    resolve_method,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_DATA_ROOTS = {
    "IA5": Path(("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
    "IA10": Path(("${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
    "CAR5": Path(("${RESEARCH_WORKSPACE}/cds-main/data/cars196").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
    "CAR10": Path(("${RESEARCH_WORKSPACE}/cds-main/data/cars196").replace("${RESEARCH_WORKSPACE}", str(_RESEARCH_ROOT))),
}
IA5_STRUCTURE_DIR = (
    ROOT.parent
    / "structure_e3_followup"
    / "outputs"
    / "imageneta"
    / "B0_STRUCTURE"
    / "seed_1993"
)
EXPECTED_TASKS = {"IA5": 5, "IA10": 10, "CAR5": 5, "CAR10": 10}


@dataclass(frozen=True)
class AuditRecord:
    protocol: str
    run_id: str
    method: str
    top_p: float | None
    alpha: float | None
    lambda_clad: float | None
    current_status: str
    action: str
    resume_from: str
    reason: str
    command: str
    output_dir: str
    selected: bool
    legacy_grid_run: bool


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def metrics_index(output_root: Path) -> dict[tuple[str, str, int], dict]:
    path = output_root / "metrics_per_seed.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        try:
            key = (row["protocol"], row["run_id"], int(row["seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        result[key] = row
    return result


def _task_final_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        result = set()
        for row in rows:
            if row.get("eval_stage") != "task_final":
                continue
            try:
                result.add(int(row["task_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return result


def classify_run(
    run_dir: Path,
    protocol: str,
    run_id: str,
    seed: int,
    metrics: dict[tuple[str, str, int], dict],
    *,
    external: bool = False,
) -> tuple[str, str]:
    expected = EXPECTED_TASKS[protocol]
    summary = read_json(run_dir / "summary.json")
    metadata = read_json(run_dir / "run_metadata.json")
    resolved = read_json(run_dir / "resolved_config.json")
    checkpoint = run_dir / "checkpoint_latest.pt"
    final_ids = _task_final_ids(run_dir / "task_summary.csv")
    metric = metrics.get((protocol, run_id, seed), {})

    summary_complete = (
        summary.get("status") == "complete"
        and int(summary.get("completed_tasks", 0)) == expected
        and int(summary.get("expected_tasks", expected)) == expected
    )
    metadata_complete = (
        metadata.get("status") == "complete"
        and int(metadata.get("completed_tasks", 0)) == expected
    )
    finals_complete = final_ids == set(range(1, expected + 1))
    metrics_complete = external or (
        metric.get("status") == "complete"
        and int(metric.get("completed_tasks", 0)) == expected
    )
    if (
        summary_complete
        and metadata_complete
        and bool(resolved)
        and checkpoint.is_file()
        and finals_complete
    ):
        evidence = (
            f"summary+metadata complete; task_final {expected}/{expected}; "
            f"checkpoint present; metrics "
            f"{'external-not-required' if external else ('present' if metrics_complete else 'missing/stale')}"
        )
        return "COMPLETE", evidence

    failed_marker = (run_dir / "FAILED.json").is_file()
    if metadata.get("status") == "failed" or failed_marker:
        task = metadata.get("last_recorded_task", metadata.get("completed_tasks", 0))
        epoch = metadata.get("last_recorded_epoch", "-")
        return "FAILED", f"failure metadata/marker; last task={task}, epoch={epoch}"

    try:
        has_artifacts = run_dir.is_dir() and any(run_dir.iterdir())
    except OSError:
        has_artifacts = False
    if has_artifacts:
        manifest = read_json(run_dir / "checkpoint_manifest.json")
        state = manifest.get("checkpoint_state_id", metadata.get("last_checkpoint_state_id", "unknown"))
        return "PARTIAL", f"artifacts preserved; latest checkpoint state={state}"
    return "NOT_STARTED", "no run artifacts found"


def _output_dir(output_root: Path, protocol: str, run_id: str, seed: int) -> tuple[Path, bool]:
    if protocol == "IA5" and run_id == "Structure" and seed == 1993:
        return IA5_STRUCTURE_DIR, True
    return output_root / protocol / run_id / f"seed_{seed}", False


def _execution_command(
    protocol: str,
    run_id: str,
    seed: int,
    data_roots: dict[str, Path],
    output_root: Path,
    resume_from: str = "",
) -> str:
    values = [
        "python", "run.py", "--protocol", protocol, "--run-id", run_id,
        "--gpu", "${GPU_ID}", "--data-root", str(data_roots[protocol]),
        "--output-root", str(output_root), "--seed", str(seed),
    ]
    if resume_from:
        values.extend(("--resume", resume_from))
    return " ".join(values)


def audit_all(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    seed: int = 1993,
    data_roots: dict[str, Path] | None = None,
) -> list[AuditRecord]:
    output_root = Path(output_root).expanduser().resolve()
    roots = {
        key: Path(value).expanduser().resolve()
        for key, value in (data_roots or DEFAULT_DATA_ROOTS).items()
    }
    metrics = metrics_index(output_root)
    legacy = method_registry()
    all_run_ids = ("Structure", *legacy.keys())
    records = []
    for protocol in PROTOCOL_ORDER:
        for run_id in all_run_ids:
            config = resolve_method(run_id)
            run_dir, external = _output_dir(output_root, protocol, run_id, seed)
            status, evidence = classify_run(
                run_dir, protocol, run_id, seed, metrics, external=external
            )
            selected = protocol in REDUCED_PROTOCOLS and run_id in REDUCED_RUN_IDS
            checkpoint = run_dir / "checkpoint_latest.pt"
            resume_from = ""
            if protocol not in REDUCED_PROTOCOLS:
                action = "SKIP"
                reason = f"10-task HOLD; {evidence}; artifacts remain untouched"
            elif run_id not in REDUCED_RUN_IDS:
                action = "SKIP"
                reason = f"not selected for reduced Stage 1; {evidence}"
            elif status == "COMPLETE":
                action = "KEEP_COMPLETE"
                reason = f"selected and already complete; {evidence}"
            elif status == "PARTIAL" and checkpoint.is_file():
                action = "RESUME"
                resume_from = str(checkpoint.resolve())
                reason = f"selected partial run; resume latest validated checkpoint; {evidence}"
            elif (
                status == "FAILED"
                and protocol == "CAR5"
                and run_id == "Structure"
                and checkpoint.is_file()
            ):
                action = "RESUME"
                resume_from = str(checkpoint.resolve())
                reason = (
                    "selected failure has a preserved checkpoint; root cause was diagnosed "
                    f"and fixed before resume; {evidence}"
                )
            elif status == "FAILED":
                action = "SKIP"
                reason = f"selected failure has no resumable checkpoint; {evidence}"
            else:
                action = "RUN_NEW"
                reason = f"selected run has no completed result; {evidence}"
            command = ""
            if action in {"RUN_NEW", "RESUME"}:
                command = _execution_command(
                    protocol, run_id, seed, roots, output_root, resume_from
                )
            records.append(
                AuditRecord(
                    protocol=protocol,
                    run_id=run_id,
                    method=str(config["method"]),
                    top_p=config.get("top_p"),
                    alpha=config.get("alpha"),
                    lambda_clad=config.get("lambda_clad"),
                    current_status=status,
                    action=action,
                    resume_from=resume_from,
                    reason=reason,
                    command=command,
                    output_dir=str(run_dir.resolve()),
                    selected=selected,
                    legacy_grid_run=run_id in legacy,
                )
            )
    return records
