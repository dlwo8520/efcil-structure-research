#!/usr/bin/env python3
"""GPU-1-only unattended Structure oracle scientific queue.

The queue is deliberately separate from the earlier sanity/legacy runner.  It
never treats those outputs as scientific results and never reruns a COMPLETE
training/audit.  IA5 is used as a hard preflight gate before the four protocols
are processed sequentially.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from oracle_audit.config import ROOT


PROTOCOL_ORDER = ("IA5", "CAR5", "IA10", "CAR10")
EXPECTED_TASKS = {"IA5": 5, "CAR5": 5, "IA10": 10, "CAR10": 10}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def training_dir(protocol: str) -> Path:
    return ROOT / "training_outputs" / protocol / "Structure_AUDIT" / "seed_1993"


def audit_dir(scientific_root: Path, protocol: str) -> Path:
    return scientific_root / protocol / "seed_1993"


def complete_training(protocol: str) -> bool:
    run_dir = training_dir(protocol)
    summary = read_json(run_dir / "summary.json")
    snapshots = list((run_dir / "task_checkpoints").glob("task_*.pt"))
    return bool(
        summary.get("status") == "complete"
        and int(summary.get("completed_tasks", 0)) == EXPECTED_TASKS[protocol]
        and len(snapshots) == EXPECTED_TASKS[protocol]
    )


def resume_checkpoint(protocol: str) -> Path | None:
    checkpoint = training_dir(protocol) / "checkpoint_latest.pt"
    return checkpoint if checkpoint.is_file() else None


def complete_audit(scientific_root: Path, protocol: str) -> bool:
    run_dir = audit_dir(scientific_root, protocol)
    manifest = read_json(run_dir / "audit_manifest.json")
    regression = read_json(run_dir / "ca_regression_check.json")
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("complete_temporal_coverage") is True
        and manifest.get("scientific_interpretation_allowed") is True
        and manifest.get("audited_tasks") == list(range(1, EXPECTED_TASKS[protocol] + 1))
        and regression.get("status") == "PASS"
        and (run_dir / f"{protocol}_structure_oracle_report.md").is_file()
    )


def gpu_compute_pids(gpu: str) -> list[int]:
    command = [
        "nvidia-smi",
        "-i",
        gpu,
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [int(line) for line in output.splitlines() if line.strip().isdigit()]


def preserve_partial(path: Path, scientific_root: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    relative = path.relative_to(scientific_root)
    destination = scientific_root / "_preserved_partial" / relative.parent / f"{relative.name}_{stamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    path.rename(destination)
    return destination


def symlink_once(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def materialize_state_index(scientific_root: Path, protocol: str) -> None:
    """Expose reconstructable pre/post-CA states without duplicating GB checkpoints."""
    source = training_dir(protocol)
    destination = audit_dir(scientific_root, protocol) / "task_states"
    resolved_config = source / "resolved_config.json"
    for task in range(1, EXPECTED_TASKS[protocol] + 1):
        checkpoint = source / "task_checkpoints" / f"task_{task:02d}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        task_root = destination / f"task_{task:02d}"
        pre = task_root / "pre_ca"
        post = task_root / "post_ca"
        symlink_once(post / "checkpoint.pt", checkpoint)
        symlink_once(pre / "post_ca_base_checkpoint.pt", checkpoint)
        if resolved_config.is_file():
            symlink_once(pre / "resolved_config.json", resolved_config)
            symlink_once(post / "resolved_config.json", resolved_config)
        sidecar = source / "pre_ca_classifier" / f"task_{task:02d}.pt"
        if sidecar.is_file():
            symlink_once(pre / "classifier_and_rng_delta.pt", sidecar)
            representation = (
                "pre-CA = post_ca_base_checkpoint with network fc replaced by "
                "classifier_and_rng_delta.classifier_state_dict; the delta also stores "
                "RNG and exact CA config"
            )
        else:
            representation = "task has no CA; pre-CA and post-CA classifier are identical"
        write_json(
            task_root / "state_manifest.json",
            {
                "protocol": protocol,
                "task": task,
                "storage": "read-only symlinks; no checkpoint duplication",
                "pre_ca_representation": representation,
                "post_ca_checkpoint": str(checkpoint.resolve()),
                "pre_ca_classifier_delta": str(sidecar.resolve()) if sidecar.is_file() else None,
                "statistics_in_base_checkpoint": [
                    "class_means",
                    "class_covs",
                    "task_source_means",
                    "task_source_covs",
                    "now_task_mean",
                    "now_task_covs",
                    "config",
                ],
            },
        )


class Queue:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.gpu = "1"
        self.scientific_root = Path(args.output_root).expanduser().resolve()
        self.scientific_root.mkdir(parents=True, exist_ok=True)
        self.logs_root = ROOT / "logs" / "scientific_gpu1"
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.status_path = self.scientific_root / "queue_status.json"
        previous = read_json(self.status_path, {})
        self.started_unix = time.time()
        self.status: dict[str, Any] = {
            "format_version": 1,
            "queue": "Structure Prototype Oracle scientific audit",
            "gpu": self.gpu,
            "protocol_order": list(PROTOCOL_ORDER),
            "status": "STARTING",
            "current_protocol": None,
            "current_stage": "startup",
            "current_task": None,
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "elapsed_seconds": 0.0,
            "protocols": {
                protocol: {
                    "structure": "PENDING",
                    "audit": "PENDING",
                    "a_d": "PENDING",
                    "visualization": "PENDING",
                    "report": "PENDING",
                    "status": "PENDING",
                }
                for protocol in PROTOCOL_ORDER
            },
            "failures": [],
            "resumed_runs": [],
            "skipped_complete_runs": [],
            "preserved_partial_runs": [],
            "gpu_work_history": [],
            "previous_queue_status": previous.get("status"),
        }
        self.update()

    def update(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at_utc"] = utc_now()
        self.status["elapsed_seconds"] = time.time() - self.started_unix
        write_json(self.status_path, self.status)
        self.write_markdown()

    def protocol_update(self, protocol: str, **updates: Any) -> None:
        self.status["protocols"][protocol].update(updates)
        self.update()

    def write_markdown(self) -> None:
        lines = [
            "# Holiday Structure Oracle Audit Status",
            "",
            f"Updated: `{self.status['updated_at_utc']}` · GPU: physical `{self.gpu}` · queue: **{self.status['status']}**",
            "",
            "| Protocol | Structure | Audit | A–D | Visualization | Report | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for protocol in PROTOCOL_ORDER:
            record = self.status["protocols"][protocol]
            lines.append(
                "| " + protocol + " | " + " | ".join(
                    str(record[key]) for key in ("structure", "audit", "a_d", "visualization", "report", "status")
                ) + " |"
            )
        lines.extend(
            [
                "",
                f"- total elapsed time: `{self.status['elapsed_seconds']:.1f}` seconds",
                f"- current: protocol=`{self.status.get('current_protocol')}`, stage=`{self.status.get('current_stage')}`, task=`{self.status.get('current_task')}`",
                f"- failures: `{json.dumps(self.status['failures'], ensure_ascii=False)}`",
                f"- resumed runs: `{json.dumps(self.status['resumed_runs'], ensure_ascii=False)}`",
                f"- skipped COMPLETE runs: `{json.dumps(self.status['skipped_complete_runs'], ensure_ascii=False)}`",
                f"- preserved partial runs: `{json.dumps(self.status['preserved_partial_runs'], ensure_ascii=False)}`",
                "",
                "## GPU 1 work history",
                "",
            ]
        )
        for record in self.status["gpu_work_history"]:
            lines.append(
                f"- `{record.get('started_at_utc')}` → `{record.get('ended_at_utc', 'running')}`: "
                f"{record.get('protocol')} / {record.get('stage')} / {record.get('status')}"
            )
        (self.scientific_root / "holiday_oracle_audit_status.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def wait_for_gpu(self) -> None:
        while True:
            pids = [pid for pid in gpu_compute_pids(self.gpu) if pid != os.getpid()]
            if not pids:
                return
            self.update(status="WAITING_FOR_GPU", current_stage="waiting_for_gpu", gpu_busy_pids=pids)
            time.sleep(self.args.poll_seconds)

    def watch_progress(self, protocol: str, stage: str) -> None:
        if stage == "training":
            summary = read_json(training_dir(protocol) / "summary.json")
            completed = int(summary.get("completed_tasks", 0))
            current = min(completed + 1, EXPECTED_TASKS[protocol])
        else:
            progress_root = (
                self.scientific_root / "_preflight"
                if stage == "preflight"
                else self.scientific_root
            )
            progress = read_json(audit_dir(progress_root, protocol) / "audit_progress.json")
            current = progress.get("current_task")
        self.status["current_task"] = current
        self.update()

    def run(self, protocol: str, stage: str, command: list[str], log_path: Path) -> None:
        self.wait_for_gpu()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "protocol": protocol,
            "stage": stage,
            "status": "RUNNING",
            "started_at_utc": utc_now(),
            "command": command,
            "log": str(log_path),
        }
        self.status["gpu_work_history"].append(record)
        self.update(
            status="RUNNING",
            current_protocol=protocol,
            current_stage=stage,
            current_task=None,
            gpu_busy_pids=[],
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] RUN {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                self.watch_progress(protocol, stage)
                time.sleep(self.args.poll_seconds)
            return_code = process.wait()
            log.write(f"[{utc_now()}] EXIT {return_code}\n")
        record["ended_at_utc"] = utc_now()
        record["return_code"] = return_code
        record["status"] = "COMPLETE" if return_code == 0 else "FAIL"
        self.update(current_task=None)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

    def ensure_training(self, protocol: str) -> None:
        if complete_training(protocol):
            self.status["skipped_complete_runs"].append(f"{protocol}:training")
            self.protocol_update(protocol, structure="COMPLETE (reused)")
            return
        command = [sys.executable, str(ROOT / "run_structure.py"), "--protocol", protocol, "--gpu", self.gpu]
        resume = resume_checkpoint(protocol)
        if resume is not None:
            command.extend(["--resume", str(resume)])
            self.status["resumed_runs"].append(f"{protocol}:training:{resume}")
        self.protocol_update(protocol, structure="RUNNING", status="RUNNING")
        self.run(protocol, "training", command, self.logs_root / f"{protocol}_training.log")
        if not complete_training(protocol):
            raise RuntimeError(f"{protocol} training exited but completion audit failed")
        self.protocol_update(protocol, structure="COMPLETE")

    def prepare_audit_directory(self, protocol: str) -> None:
        target = audit_dir(self.scientific_root, protocol)
        if target.exists() and not complete_audit(self.scientific_root, protocol):
            preserved = preserve_partial(target, self.scientific_root)
            self.status["preserved_partial_runs"].append(str(preserved))
        target.mkdir(parents=True, exist_ok=True)
        materialize_state_index(self.scientific_root, protocol)
        self.update()

    def run_full_audit(self, protocol: str) -> None:
        if complete_audit(self.scientific_root, protocol):
            self.status["skipped_complete_runs"].append(f"{protocol}:scientific_audit")
            self.protocol_update(
                protocol,
                audit="COMPLETE (reused)",
                a_d="PASS",
                visualization="COMPLETE",
                report="COMPLETE",
                status="COMPLETE",
            )
            return
        self.prepare_audit_directory(protocol)
        self.protocol_update(protocol, audit="RUNNING", a_d="RUNNING", status="RUNNING")
        command = [
            sys.executable,
            str(ROOT / "run_audit.py"),
            "--protocol",
            protocol,
            "--gpu",
            self.gpu,
            "--output-root",
            str(self.scientific_root),
            "--covariance-mode",
            "both",
            "--oracle-head-eval-tasks",
            "all",
            "--require-ca-regression-pass",
        ]
        self.run(protocol, "scientific_audit", command, self.logs_root / f"{protocol}_audit.log")
        if not complete_audit(self.scientific_root, protocol):
            raise RuntimeError(f"{protocol} audit exited but scientific completion audit failed")
        self.protocol_update(
            protocol,
            audit="COMPLETE",
            a_d="PASS",
            visualization="COMPLETE",
            report="COMPLETE",
            status="COMPLETE",
        )

    def run_preflight(self) -> None:
        preflight_root = self.scientific_root / "_preflight"
        target = audit_dir(preflight_root, "IA5")
        regression = read_json(target / "ca_regression_check.json")
        check = read_json(self.scientific_root / "preflight_check.json")
        if regression.get("status") == "PASS" and check.get("status") == "PASS":
            self.status["skipped_complete_runs"].append("IA5:preflight")
            self.update()
            return
        if target.exists():
            preserved = preserve_partial(target, preflight_root)
            self.status["preserved_partial_runs"].append(str(preserved))
        command = [
            sys.executable,
            str(ROOT / "run_audit.py"),
            "--protocol",
            "IA5",
            "--gpu",
            self.gpu,
            "--task",
            "2",
            "--output-root",
            str(preflight_root),
            "--covariance-mode",
            "diagonal",
            "--oracle-head-eval-tasks",
            "all",
            "--skip-visualizations",
            "--require-ca-regression-pass",
        ]
        self.run("IA5", "preflight", command, self.logs_root / "IA5_preflight.log")
        self.validate_preflight(target)

    def validate_preflight(self, target: Path) -> None:
        manifest = read_json(target / "audit_manifest.json")
        regression = read_json(target / "ca_regression_check.json")
        checkpoint = training_dir("IA5") / "task_checkpoints" / "task_02.pt"
        sidecar = training_dir("IA5") / "pre_ca_classifier" / "task_02.pt"
        rows: list[dict[str, str]] = []
        replacement = target / "oracle_classifier_replacement.csv"
        if replacement.is_file():
            with replacement.open(newline="", encoding="utf-8") as handle:
                rows = [
                    row
                    for row in csv.DictReader(handle)
                    if row.get("condition") in {
                        "A_STRUCTURE",
                        "B_ORACLE_MEAN",
                        "C_ORACLE_COVARIANCE",
                        "D_FULL_ORACLE",
                    }
                ]
        guards = manifest.get("leakage_guards", {})
        checks = [
            {"name": "pre_ca_checkpoint", "pass": sidecar.is_file()},
            {"name": "post_ca_checkpoint", "pass": checkpoint.is_file()},
            {"name": "condition_A_reproduces_post_ca", "pass": regression.get("status") == "PASS"},
            {
                "name": "A_B_C_D_identical_pre_ca_initialization",
                "pass": len(rows) == 4
                and len({row.get("initialization_checksum") for row in rows}) == 1
                and {row.get("initialization_source") for row in rows} == {"exact_pre_ca_sidecar"},
            },
            {
                "name": "full_old_training_samples",
                "pass": manifest.get("max_samples_per_class") is None
                and manifest.get("max_classes") is None
                and manifest.get("primary_oracle") == "oracle_train_statistics",
            },
            {
                "name": "oracle_train_and_evaluation_test_are_separate",
                "pass": manifest.get("oracle_splits") == ["train", "test"],
            },
            {
                "name": "old_samples_not_in_training_graph",
                "pass": all(
                    guards.get(key) is True
                    for key in (
                        "model_eval",
                        "torch_no_grad",
                        "parameters_require_grad_false",
                        "diagnostic_optimizer_is_head_only",
                    )
                )
                and guards.get("audit_values_returned_to_training") is False,
            },
            {
                "name": "diagnostic_does_not_modify_main_checkpoint",
                "pass": manifest.get("checkpoint_fingerprints_before_after_equal") is True,
            },
        ]
        result = {
            "protocol": "IA5",
            "transition": "T2",
            "status": "PASS" if all(item["pass"] for item in checks) else "FAIL",
            "checked_at_utc": utc_now(),
            "checks": checks,
            "source": str(target),
        }
        write_json(self.scientific_root / "preflight_check.json", result)
        if result["status"] != "PASS":
            raise RuntimeError(f"IA5 scientific preflight failed: {result}")

    def execute(self) -> None:
        self.update(status="RUNNING", current_stage="IA5_preflight_training")
        self.ensure_training("IA5")
        self.run_preflight()
        for protocol in PROTOCOL_ORDER:
            try:
                self.ensure_training(protocol)
                self.run_full_audit(protocol)
            except Exception as error:  # keep independent protocols moving
                failure = {
                    "protocol": protocol,
                    "stage": self.status.get("current_stage"),
                    "type": type(error).__name__,
                    "reason": str(error),
                    "traceback": traceback.format_exc(),
                    "at_utc": utc_now(),
                }
                self.status["failures"].append(failure)
                self.protocol_update(protocol, status="FAIL")
        all_complete = all(
            self.status["protocols"][protocol]["status"] == "COMPLETE"
            for protocol in PROTOCOL_ORDER
        )
        self.update(
            status="COMPLETE" if all_complete else "COMPLETE_WITH_FAILURES",
            current_protocol=None,
            current_stage="finished",
            current_task=None,
            ended_at_utc=utc_now(),
        )
        if not all_complete:
            raise RuntimeError("scientific queue finished with one or more failed protocols")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="1", choices=("1",), help="fixed by design")
    parser.add_argument("--output-root", default=str(ROOT / "scientific_audit"))
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lock_path = ROOT / "logs" / "scientific_gpu1.queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another scientific GPU-1 queue holds {lock_path}") from error
        lock.write(str(os.getpid()))
        lock.flush()
        queue = Queue(args)
        try:
            queue.execute()
        except Exception as error:
            if queue.status.get("status") not in {"COMPLETE", "COMPLETE_WITH_FAILURES"}:
                queue.status["failures"].append(
                    {
                        "protocol": queue.status.get("current_protocol"),
                        "stage": queue.status.get("current_stage"),
                        "type": type(error).__name__,
                        "reason": str(error),
                        "traceback": traceback.format_exc(),
                        "at_utc": utc_now(),
                    }
                )
                queue.update(status="FAIL", ended_at_utc=utc_now())
            raise


if __name__ == "__main__":
    main()
