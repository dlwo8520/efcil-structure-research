"""Immutable, source-only run snapshots for reproducibility."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List


SOURCE_SUFFIXES = {".py", ".json", ".md", ".txt", ".sh", ".yaml", ".yml", ".toml"}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "datasets",
    "checkpoints",
    "reports",
    "source_snapshot",
}


def _command(command: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            command, cwd=cwd, text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}\n"


def _source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        yield path


def _copy_tree(root: Path, destination: Path, label: str) -> List[Dict[str, str | int]]:
    manifest = []
    for source in _source_files(root):
        relative = source.relative_to(root)
        target = destination / label / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        payload = source.read_bytes()
        manifest.append(
            {
                "tree": label,
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return manifest


def create_run_snapshot(
    run_dir: str | Path,
    *,
    config: Dict,
    base_root: str | Path,
    extension_root: str | Path,
    launch_command: str,
) -> None:
    """Create the snapshot once; resumed runs preserve the original evidence."""
    run_path = Path(run_dir).resolve()
    base_path = Path(base_root).resolve()
    extension_path = Path(extension_root).resolve()
    snapshot = run_path / "source_snapshot"
    manifest_path = snapshot / "manifest.json"
    if manifest_path.is_file():
        return
    snapshot.mkdir(parents=True, exist_ok=True)
    (run_path / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    (run_path / "command.txt").write_text(launch_command.rstrip() + "\n", encoding="utf-8")
    git_status = []
    git_diff = []
    for label, root in (("base", base_path), ("extension", extension_path)):
        commit = _command(["git", "rev-parse", "HEAD"], root).strip()
        git_status.append(
            f"[{label}] {root}\ncommit={commit}\n"
            + _command(["git", "status", "--short"], root)
        )
        git_diff.append(f"[{label}] {root}\n" + _command(["git", "diff", "--binary", "HEAD"], root))
    (run_path / "git_status.txt").write_text("\n".join(git_status), encoding="utf-8")
    (run_path / "git_diff.patch").write_text("\n".join(git_diff), encoding="utf-8")

    environment = [
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
    ]
    try:
        import torch

        environment.extend(
            [
                f"torch={torch.__version__}",
                f"cuda_runtime={torch.version.cuda}",
                f"cuda_available={torch.cuda.is_available()}",
                f"gpu_count={torch.cuda.device_count()}",
            ]
        )
        for index in range(torch.cuda.device_count()):
            environment.append(f"gpu_{index}={torch.cuda.get_device_name(index)}")
    except ImportError:
        environment.append("torch=UNAVAILABLE")
    (run_path / "environment.txt").write_text("\n".join(environment) + "\n", encoding="utf-8")

    manifest = _copy_tree(base_path, snapshot, "structure_base")
    manifest.extend(_copy_tree(extension_path, snapshot, "e3_followup"))
    manifest_path.write_text(
        json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
