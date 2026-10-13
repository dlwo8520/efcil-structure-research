"""Snapshot the control layer and record immutable dependency digests."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".json", ".md", ".sh", ".txt", ".yaml", ".yml", ".toml"}
EXCLUDED = {".git", "__pycache__", "outputs", "logs", "reports", ".pytest_cache"}


def _files(root):
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        yield path, relative


def _tree_digest(root):
    digest = hashlib.sha256()
    count = 0
    for path, relative in _files(root):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return digest.hexdigest(), count


def create_control_snapshot(
    run_dir, *, base_root, e3_root, similarity_root, control_root
):
    """Create the fourth-layer source snapshot once per run."""
    run_dir = Path(run_dir).resolve()
    control_root = Path(control_root).resolve()
    destination = run_dir / "control_source_snapshot"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for source, relative in _files(control_root):
        payload = source.read_bytes()
        target = destination / "extension" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    dependencies = {}
    for name, root in (
        ("structure_base", base_root),
        ("e3_followup", e3_root),
        ("similarity_extension", similarity_root),
        ("control_extension", control_root),
    ):
        resolved = Path(root).resolve()
        digest, count = _tree_digest(resolved)
        dependencies[name] = {
            "root": str(resolved),
            "sha256": digest,
            "source_file_count": count,
        }
    manifest_path.write_text(
        json.dumps(
            {"dependencies": dependencies, "extension_files": records},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
