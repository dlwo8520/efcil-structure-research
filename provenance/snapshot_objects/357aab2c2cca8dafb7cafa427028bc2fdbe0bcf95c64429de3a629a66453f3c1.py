"""Immutable source snapshot for the fourth-layer extension."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".json", ".md", ".sh", ".txt", ".yaml", ".yml", ".toml"}
EXCLUDED = {".git", "__pycache__", "outputs", "logs", "reports", ".pytest_cache"}


def source_files(root):
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        yield path, relative


def tree_digest(root):
    digest = hashlib.sha256()
    count = 0
    for path, relative in source_files(root):
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        count += 1
    return digest.hexdigest(), count


def create_sweep_snapshot(run_dir, roots):
    destination = Path(run_dir).resolve() / "selective_sweep_source_snapshot"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    project_root = Path(roots["selective_sweep"]).resolve()
    records = []
    for source, relative in source_files(project_root):
        payload = source.read_bytes()
        target = destination / "extension" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {"path": relative.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
    dependencies = {}
    for name, root in roots.items():
        digest, count = tree_digest(root)
        dependencies[name] = {
            "root": str(Path(root).resolve()),
            "sha256": digest,
            "source_file_count": count,
        }
    record = {"dependencies": dependencies, "extension_files": records}
    manifest_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record
