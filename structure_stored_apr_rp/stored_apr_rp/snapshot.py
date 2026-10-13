"""Snapshot only this variant's source, excluding mutable results and reports."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SUFFIXES = {".py", ".json", ".md", ".sh"}
EXCLUDED_PARTS = {"outputs", "logs", "__pycache__", ".git", ".pytest_cache"}
EXCLUDED_NAMES = {"stored_apr_rp_report.md"}


def source_files(root):
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        relative = path.relative_to(root)
        if relative.name in EXCLUDED_NAMES or any(part in EXCLUDED_PARTS for part in relative.parts):
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


def create_snapshot(run_dir, root):
    destination = Path(run_dir) / "stored_apr_source_snapshot"
    manifest = destination / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    records = []
    for source, relative in source_files(root):
        target = destination / "extension" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({
            "path": relative.as_posix(),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        })
    digest, count = tree_digest(root)
    result = {"root": str(Path(root).resolve()), "sha256": digest,
              "source_file_count": count, "files": records}
    manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
