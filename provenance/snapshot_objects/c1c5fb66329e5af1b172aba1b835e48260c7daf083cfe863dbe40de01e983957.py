"""Immutable source snapshot for the isolated RP extension."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".json", ".md", ".sh", ".txt", ".yaml", ".yml", ".toml"}
EXCLUDED_PARTS = {".git", "__pycache__", "outputs", "logs", ".pytest_cache"}
GENERATED_FILES = {"rp_experiment_plan.md", "rp_protoclad_report.md"}


def source_files(root):
    root = Path(root).resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if relative.name in GENERATED_FILES or any(part in EXCLUDED_PARTS for part in relative.parts):
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


def create_rp_snapshot(run_dir, root):
    destination = Path(run_dir).resolve() / "rp_source_snapshot"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for source, relative in source_files(root):
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
    digest, count = tree_digest(root)
    record = {
        "root": str(Path(root).resolve()),
        "sha256": digest,
        "source_file_count": count,
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record

