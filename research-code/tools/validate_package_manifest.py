#!/usr/bin/env python3
"""Validate sizes and SHA-256 digests in the release manifest."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


manifest_path = Path(sys.argv[1] if len(sys.argv) > 1 else "package_manifest.json")
root = manifest_path.resolve().parent
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
errors: list[str] = []
exclude = {"package_manifest.json", "AUDIT_SHA256.txt", ".DS_Store"}
transient_suffixes = {".aux", ".log", ".out", ".pyc", ".fls", ".fdb_latexmk"}

def governed(path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return (path.is_file() and path.name not in exclude
            and path.suffix not in transient_suffixes
            and not any(part.startswith("Nature_TAC2023_GitHub_Release_") for part in path.parts)
            and not any(part in {".git", ".venv", ".venv-figures"} for part in path.parts)
            and not rel.startswith("external_physical_validation/raw/")
            and not (rel.startswith("public_data/raw/") and path.name != "README.md")
            and not rel.startswith("uav_causal_validation/vendor/")
            and not rel.startswith("uav_causal_validation/runtime_py312/")
            and not (rel.startswith("public_cluster_cases/cache/") and path.name != "README.md")
            and "__pycache__" not in path.parts)

listed = {item["path"] for item in payload["files"]}
actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if governed(path)}
for rel in sorted(actual - listed):
    errors.append(f"unlisted: {rel}")
for rel in sorted(listed - actual):
    errors.append(f"not-governed-or-absent: {rel}")
for item in payload["files"]:
    path = root / item["path"]
    if not path.is_file():
        errors.append(f"missing: {item['path']}")
        continue
    if item["bytes"] is not None and path.stat().st_size != item["bytes"]:
        errors.append(f"size: {item['path']}")
    if item["sha256"] is not None and digest(path) != item["sha256"]:
        errors.append(f"sha256: {item['path']}")
if errors:
    raise SystemExit("FAIL manifest\n" + "\n".join(errors))
print(f"PASS: {len(payload['files'])} manifest entries validated")
