#!/usr/bin/env python3
"""Verify the immutable pre-Round37 figure snapshot before rebuilding figures."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "figures" / "originals_round37"
MANIFEST = SNAPSHOT / "MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = payload.get("records")
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("unsupported schema version")
    if payload.get("status") != "immutable pre-Round37 figure snapshot":
        errors.append("snapshot status changed")
    if not isinstance(records, list):
        raise SystemExit("FAIL original figure snapshot\nrecords is not a list")
    if payload.get("file_count") != 48 or len(records) != 48:
        errors.append(f"expected 48 records, found {len(records)}")
    declared = {record.get("path") for record in records if isinstance(record, dict)}
    actual = {path.relative_to(SNAPSHOT).as_posix() for path in SNAPSHOT.iterdir()
              if path.is_file() and path.name != MANIFEST.name}
    if declared != actual:
        errors.append("declared and actual snapshot file sets differ")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            errors.append("malformed record")
            continue
        path = SNAPSHOT / record["path"]
        if not path.is_file():
            errors.append(f"missing {record['path']}")
            continue
        if record.get("bytes") != path.stat().st_size:
            errors.append(f"size mismatch {record['path']}")
        if record.get("sha256") != sha256(path):
            errors.append(f"sha256 mismatch {record['path']}")
    if errors:
        raise SystemExit("FAIL original figure snapshot\n" + "\n".join(errors))
    print("PASS: immutable pre-Round37 figure snapshot, 48/48 files")


if __name__ == "__main__":
    main()
