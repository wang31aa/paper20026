#!/usr/bin/env python3
"""Validate the rights-filtered public claim-to-source provenance graph."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


payload = json.loads((AUDIT / "provenance_graph.json").read_text(encoding="utf-8"))
rows = payload["rows"]
csv_rows = list(csv.DictReader((AUDIT / "provenance_graph.csv").open(
    encoding="utf-8", newline="")))
errors: list[str] = []
if [row["claim_id"] for row in rows] != [row["claim_id"] for row in csv_rows]:
    errors.append("CSV/JSON claim sets differ")
for row in rows:
    checks = [
        (row["source_object"], row["source_sha256"]),
        (row["derived_artifact"], row["derived_sha256"]),
        (row["raw_identity_object"], row["raw_identity_sha256"]),
    ]
    checks.extend((rel, None) for rel in str(row["analysis_script"]).split(";"))
    for rel, expected in checks:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"missing {row['claim_id']} path: {rel}")
        elif expected is not None and sha256(path) != expected:
            errors.append(f"hash mismatch {row['claim_id']}: {rel}")
exclusions = json.loads((AUDIT / "RELEASE_PROVENANCE_EXCLUSIONS.json").read_text(
    encoding="utf-8"))["excluded"]
if any(item["claim_id"] in {row["claim_id"] for row in rows} for item in exclusions):
    errors.append("a claim is both retained and excluded")
if errors:
    raise SystemExit("FAIL release provenance\n" + "\n".join(errors))
print(f"PASS: {len(rows)} public claim paths; {len(exclusions)} rights-controlled path recorded")
