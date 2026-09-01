#!/usr/bin/env python3
"""Fail closed on release hashes, forbidden paths and GitHub file limits."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
FORBIDDEN = ("incoming/", "external_physical_validation/raw/", "public_data/raw/",
             "chaotic-image-encryption-master/", "matlab_reproduction/",
             "matlab_branch_reproduction/")
LIMIT = 100_000_000
RTHS_DERIVED = {
    "cps_transfer_benchmark/results/SHA256SUMS.json",
    "cps_transfer_benchmark/results/rths_fault_metrics.csv",
    "cps_transfer_benchmark/results/rths_run_metrics.csv",
    "cps_transfer_benchmark/results/summary.json",
}
OPEN_CC_BY_DERIVED_PREFIXES = (
    "cps_transfer_benchmark/results/openmct_",
    "cps_transfer_benchmark/results/water_hil_",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def licence(rel: str) -> str:
    if rel in RTHS_DERIVED:
        return "CC-BY-SA-4.0-upstream-derived-author-confirmation-required"
    if rel.startswith(OPEN_CC_BY_DERIVED_PREFIXES):
        return "CC-BY-4.0-upstream-derived-author-confirmation-required"
    if rel.startswith("external_physical_validation/results/"):
        return "CC-BY-4.0-upstream-derived-author-confirmation-required"
    if rel.endswith((".py", ".sh", "Makefile")) or rel.startswith(".github/"):
        return "MIT-provisional-author-confirmation-required"
    return "CC-BY-4.0-provisional-author-confirmation-required"


payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
if "--seal" in sys.argv:
    files = []
    old = {item["path"]: item for item in payload["files"]}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        files.append({
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "licence": licence(rel),
        })
    payload["files"] = files
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"SEALED: {len(files)} release files")
expected = {item["path"]: item for item in payload["files"]}
actual = {
    p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*")
    if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts
}
errors: list[str] = []
if set(expected) != set(actual):
    errors.append("manifest file set differs from repository file set")
for rel, item in expected.items():
    path = actual.get(rel)
    if path is None:
        continue
    if (any(token in rel for token in FORBIDDEN)
            and rel != "public_data/raw/README.md"):
        errors.append(f"forbidden path: {rel}")
    if path.stat().st_size >= LIMIT:
        errors.append(f"GitHub-size violation: {rel}")
    if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
        errors.append(f"integrity mismatch: {rel}")
    if not item.get("licence"):
        errors.append(f"missing licence status: {rel}")
    elif item["licence"] != licence(rel):
        errors.append(
            f"licence classification mismatch: {rel}: "
            f"{item['licence']} != {licence(rel)}"
        )
if payload.get("public_doi") is not None or payload.get("repository_url") is not None:
    errors.append("unverified public identifier present")
readme = ROOT / "README.md"
if readme.is_file():
    for ref in re.findall(r"`([^`\n]+\.md)`", readme.read_text(encoding="utf-8")):
        if not (ROOT / ref).is_file():
            errors.append(f"README local Markdown reference missing: {ref}")
if errors:
    raise SystemExit("FAIL release validation\n" + "\n".join(errors))
print(f"PASS: {len(expected)} release files; technical path/size/hash gates closed; rights remain provisional")
