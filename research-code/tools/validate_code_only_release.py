#!/usr/bin/env python3
"""Fail closed if a code-only public candidate contains evidence documents."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "CODE_ONLY_MANIFEST.json"
FORBIDDEN_PARTS = {"manuscript", "audit", "results", "raw", "cache", "source_data", "author_input"}
FORBIDDEN_SUFFIXES = {".tex", ".pdf", ".doc", ".docx", ".csv", ".npz", ".mat", ".png", ".svg", ".tif", ".tiff"}
LIMIT = 100_000_000
ALLOWED_REPOSITORY_URLS = {None, "https://github.com/wang31aa/paper20026"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


x = json.loads(MANIFEST.read_text(encoding="utf-8"))
expected = {row["path"]: row for row in x["files"]}
actual = {
    p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob("*")
    if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts
}
errors = []
if set(expected) != set(actual):
    errors.append("manifest file set differs")
for rel, path in actual.items():
    row = expected.get(rel, {})
    if any(part in FORBIDDEN_PARTS for part in Path(rel).parts):
        errors.append(f"forbidden directory: {rel}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        errors.append(f"forbidden evidence/document suffix: {rel}")
    if path.stat().st_size >= LIMIT:
        errors.append(f"GitHub size violation: {rel}")
    if row and (row.get("bytes") != path.stat().st_size or row.get("sha256") != digest(path)):
        errors.append(f"manifest mismatch: {rel}")
if x.get("repository_url") not in ALLOWED_REPOSITORY_URLS:
    errors.append("unverified repository identifier present")
if x.get("public_doi") is not None:
    errors.append("unverified DOI present")
if errors:
    raise SystemExit("FAIL code-only release\n" + "\n".join(errors))
print(f"PASS: {len(actual)} files; code-only scope and manifest integrity verified")
