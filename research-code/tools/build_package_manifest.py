#!/usr/bin/env python3
"""Build the deterministic release inventory (the inventory excludes itself)."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"package_manifest.json", "AUDIT_SHA256.txt", ".DS_Store"}
TRANSIENT_SUFFIXES = {".aux", ".log", ".out", ".pyc", ".fls", ".fdb_latexmk"}
CODE_SUFFIXES = {".py", ".sh"}
CODE_NAMES = {"Makefile", "Dockerfile", ".dockerignore", "requirements.txt",
              "requirements-linux-py313.lock", "environment.yml"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def licence(path: Path) -> str:
    rel = path.relative_to(ROOT).as_posix()
    if rel.startswith(
        "incoming/2026-07-28_author_matlab/source_tree/chaotic-image-encryption-master/"
    ) and path.suffix == ".m":
        return "origin-and-rights-unresolved-exclude-from-public-release"
    if rel.startswith("incoming/2026-07-28_author_matlab/source_tree/"):
        return "author-redistribution-authorized-no-standard-licence"
    if path.suffix in CODE_SUFFIXES or path.name in CODE_NAMES:
        return "MIT-provisional"
    return "CC-BY-4.0-provisional"


files = []
for path in sorted(ROOT.rglob("*")):
    rel = path.relative_to(ROOT).as_posix()
    if not path.is_file() or path.name in EXCLUDE or path.suffix in TRANSIENT_SUFFIXES:
        continue
    if any(part.startswith("Nature_TAC2023_GitHub_Release_") for part in path.parts):
        continue
    if any(part in {".git", ".venv", ".venv-figures"} for part in path.parts):
        continue
    if (rel.startswith("external_physical_validation/raw/")
            or (rel.startswith("public_data/raw/") and path.name != "README.md")):
        continue
    if rel.startswith("uav_causal_validation/vendor/"):
        continue
    if rel.startswith("uav_causal_validation/runtime_py312/"):
        continue
    if rel.startswith("public_cluster_cases/cache/") and path.name != "README.md":
        continue
    if "__pycache__" in path.parts:
        continue
    files.append({
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "integrity": "sha256",
        "licence": licence(path),
    })

payload = {
    "schema_version": "1.0",
    "archive_version": "0.6.0",
    "generated_date": date.today().isoformat(),
    "identity": "local release candidate; no DOI or repository deposit claimed",
    "licence_status": "author MATLAB redistribution authorized; remaining archive licensing provisional",
    "excluded": ["package_manifest.json (self)", "AUDIT_SHA256.txt (parallel ledger)",
                 "LaTeX/Python transient files",
                 "external_physical_validation/raw (third-party download cache)",
                 "uav_causal_validation/vendor (environment-specific dependency mirror)",
                 "uav_causal_validation/runtime_py312 (environment-specific dependency mirror)",
                 ".git and local virtual environments",
                 "public_data/raw (opt-in third-party download cache)"],
    "derived_policy": "Every final file, including regenerated PDFs, is byte-hashed at release sealing time",
    "file_count": len(files),
    "files": files,
}
(ROOT / "package_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
)
print(f"WROTE: {len(files)} entries")

# Keep a stable, portable parallel ledger for every manifest entry.
audit_lines = [
    f"{item['sha256']}  {item['path']}"
    for item in files
    if item["sha256"] is not None
]
(ROOT / "AUDIT_SHA256.txt").write_text(
    "\n".join(audit_lines) + "\n", encoding="utf-8"
)
print(f"WROTE: {len(audit_lines)} audit hashes")
