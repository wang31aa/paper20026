#!/usr/bin/env python3
"""Build a public candidate containing programs and required configs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {".py", ".sh", ".m"}
CONFIG_SUFFIXES = {".json", ".yml", ".yaml", ".toml"}
ROOT_FILES = {
    "requirements.txt", "requirements-linux-py313.lock", "environment.yml",
    "LICENSE", "CITATION.cff", ".gitignore",
}
EXCLUDED_PARTS = {
    ".git", ".venv", ".venv-figures", "__pycache__", "vendor",
    "runtime_py312", "submission_bundle", "manuscript", "audit", "results",
    "raw", "cache", "incoming", "author_input", "source_data",
}
EXCLUDED_PREFIXES = (
    "Nature_TAC2023_GitHub_Release_", "Nature_TAC2023_Code_Only_",
    "release/", "sources/",
)
CONFIG_MARKERS = (
    "config", "contract", "protocol", "schema", "parameter", "registry",
    "topology", "environment", "preregistration",
)
MAX_BYTES = 20_000_000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eligible(path: Path) -> bool:
    if not path.is_file() or path.name == ".DS_Store":
        return False
    rel = path.relative_to(ROOT).as_posix()
    if rel in ROOT_FILES:
        return True
    if rel.startswith(EXCLUDED_PREFIXES) or any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
        return False
    if path.stat().st_size >= MAX_BYTES:
        return False
    if path.name == "Makefile":
        return True
    if path.suffix.lower() in CODE_SUFFIXES:
        return True
    return path.suffix.lower() in CONFIG_SUFFIXES and any(
        marker in path.name.lower() for marker in CONFIG_MARKERS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--repository-url",
        default=None,
        choices=["https://github.com/wang31aa/paper20026"],
        help="Verified publication target; omit for a local-only candidate.",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=False)

    files = []
    for src in sorted(ROOT.rglob("*")):
        if not eligible(src):
            continue
        rel = src.relative_to(ROOT)
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(rel.as_posix())

    shutil.copy2(ROOT / "release" / "README_CODE_ONLY_GITHUB.md", out / "README.md")
    workflow = out / ".github" / "workflows" / "code-check.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        "name: code-check\non: [push, pull_request, workflow_dispatch]\n"
        "jobs:\n  verify:\n    runs-on: ubuntu-latest\n    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n        with:\n          python-version: '3.13'\n"
        "      - run: python -m compileall -q .\n"
        "      - run: python tools/validate_code_only_release.py\n",
        encoding="utf-8",
    )

    manifest_files = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "CODE_ONLY_MANIFEST.json":
            manifest_files.append({
                "path": path.relative_to(out).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    manifest = {
        "schema": "CODE-ONLY-PUBLIC-CANDIDATE-1",
        "status": "publication candidate" if args.repository_url else "local candidate; not uploaded",
        "repository_url": args.repository_url,
        "public_doi": None,
        "excluded_classes": [
            "manuscript", "proof documents", "internal audits", "raw data",
            "derived data", "frozen numerical results", "rendered figures",
        ],
        "files": manifest_files,
    }
    (out / "CODE_ONLY_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS: built code-only candidate with {len(manifest_files)} files at {out}")


if __name__ == "__main__":
    main()
