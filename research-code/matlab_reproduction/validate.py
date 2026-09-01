#!/usr/bin/env python3
"""Validate immutable phase-1 MATLAB reproduction artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    errors: list[str] = []
    with (HERE / "source_manifest.csv").open() as f:
        for row in csv.DictReader(f):
            p = resolve(row["path"])
            if not p.is_file(): errors.append(f"missing: {p}")
            elif digest(p) != row["sha256"]: errors.append(f"hash mismatch: {p}")

    g1 = json.loads((HERE / "results/graph1_full/parity_summary.json").read_text())
    g2 = json.loads((HERE / "results/graph2_heldout/parity_summary.json").read_text())
    post = json.loads((HERE / "posthoc_correction/graph2_full/parity_summary.json").read_text())
    if not (g1["early_gate_pass"] and g1["full_gate_pass"]): errors.append("graph1 expected PASS")
    if g2["early_gate_pass"] or g2["full_gate_pass"]: errors.append("original graph2 held-out failure was altered")
    if not post["early_gate_pass"]: errors.append("posthoc graph2 early gate expected PASS")
    if post["full_gate_pass"]: errors.append("posthoc graph2 full failure was altered")
    if errors:
        print("MATLAB_REPRODUCTION_VALIDATION: FAIL")
        for e in errors: print("-", e)
        raise SystemExit(1)
    print("MATLAB_REPRODUCTION_VALIDATION: PASS")
    print("Graph1 full PASS; original Graph2 held-out FAIL preserved; posthoc early PASS/full FAIL preserved; 10 hashes PASS.")


if __name__ == "__main__":
    main()
