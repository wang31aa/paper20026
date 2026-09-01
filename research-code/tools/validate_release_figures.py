#!/usr/bin/env python3
"""Dependency-free publication-export checks for all figure builders."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "figures/build_figures.py", "figures/build_observer_figure.py",
    "figures/build_cps_transfer_figure.py", "figures/build_openmct_figure.py",
    "figures/build_observer_in_loop_certified_figure.py",
]
required = ["svg.fonttype", "svg.hashsalt", ".svg", ".pdf", ".png", ".tiff", "dpi=600"]
errors: list[str] = []
for rel in FILES:
    path = ROOT / rel
    source = path.read_text(encoding="utf-8")
    ast.parse(source, filename=rel)
    for token in required:
        haystack = source.replace(" ", "") if token == "dpi=600" else source
        if token not in haystack:
            errors.append(f"{rel}: missing {token}")
    if 'metadata={"Date": None}' not in source:
        errors.append(f"{rel}: SVG date metadata is not disabled")
    if 'metadata={"CreationDate": None, "ModDate": None}' not in source:
        errors.append(f"{rel}: PDF creation/modification metadata is not disabled")
if errors:
    raise SystemExit("FAIL figure release checks\n" + "\n".join(errors))
print(f"PASS: {len(FILES)} figure builders have deterministic editable/vector/600-dpi exports")
