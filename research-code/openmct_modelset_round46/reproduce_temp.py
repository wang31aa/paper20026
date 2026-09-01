#!/usr/bin/env python3
"""Regenerate Round46 in a temporary directory and compare frozen decisions."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "results"


def tree_hash() -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in FROZEN.iterdir() if p.is_file()}


before = tree_hash()
with tempfile.TemporaryDirectory(prefix="openmct_r46_") as tmp:
    out = Path(tmp) / "results"
    subprocess.run([sys.executable, str(HERE / "run_preflight.py"), "--out", str(out)], check=True)
    subprocess.run([sys.executable, str(HERE / "independent_sdp_check.py"), "--out", str(out)], check=True)
    frozen = json.loads((FROZEN / "preflight.json").read_text(encoding="utf-8"))
    rebuilt = json.loads((out / "preflight.json").read_text(encoding="utf-8"))
    frozen_sdp = json.loads((FROZEN / "independent_sdp.json").read_text(encoding="utf-8"))
    rebuilt_sdp = json.loads((out / "independent_sdp.json").read_text(encoding="utf-8"))
    for key in ("protocol_sha256", "rho_interval", "rho_corner_count", "source_checks",
                "graph_checks", "exploratory_metric_preflight_supported",
                "exploratory_sensitivity_simulation_authorized",
                "qualified_openmct_network_model", "physical_evidence"):
        if rebuilt[key] != frozen[key]:
            raise SystemExit(f"FAIL temporary rebuild field: {key}")
    for name in frozen["graphs"]:
        for key in ("g", "euclidean_margin", "minimum_recomputed_euclidean_margin",
                    "generalized_margins", "minimum_generalized_margin"):
            if not np.allclose(rebuilt["graphs"][name][key], frozen["graphs"][name][key],
                               rtol=0, atol=6e-8):
                raise SystemExit(f"FAIL temporary rebuild numeric: {name}/{key}")
    if rebuilt_sdp["all_graphs_positive"] != frozen_sdp["all_graphs_positive"]:
        raise SystemExit("FAIL temporary independent-SDP decision")
    for name in frozen_sdp["graphs"]:
        for solver in ("CLARABEL", "SCS"):
            for key in ("margin", "g"):
                if not np.allclose(
                    rebuilt_sdp["graphs"][name][solver][key],
                    frozen_sdp["graphs"][name][solver][key],
                    rtol=0,
                    atol=6e-8,
                ):
                    raise SystemExit(
                        f"FAIL temporary independent-SDP numeric: {name}/{solver}/{key}"
                    )
if tree_hash() != before:
    raise SystemExit("FAIL frozen results changed during temporary rebuild")
print("PASS: temporary Round46 rebuild agrees within 6e-8; frozen bytes unchanged")
