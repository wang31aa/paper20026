#!/usr/bin/env python3
"""Fail closed on the frozen CERT-OIL-R4 mechanism-scan outputs."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / "results_r4/raw_runs.csv").open()))
summary = json.loads((ROOT / "results_r4/validation.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "PREREGISTRATION_R4.md").read_bytes()).hexdigest()
assert summary["protocol_id"] == "CERT-OIL-R4"
assert summary["protocol_sha256"] == protocol_hash
assert len(rows) == summary["runs"] == summary["finite_runs"] == 576
expected = set(itertools.product(
    ("1.5", "2.0", "4.0", "8.0"),
    ("2.0", "5.0", "15.0", "35.0"),
    ("chain", "star", "branch", "cyclic"),
    ("0.05", "0.15", "0.3"),
    tuple(str(i) for i in range(3)),
))
observed = {(r["alpha"], r["gamma_state"], r["topology"],
             r["heterogeneity"], r["seed"]) for r in rows}
assert observed == expected
assert all(r["finite"].lower() == "true" for r in rows)
assert all(float(r["max_tracking_envelope_ratio"]) <= 1 + 1e-10 for r in rows)
assert all(float(r["max_observer_envelope_ratio"]) <= 1 + 1e-10 for r in rows)
assert summary["tracking_envelope_violations"] == 0
assert summary["observer_envelope_violations"] == 0
print("PASS: CERT-OIL-R4 has 576/576 unique finite runs and no envelope violations")
