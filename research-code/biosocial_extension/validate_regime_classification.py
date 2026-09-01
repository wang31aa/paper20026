#!/usr/bin/env python3
"""Numerical property tests for Corollary 4.1; not a substitute for proof."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(20260823)
rho = np.linspace(0.12, 4.0, 801)


def curve(a, d0, db, u, vh, ep, eu):
    residual = np.maximum(d0[:, None] + db[:, None]*rho - u[:, None], 0) / a[:, None]
    peak = residual + vh[:, None]/rho
    return np.maximum(np.max(peak/ep, axis=0), np.max(residual/eu, axis=0))


checks = {"zero_exposure_monotone_nonincreasing": True,
          "zero_staleness_monotone_nondecreasing": True,
          "convexity_general": True,
          "feasible_sublevel_is_interval": True}
tested = 0
for _ in range(500):
    n = int(rng.integers(1, 9))
    a = rng.uniform(0.3, 2.2, n)
    d0 = rng.uniform(0.0, 0.5, n)
    u = rng.uniform(0.0, 0.45, n)
    ep, eu = rng.uniform(0.3, 1.4, 2)
    vh = rng.uniform(0.0, 0.8, n)
    z = curve(a, d0, np.zeros(n), u, vh, ep, eu)
    checks["zero_exposure_monotone_nonincreasing"] &= bool(np.all(np.diff(z) <= 1e-10))
    db = rng.uniform(0.0, 0.7, n)
    z = curve(a, d0, db, u, np.zeros(n), ep, eu)
    checks["zero_staleness_monotone_nondecreasing"] &= bool(np.all(np.diff(z) >= -1e-10))
    z = curve(a, d0, db, u, vh, ep, eu)
    # Convexity on an equally spaced grid: non-negative second differences.
    checks["convexity_general"] &= bool(np.all(np.diff(z, 2) >= -2e-8))
    feasible = np.flatnonzero(z <= 1.0)
    checks["feasible_sublevel_is_interval"] &= bool(
        len(feasible) == 0 or np.all(np.diff(feasible) == 1))
    tested += 1

report = {"qualified": all(checks.values()), "random_parameter_families": tested,
          "grid_points_per_family": len(rho), "checks": checks,
          "scope": "numerical property test supporting an analytic proof"}
out = ROOT / "results" / "regime_classification_validation.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["qualified"] else 1)
