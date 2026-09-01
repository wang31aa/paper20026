#!/usr/bin/env python3
"""Property tests for the general participation-regime theorem."""
from __future__ import annotations
import json, math, random
from pathlib import Path

rng = random.Random(20260823)
grid = [0.2 + 2.8 * i / 1200 for i in range(1201)]
tol = 2e-9
counts = {"decreasing": 0, "increasing": 0, "interior": 0}

for case in range(1200):
    mode = case % 3
    curves = []
    for _ in range(4):
        a = 10 ** rng.uniform(-2.0, 0.6)
        c = 10 ** rng.uniform(-2.0, 0.4)
        b = rng.uniform(-0.5, 0.8)
        if mode == 0:
            c = 0.0
        elif mode == 1:
            a = 0.0
        curves.append((a, c, b))
    y = [max(b + a/r + c*r*r for a, c, b in curves) for r in grid]
    d = [v-u for u, v in zip(y, y[1:])]
    assert all(y[i] <= (y[i-1]+y[i+1])/2 + tol for i in range(1,len(y)-1))
    feasible = [i for i,v in enumerate(y) if v <= 1]
    if feasible:
        assert feasible == list(range(min(feasible), max(feasible)+1))
    if mode == 0:
        assert all(x <= tol for x in d); counts["decreasing"] += 1
    elif mode == 1:
        assert all(x >= -tol for x in d); counts["increasing"] += 1
    else:
        if d[0] < 0 < d[-1]:
            assert 0 < y.index(min(y)) < len(y)-1
            counts["interior"] += 1

# Strictly increasing coordinate and risk transformations preserve membership.
sample = [(0.7/r + 0.18*r*r) for r in grid]
membership = [v <= 1 for v in sample]
rho_tilde = [math.log(r) for r in grid]
risk_tilde = [math.exp(v) for v in sample]
assert all(a == (b <= math.e) for a,b in zip(membership,risk_tilde))
assert all(x < y for x,y in zip(rho_tilde,rho_tilde[1:]))

out = {
    "status": "PASS",
    "seed": 20260823,
    "random_families": 1200,
    "grid_points": len(grid),
    "checks": {
        "convexity": True,
        "sublevel_interval": True,
        "zero_exposure_nonincreasing": True,
        "zero_information_loss_nondecreasing": True,
        "interior_derivative_test": True,
        "order_preserving_coordinate_invariance": True,
        "monotone_risk_transform_invariance": True
    },
    "classification_counts": counts,
    "scope": "numerical property tests supporting, not replacing, the analytic proof"
}
path = Path(__file__).with_name("general_participation_regime_validation.json")
path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
