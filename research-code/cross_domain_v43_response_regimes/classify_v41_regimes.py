#!/usr/bin/env python3
"""Retrospective, rule-based response classification of frozen V41 holdouts."""
from __future__ import annotations
import csv, json, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "cross_domain_v41/results/v41_all_policy_runs.csv"
OUT = Path(__file__).resolve().parent / "results"
POLICY = "all_coupled"
TOL = 0.05
WINDOW_GAP = 0.10


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (float("nan"), float("nan"))
    p = k/n; den = 1 + z*z/n
    centre = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return centre-half, centre+half


def classify(y):
    d = [b-a for a, b in zip(y, y[1:])]
    if max(y)-min(y) <= TOL: return "plateau"
    if all(x >= -TOL for x in d): return "monotone_improving"
    if all(x <= TOL for x in d): return "monotone_worsening"
    m = max(y); ids = [i for i, v in enumerate(y) if abs(v-m) <= TOL]
    if ids and min(ids) > 0 and max(ids) < len(y)-1 and \
       m-y[0] >= WINDOW_GAP and m-y[-1] >= WINDOW_GAP:
        return "finite_window"
    return "irregular_or_unresolved"


rows = list(csv.DictReader(SOURCE.open()))
rows = [r for r in rows if r["split"] == "heldout" and r["policy"] == POLICY]
group = defaultdict(list)
for r in rows:
    group[(r["domain"], float(r["rho"]))].append(int(r["task_success"]))
rhos = sorted({float(r["rho"]) for r in rows})
domains = sorted({r["domain"] for r in rows})
curves, summary = [], []
for domain in domains:
    rates = []
    for rho in rhos:
        yy = group[(domain, rho)]; k, n = sum(yy), len(yy)
        lo, hi = wilson(k, n); rate = k/n; rates.append(rate)
        curves.append({"domain": domain, "rho": rho, "successes": k,
                       "runs": n, "success_rate": rate,
                       "wilson_low": lo, "wilson_high": hi})
    summary.append({"domain": domain, "policy": POLICY,
                    "response_class": classify(rates),
                    "minimum_rate": min(rates), "maximum_rate": max(rates),
                    "best_rho": rhos[rates.index(max(rates))]})

OUT.mkdir(exist_ok=True)
for name, data in (("v43_curve_summary.csv", curves), ("v43_regime_summary.csv", summary)):
    with (OUT/name).open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
report = {
    "evidence_class": "RETROSPECTIVE_CLASSIFICATION_OF_FROZEN_V41_HELDOUTS",
    "source": str(SOURCE.relative_to(ROOT)), "policy": POLICY,
    "rules": {"difference_tolerance": TOL, "window_gap": WINDOW_GAP},
    "domains": summary,
    "limitations": [
        "Classification rules were created after V41 outcomes existed.",
        "This is not leave-domain-out prediction.",
        "Domain descriptors are incomplete for a fair universal predictor.",
        "Success curves aggregate two sizes and cannot identify a physical law."
    ]
}
(OUT/"V43_RESPONSE_REGIME_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
