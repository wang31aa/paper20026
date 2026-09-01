#!/usr/bin/env python3
"""Fail-closed validation for the discovery extension."""
from __future__ import annotations
import csv, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent

def main() -> None:
    with (HERE / "results/directed_network_lower_bound.csv").open(newline="") as f:
        lower = list(csv.DictReader(f))
    with (HERE / "results/domain_interventions.csv").open(newline="") as f:
        domain = list(csv.DictReader(f))
    summary = json.loads((HERE / "results/summary.json").read_text())
    assert len(lower) == 240 and len(domain) == 15
    assert {(r["case"], r["intervention"]) for r in domain} == {
        (c, i) for c in ("robot", "uav", "vehicle")
        for i in ("baseline", "information", "coupling", "heterogeneity", "topology")}
    assert all(math.isfinite(float(r["attainment_ratio"])) and 0 < float(r["attainment_ratio"]) <= 1 + 1e-10 for r in lower)
    assert all(math.isfinite(float(r["normalized_task_margin"])) and float(r["normalized_task_margin"]) >= 0 for r in domain)
    assert float(summary["directed_chain_max_attainment"]) > .80
    assert float(summary["directed_cyclic_max_attainment"]) > .86
    for name in ("Fig16_discovery_extension.pdf", "Fig16_discovery_extension.svg", "Fig16_discovery_extension.png"):
        assert (ROOT / "figures" / name).stat().st_size > 1000
    print("PASS: N=5 directed lower bounds and 15 domain-intervention runs validated")

if __name__ == "__main__":
    main()
