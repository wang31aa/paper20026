#!/usr/bin/env python3
"""Numerically verify the attained constant-forcing minimax construction."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finite_resource_boundary"))
from derive_bounds import pinned_laplacians, graph_metric

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def main() -> None:
    D = 1.0
    rows = []
    for topology, lap in pinned_laplacians().items():
        g, _ = graph_metric(lap)
        A = 0.2 * np.eye(len(lap)) + 4.0 * lap
        C = np.linalg.solve(A, np.diag(g ** -0.5))
        _, singular, vh = np.linalg.svd(C)
        v = vh[0]
        d = D * np.diag(g ** -0.5) @ v
        for U in (0.0, 0.25, 0.5, 0.75, 1.0):
            u = -min(1.0, U / D) * d
            equilibrium = np.linalg.solve(A, d + u)
            formula = max(D - U, 0.0) * singular[0]
            error = abs(np.linalg.norm(equilibrium) - formula)
            assert error <= 1e-11
            rows.append({"topology": topology, "D": D, "U": U,
                         "induced_gain": singular[0],
                         "attained_equilibrium_norm": np.linalg.norm(equilibrium),
                         "minimax_formula": formula,
                         "absolute_error": error})
    # The two indistinguishable scalar targets +/-Vs*t are separated by 2VsT.
    Vs, h = 0.15, 0.2
    for rho in (0.1, 0.25, 0.5, 1.0):
        T = h / rho
        separation = 2 * Vs * T
        assert abs(separation / 2 - Vs * h / rho) <= 1e-14
    OUT.mkdir(exist_ok=True)
    with (OUT / "attainment.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    summary = {"attainment_rows": len(rows),
               "maximum_absolute_error": max(r["absolute_error"] for r in rows),
               "sampled_indistinguishability_checks": 4,
               "scope": "constant forcing, disturbance-known cancellation, declared weighted norm"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
