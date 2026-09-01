#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RHO_MIN, RHO_MAX, N = 0.12, 4.0, 240
SCENARIOS = {
    "interior_window": dict(lam=1.0, ell=0.30, D=2.0, U=0.15, V=0.55, h=0.20, eps_p=0.88, eps_u=1.20),
    "monotone_benefit": dict(lam=1.0, ell=1.0, D=0.0, U=0.0, V=0.55, h=0.20, eps_p=0.88, eps_u=1.20),
    "topology_dominant": dict(lam=1.0, ell=2.5, D=0.30, U=0.05, V=0.25, h=0.20, eps_p=0.45, eps_u=0.55),
}


def log_grid(lo, hi, n):
    ratio = (hi / lo) ** (1.0 / (n - 1))
    return [lo * ratio ** k for k in range(n)]


def evaluate(rho, p):
    a = p["lam"] + rho * p["ell"]
    residual_input = max(rho * p["D"] - p["U"], 0.0)
    r = residual_input / a
    b = p["V"] * p["h"] / rho
    peak = r + b
    ultimate = r
    # Exact scalar flow over many update intervals. q resets at each update;
    # the supremum is the left limit at the final update.
    period = p["h"] / rho
    phi = math.exp(-a * period)
    z = 0.0
    numerical_peak = 0.0
    for _ in range(400):
        z_left = phi * z + (1.0 - phi) * r
        numerical_peak = max(numerical_peak, abs(z_left) + b)
        z = z_left
    numerical_ultimate = abs(z)
    psi = max(peak / p["eps_p"], ultimate / p["eps_u"])
    return r, b, peak, ultimate, numerical_peak, numerical_ultimate, psi


def main():
    rows, summary = [], {}
    grid = log_grid(RHO_MIN, RHO_MAX, N)
    for name, p in SCENARIOS.items():
        vals = []
        for rho in grid:
            result = evaluate(rho, p)
            vals.append((rho, *result))
            rows.append({
                "scenario": name, "rho": rho, "residual_radius": result[0],
                "information_age_radius": result[1], "analytic_peak": result[2],
                "analytic_update_ultimate": result[3], "numeric_peak": result[4],
                "numeric_update_ultimate": result[5], "capability_function": result[6],
                "feasible": int(result[6] <= 1.0),
            })
        best = min(vals, key=lambda x: x[-1])
        feasible = [v[0] for v in vals if v[-1] <= 1.0]
        err = max(max(abs(v[3] - v[5]), abs(v[4] - v[6])) for v in vals)
        summary[name] = {
            "rho_at_minimum": best[0], "minimum_capability": best[-1],
            "feasible_rho_min": min(feasible) if feasible else None,
            "feasible_rho_max": max(feasible) if feasible else None,
            "maximum_exact_flow_error": err,
        }
    with (ROOT / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert all(v["maximum_exact_flow_error"] < 1e-10 for v in summary.values())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
