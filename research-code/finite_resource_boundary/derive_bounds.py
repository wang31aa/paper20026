#!/usr/bin/env python3
"""Compute matched finite-resource bounds on retained N=5 graph families."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def pinned_laplacians(n: int = 5) -> dict[str, np.ndarray]:
    chain = np.zeros((n, n))
    chain[0, 0] = 1.0
    for i in range(1, n):
        chain[i, i] = 1.0
        chain[i, i - 1] = -1.0
    direct = np.eye(n)
    cyclic = chain.copy()
    cyclic[0, 0] += 0.25
    cyclic[0, -1] = -0.25
    cyclic[2, 2] += 0.4
    cyclic[2, 0] = -0.4
    return {"directed_chain": chain, "direct_pinning": direct,
            "directed_cyclic": cyclic}


def graph_metric(lap: np.ndarray) -> tuple[np.ndarray, float]:
    g = np.linalg.solve(lap.T, np.ones(lap.shape[0]))
    root = np.diag(g ** -0.5)
    sym = np.diag(g) @ lap + lap.T @ np.diag(g)
    return g, float(np.linalg.eigvalsh(root @ sym @ root).min())


def rows() -> list[dict[str, float | str]]:
    out: list[dict[str, float | str]] = []
    lam, alpha, disturbance = 0.2, 4.0, 1.0
    target_speed, update_period, bits, target_half_range = 0.15, 0.2, 6, 2.0
    for name, lap in pinned_laplacians().items():
        g, mu = graph_metric(lap)
        a = lam * np.eye(len(lap)) + alpha * lap
        # Match the theorem's weighted forcing contract:
        # ||G^{1/2}d||_2 <= D and ||G^{1/2}u||_2 <= U.
        # The exact steady minimax value is (D-U)_+ ||A^{-1}G^{-1/2}||_2.
        weighted_gain = np.linalg.norm(
            np.linalg.solve(a, np.diag(g ** -0.5)), 2)
        upper = 2.0 * disturbance / ((alpha * mu + 2.0 * lam) * np.sqrt(g.min()))
        for authority_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            authority = authority_fraction * disturbance
            act_lower = max(disturbance - authority, 0.0) * weighted_gain
            sample_lower = target_speed * update_period
            bit_lower = target_half_range * 2.0 ** (-bits)
            out.append({
                "topology": name,
                "nodes": len(lap),
                "lambda": lam,
                "alpha": alpha,
                "graph_mu": mu,
                "disturbance_norm": disturbance,
                "control_authority": authority,
                "upper_radius": upper,
                "weighted_steady_gain": weighted_gain,
                "actuation_lower": act_lower,
                "sampled_information_lower": sample_lower,
                "quantization_lower": bit_lower,
                "actuation_upper_lower_gap": (upper / act_lower
                                                if act_lower > 0 else ""),
            })
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    data = rows()
    with (RESULTS / "finite_resource_bounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
    summary = {
        "rows": len(data),
        "topologies": sorted({str(row["topology"]) for row in data}),
        "minimum_positive_actuation_gap": min(float(row["actuation_upper_lower_gap"])
                                                for row in data
                                                if row["actuation_upper_lower_gap"] != ""),
        "maximum_positive_actuation_gap": max(float(row["actuation_upper_lower_gap"])
                                                for row in data
                                                if row["actuation_upper_lower_gap"] != ""),
        "decision": "ultimate actuation and peak information bounds kept separate",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
