#!/usr/bin/env python3
"""Execute the frozen CERT-OIL-R4 mechanism scan."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run_benchmark as bench

ALPHAS = (1.5, 2.0, 4.0, 8.0)
GAMMAS = (2.0, 5.0, 15.0, 35.0)
SEEDS = (0, 1, 2)


def main() -> None:
    root = Path(__file__).resolve().parent
    protocol = root / "PREREGISTRATION_R4.md"
    out = root / "results_r4"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir()
    rows: list[dict] = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            bench.ALPHA = alpha
            bench.GAMMA_STATE = gamma
            for topology, weights in bench.topologies().items():
                for heterogeneity in bench.LEVELS:
                    for seed in SEEDS:
                        run, _ = bench.simulate(
                            topology, weights, heterogeneity, seed, store=False
                        )
                        row = dict(run.__dict__)
                        row.update(alpha=alpha, gamma_state=gamma)
                        rows.append(row)
    fields = list(rows[0])
    with (out / "raw_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    finite = np.array([str(r["finite"]).lower() == "true" for r in rows])
    env = np.array([float(r["max_tracking_envelope_ratio"]) for r in rows])
    obs = np.array([float(r["max_observer_envelope_ratio"]) for r in rows])
    summary = {
        "protocol_id": "CERT-OIL-R4",
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "runs": len(rows),
        "finite_runs": int(finite.sum()),
        "max_tracking_envelope_ratio": float(env.max()),
        "max_observer_envelope_ratio": float(obs.max()),
        "tracking_envelope_violations": int(np.sum(env > 1 + 1e-10)),
        "observer_envelope_violations": int(np.sum(obs > 1 + 1e-10)),
        "design": {
            "alphas": ALPHAS,
            "gamma_state": GAMMAS,
            "heterogeneity": bench.LEVELS,
            "topologies": list(bench.topologies()),
            "seeds": SEEDS,
            "dt": bench.DT,
            "t_end": bench.T_END,
        },
    }
    (out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
