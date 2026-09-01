#!/usr/bin/env python3
"""Fail closed on the Source Data added for Extended Data 7, 8 and Table 1."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream))
    if not data:
        raise AssertionError(f"empty table: {path}")
    return data


def main() -> None:
    curve = rows(ROOT / "cross_domain_v43_response_regimes/results/v43_curve_summary.csv")
    regimes = rows(ROOT / "cross_domain_v43_response_regimes/results/v43_regime_summary.csv")
    assert {"domain", "rho", "successes", "runs", "success_rate", "wilson_low", "wilson_high"} <= set(curve[0])
    assert {"domain", "policy", "response_class"} <= set(regimes[0])
    for row in curve:
        successes, runs, rate = int(row["successes"]), int(row["runs"]), float(row["success_rate"])
        assert 0 <= successes <= runs and abs(rate - successes / runs) < 1e-12

    required_v47 = {"rho", "policy", "task_success", "minimum_margin_m", "control_energy", "communication_messages"}
    development = rows(ROOT / "vehicle_v47_dense/results/V47_DEVELOPMENT.csv")
    heldout = rows(ROOT / "vehicle_v47_dense/results/V47_HELDOUT.csv")
    assert required_v47 <= set(development[0]) and required_v47 <= set(heldout[0])
    frozen = json.loads((ROOT / "vehicle_v47_dense/results/V47_FROZEN_PREDICTIONS.json").read_text(encoding="utf-8"))
    assert "contract_sha256" in frozen and "policies" in frozen

    table = rows(ROOT / "figures/source_data/Extended_Data_Table1_graph_constants.csv")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in table:
        grouped[row["topology"]].append(float(row["d"]))
    displayed = {
        "Chain": (4.223945, 4.883945),
        "Star": (10.008000, 10.668000),
        "Branch": (5.517613, 6.177613),
        "Cyclic": (3.971773, 4.631773),
    }
    assert set(grouped) == set(displayed)
    for topology, values in grouped.items():
        assert len(values) == 3
        assert abs(min(values) - displayed[topology][0]) < 5e-7
        assert abs(max(values) - displayed[topology][1]) < 5e-7

    print(
        "PASS: Extended Data 7, 8 and Table 1 Source Data validated "
        f"({len(curve)} V43 rows, {len(development) + len(heldout)} V47 rows, {len(table)} table rows)"
    )


if __name__ == "__main__":
    main()
