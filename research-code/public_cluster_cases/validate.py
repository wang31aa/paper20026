#!/usr/bin/env python3
"""Validate dataset-informed mobile-cluster outputs and evidence labels."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    provenance = json.loads((HERE / "results/provenance.json").read_text())
    with (HERE / "results/summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    assert provenance["protocol"] == "PUBLIC-CLUSTER-CASES-R1"
    assert provenance["classification"] == "dataset-informed computational validation"
    assert [row["case"] for row in summary] == ["robot", "uav", "vehicle"]
    assert [int(row["n_agents"]) for row in summary] == [5, 20, 3]
    assert all(row["envelope_pass"] == "True" for row in summary)
    assert all(float(row["max_positive_time_envelope_ratio"]) <= 1 for row in summary)
    files = {
        "robot": HERE / "cache/robot_swarm_validation.csv",
        "uav": HERE / "cache/uav_swarm_synthetic.csv",
        "vehicle": HERE / "cache/adas_two_vehicle_sample.csv",
    }
    # Raw public-data caches are intentionally absent from the rights-filtered
    # GitHub candidate. Validate their frozen hashes whenever they are present;
    # otherwise validate the derived traces and retained provenance only.
    for case, path in files.items():
        if path.exists():
            assert provenance["sources"][case]["sha256"] == sha256(path)
        trace = HERE / f"results/{case}_trace.csv"
        assert trace.is_file() and trace.stat().st_size > 1000
    assert "not stated" in provenance["sources"]["uav"]["license"]
    assert (ROOT / "figures/Fig15_public_cluster_cases.svg").is_file()
    source_status = "source hashes" if all(path.exists() for path in files.values()) else "retained provenance"
    print(f"PASS: 3 dataset-informed cluster cases, {source_status} and envelopes validated")


if __name__ == "__main__":
    main()
