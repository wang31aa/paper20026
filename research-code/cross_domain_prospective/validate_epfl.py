#!/usr/bin/env python3
import csv, json, math
from pathlib import Path

here = Path(__file__).resolve().parent
rows = list(csv.DictReader((here / "results/epfl_run_qualification.csv").open()))
excluded = list(csv.DictReader((here / "results/epfl_exclusions.csv").open()))
summary = json.loads((here / "results/epfl_qualification_summary.json").read_text())
assert summary["mat_files"] == 184
assert len(rows) == 183 == summary["records"]
assert len(excluded) == 1 == summary["excluded_records"]
assert len({r["path"] for r in rows}) == 183
assert not ({r["path"] for r in rows} & {r["path"] for r in excluded})
assert sum(r["study"] == "hitl" for r in rows) == 1
for r in rows:
    for key in ("samples", "agents", "collision_radius_source", "minimum_pair_separation_source",
                "separation_margin_source", "maximum_speed_source", "declared_max_speed_source"):
        assert math.isfinite(float(r[key]))
    assert int(r["agents"]) == 5
    assert r["split"] in {"calibration", "validation", "retrospective_test", "transport_only"}
assert summary["claim_boundary"].endswith("no proposed-gate intervention")
print("PASS: 184 EPFL MAT files audited; 183 eligible records; 1 exclusion retained")
