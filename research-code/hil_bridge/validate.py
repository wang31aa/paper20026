#!/usr/bin/env python3
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
status = json.loads((HERE / "results/status.json").read_text())
rows = list(csv.DictReader((HERE / "results/sil_unified_log.csv").open()))
assert status["status"] in {"sil_interface_qualified", "sil_interface_qualified_with_deadline_misses"}
assert status["interface_schema_qualified"] is True
assert status["realtime_deadline_qualified"] == (status["deadline_misses"] == 0)
assert status["hil_executed"] is False
assert status["heterogeneous_plant"] is True
assert len(rows) == status["cycles"] * status["nodes"]
assert all(len(json.loads(r["adjacency_row"])) == status["nodes"] for r in rows)
assert all(float(r["compute_time_s"]) >= 0 for r in rows)
assert all(int(r["target_reachable"]) == 1 for r in rows)
assert all(float(r["message_age_s"]) >= 0 for r in rows)
assert all(int(r["communication_bytes"]) >= 0 for r in rows)
for field in ("actuator_gain", "actuator_tau_s", "actuator_limit", "persistent_bias"):
    assert len({float(r[field]) for r in rows}) > 1, field
assert any(abs(float(r["limited_control"])-float(r["applied_control"])) > 1e-9 for r in rows)
print("PASS: heterogeneous SIL schema, intervention and timing fields are internally consistent; deadline and HIL promotion remain fail-closed")
