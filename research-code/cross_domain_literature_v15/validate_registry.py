#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
d = json.loads(Path(__file__).with_name("DOMAIN_EVIDENCE_REGISTRY.json").read_text())
expected = {"robot", "uav", "vehicle", "motor", "circuit", "water", "structure", "microgrid"}
assert set(d["domains"]) == expected
assert "only across policies within the same domain" in d["principle"]
for name, item in d["domains"].items():
    for rel in item["source_objects"]:
        assert (root / rel).exists(), (name, rel)
    assert item["permitted_simulation_label"] and item["parameter_identification_status"]
assert all(item["parameter_identification_status"] != "FULL" for item in d["domains"].values())
print("PASS: eight domain-specific evidence contracts validated; zero full literature reproductions")
