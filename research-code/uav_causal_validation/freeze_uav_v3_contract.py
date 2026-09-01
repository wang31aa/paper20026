#!/usr/bin/env python3
"""Create the immutable V3 parameter contract and its content hash."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
protocol_bytes = (HERE / "UAV_V3_PREREGISTRATION.json").read_bytes()
cal_bytes = (OUT / "uav_v3_development_calibration.json").read_bytes()
cal = json.loads(cal_bytes)
if cal["heldout_seed_count_read"] != 0: raise SystemExit("refuse freeze: heldout data were read")
payload = {"schema_version": "3.0", "status": "FROZEN_BEFORE_HELDOUT",
           "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
           "calibration_sha256": hashlib.sha256(cal_bytes).hexdigest(),
           "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
           "rho_grid": json.loads(protocol_bytes)["rho_grid"],
           "modes": cal["modes"], "epsilon_peak": cal["epsilon_peak"],
           "epsilon_ultimate": cal["epsilon_ultimate"],
           "development_seeds": cal["development_seeds"]}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
(OUT / "UAV_V3_FROZEN_CONTRACT.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
