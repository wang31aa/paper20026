#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
import numpy as np
from uav_v3_engine import P, simulate, adjacency

H = Path(__file__).resolve().parent
O = H / "results"
Q = json.loads((H / "UAV_V5_PREREGISTRATION.json").read_text())
rows = []
for env in P["development_environments"]:
    for seed in Q["development_seeds"]:
        for rho in Q["rho_grid"]:
            rows.append(simulate(
                "connectivity_constrained_observer_gate", rho, env, seed,
                "all_heterogeneous", Q["observer_mode"],
                Q["command_reserve_mps2"], Q["max_missed_updates"]))
with (O / "uav_v5_development_runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

z = Q["development_quantile"]
Ep = float(np.quantile([r["peak_post_reset_position_error_m"] for r in rows], z))
Ev = float(np.quantile([r["peak_post_reset_velocity_error_mps"] for r in rows], z))
Vph = float(np.quantile([r["rho"] * r["peak_position_growth_since_reset_m"] for r in rows], z))
Vvh = float(np.quantile([r["rho"] * r["peak_velocity_growth_since_reset_mps"] for r in rows], z))
trim = np.asarray(P["declared_trim_acceleration_mps2"]) * 1.05
D0 = float(np.max(np.linalg.norm(trim, axis=1)))
Db = float(P["trim_sharing_gain"] * np.max(np.linalg.norm(adjacency() @ trim, axis=1)))
U = Q["certified_acceleration_reserve_mps2"]
gain = {(float(r["rho"]), int(r["mode"])): r for r in csv.DictReader((O / "uav_v31_output_gains.csv").open())}
crit, parts = [], []
for rho in Q["rho_grid"]:
    bounds = []
    for m in range(4):
        g = gain[(rho, m)]; K0 = float(g["K0_s2"]); Kp = float(g["Kpeak_s2"])
        R = max(D0 + rho * Db - U, 0.0)
        force = Kp * R
        position = Ep + Vph / rho
        velocity = Kp * 1.45 * (Ev + Vvh / rho)
        peak = force + position + velocity
        ultimate = K0 * R
        psi = max(peak / Q["epsilon_peak_m"], ultimate / Q["epsilon_ultimate_m"])
        bounds.append((psi, m, peak, ultimate))
        parts.append({"rho": rho, "mode": m, "force_m": force,
                      "position_info_m": position, "velocity_info_m": velocity,
                      "peak_m": peak, "ultimate_m": ultimate})
    worst = max(bounds)
    crit.append({"rho": rho, "psi": worst[0], "active_mode": worst[1],
                 "peak_bound_m": worst[2], "ultimate_bound_m": worst[3],
                 "predicted_feasible": int(worst[0] <= 1)})
for name, data in (("uav_v5_term_breakdown.csv", parts), ("uav_v5_critical_function.csv", crit)):
    with (O / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
feas = [r for r in crit if r["predicted_feasible"]]
low = [r for r in crit if feas and not r["predicted_feasible"] and r["rho"] < min(x["rho"] for x in feas)]
high = [r for r in crit if feas and not r["predicted_feasible"] and r["rho"] > max(x["rho"] for x in feas)]
C = {
    "schema_version": "5.0", "status": "FROZEN_BEFORE_V5_HELDOUT",
    "heldout_seed_count_read": 0, "observer_mode": Q["observer_mode"],
    "max_missed_updates": Q["max_missed_updates"],
    "Ep_m": Ep, "Vph_m": Vph, "Ev_mps": Ev, "Vvh_mps": Vvh,
    "D0_mps2": D0, "Db_mps2": Db, "U_mps2": U,
    "critical_function": crit, "feasible_rho": [r["rho"] for r in feas],
    "low_witness": low[-1]["rho"] if low else None,
    "interior_witness": min(feas, key=lambda r: r["psi"])["rho"] if feas else None,
    "high_witness": high[0]["rho"] if high else None,
    "heldout_authorized": bool(feas and low and high)
}
raw = json.dumps(C, sort_keys=True, separators=(",", ":")).encode()
C["contract_sha256"] = hashlib.sha256(raw).hexdigest()
(O / "UAV_V5_FROZEN_CONTRACT.json").write_text(json.dumps(C, indent=2) + "\n")
print(json.dumps(C, indent=2))
