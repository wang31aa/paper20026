#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from python_translation.pf import PFParameters, free_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "pf_source_replay_metrics.json")
    args = parser.parse_args()
    raw = loadmat(args.workspace, simplify_cells=True)
    s = raw["S"]
    times = np.asarray(raw["time_history"], dtype=float).reshape(-1)
    recorded_p = np.asarray(raw["pos_history"], dtype=float).reshape(len(times), int(s["nb_agents"]), 3)
    recorded_v = np.asarray(raw["vel_history"], dtype=float).reshape(len(times), int(s["nb_agents"]), 3)
    cylinders = np.asarray(s["cylinders"], dtype=float).reshape(3, -1).T
    p = PFParameters(float(s["r"]), int(s["max_neig"]), float(s["r_coll"]), cylinders,
                     np.asarray(s["u_migration"], dtype=float).reshape(3), float(s["v_swarm"]),
                     float(s["max_a"]), float(s["max_v"]), float(s["r0_rep"]), float(s["p_rep"]),
                     float(s["r0_fric"]), float(s["C_fric"]), float(s["v_fric"]), float(s["p_fric"]),
                     float(s["a_fric"]), float(s["r0_shill"]), float(s["v_shill"]),
                     float(s["p_shill"]), float(s["a_shill"]))
    dt = float(np.median(np.diff(times)))
    # Recorded completion supplies the exact number of transitions. The free run
    # itself receives no recorded intermediate states or controls.
    translated_p, translated_v, translated_u, events = free_run(
        recorded_p[0], recorded_v[0], p, dt, len(times) - 1, np.inf)
    position_rmse = float(np.sqrt(np.mean((translated_p - recorded_p) ** 2)))
    velocity_rmse = float(np.sqrt(np.mean((translated_v - recorded_v) ** 2)))
    position_range = float(np.ptp(recorded_p))
    normalized_position_rmse = position_rmse / position_range
    min_pair_recorded = min(np.linalg.norm(frame[i] - frame[j])
                            for frame in recorded_p for i in range(p.maximum_neighbours + 2)
                            for j in range(i + 1, p.maximum_neighbours + 2))
    min_pair_translated = min(np.linalg.norm(frame[i] - frame[j])
                              for frame in translated_p for i in range(p.maximum_neighbours + 2)
                              for j in range(i + 1, p.maximum_neighbours + 2))
    speed_recorded = np.linalg.norm(recorded_v, axis=2)
    speed_translated = np.linalg.norm(translated_v, axis=2)
    control_recorded = np.diff(recorded_v, axis=0) / dt
    control_translated = np.diff(translated_v, axis=0) / dt
    energy_recorded = float(np.sum(control_recorded**2) * dt)
    energy_translated = float(np.sum(control_translated**2) * dt)
    criteria = {
        "position_rmse": bool(position_rmse <= 0.10),
        "normalized_position_rmse": bool(normalized_position_rmse <= 0.05),
        "minimum_pair_distance": bool(abs(min_pair_translated-min_pair_recorded) <= max(.05, .05*min_pair_recorded)),
        "mean_speed": bool(abs(float(speed_translated.mean()-speed_recorded.mean())) <= max(.05, .05*float(speed_recorded.mean()))),
        "p95_speed": bool(abs(float(np.percentile(speed_translated,95)-np.percentile(speed_recorded,95))) <= max(.05,.05*float(np.percentile(speed_recorded,95)))),
        "control_integral": bool(abs(energy_translated-energy_recorded) <= .1*max(energy_recorded,1e-12)),
        "collision_event": bool(sum(e["agent_collisions"]+e["obstacle_collisions"] for e in events) == 0),
    }
    report = {"workspace": str(args.workspace), "source_tag": "v1.0",
              "source_commit": "377ffd75b4581d642881f3d349a03cbc1a26e24d",
              "source_archive_sha256": "aa21629dd4e8a78d38218e9dcbe033c4778c84fe30fd78246ce735458d84d2c6",
              "free_running": True, "recorded_intermediate_state_used": False,
              "metrics": {"position_rmse": position_rmse, "normalized_position_rmse": normalized_position_rmse,
                          "velocity_rmse": velocity_rmse, "minimum_pair_recorded": min_pair_recorded,
                          "minimum_pair_translated": min_pair_translated,
                          "mean_speed_recorded": float(speed_recorded.mean()),
                          "mean_speed_translated": float(speed_translated.mean()),
                          "p95_speed_recorded": float(np.percentile(speed_recorded,95)),
                          "p95_speed_translated": float(np.percentile(speed_translated,95)),
                          "control_integral_recorded": energy_recorded,
                          "control_integral_translated": energy_translated},
              "criteria": criteria, "qualified": all(criteria.values())}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
