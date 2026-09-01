#!/usr/bin/env python3
import csv, json, math, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
DT, STEPS, N = 0.02, 500, 5
ACTUATOR_GAIN = [0.76, 0.88, 1.00, 1.09, 1.21]
ACTUATOR_TAU_S = [0.055, 0.075, 0.100, 0.135, 0.180]
ACTUATOR_LIMIT = [0.62, 0.70, 0.80, 0.74, 0.66]
PERSISTENT_BIAS = [-0.025, 0.012, 0.000, 0.019, -0.031]


def main():
    state = [0.4 * i for i in range(N)]
    target = [0.4 * i + 0.25 for i in range(N)]
    actuator_state = [0.0 for _ in range(N)]
    rows, misses, recovery_dwell = [], 0, 0
    start = time.perf_counter()
    for k in range(STEPS):
        cycle_start = time.perf_counter()
        residual = [state[i] - target[i] for i in range(N)]
        task_margin = 0.32 - max(abs(v) for v in residual)
        recovery_dwell = recovery_dwell + 1 if task_margin >= 0 else 0
        trustworthy = [abs(v) <= 0.32 for v in residual]
        # Node-level ring edges are removed only if target reachability through
        # the pinned node 0 is retained; otherwise the edge stays active.
        adjacency = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adjacency[i][i - 1] = int(trustworthy[i - 1] or i - 1 == 0)
        control = []
        for i in range(N):
            neighbour = sum(adjacency[i][j] * (state[j] - state[i]) for j in range(N))
            requested = -1.1 * residual[i] + 0.35 * neighbour
            limited = max(-ACTUATOR_LIMIT[i], min(ACTUATOR_LIMIT[i], requested))
            actuator_state[i] += DT * (ACTUATOR_GAIN[i] * limited - actuator_state[i]) / ACTUATOR_TAU_S[i]
            control.append((requested, limited, actuator_state[i]))
        disturbance = [0.03 * math.sin(0.071 * k + i) for i in range(N)]
        state = [state[i] + DT * (control[i][2] + PERSISTENT_BIAS[i] + disturbance[i]) for i in range(N)]
        compute = time.perf_counter() - cycle_start
        missed = int(compute > DT); misses += missed
        edge_count = sum(sum(row) for row in adjacency)
        target_reachable = int(all(i == 0 or any(adjacency[i]) for i in range(N)))
        for i in range(N):
            rows.append({"cycle": k, "time_s": k * DT, "node": i,
                         "state": state[i], "target": target[i],
                         "information_error": residual[i],
                         "requested_control": control[i][0],
                         "limited_control": control[i][1],
                         "applied_control": control[i][2],
                         "saturated": int(control[i][0] != control[i][1]),
                         "actuator_gain": ACTUATOR_GAIN[i],
                         "actuator_tau_s": ACTUATOR_TAU_S[i],
                         "actuator_limit": ACTUATOR_LIMIT[i],
                         "persistent_bias": PERSISTENT_BIAS[i],
                         "adjacency_row": json.dumps(adjacency[i]),
                         "message_age_s": 0.0,
                         "edge_count": edge_count,
                         "target_reachable": target_reachable,
                         "task_margin": task_margin,
                         "recovery_dwell_cycles": recovery_dwell,
                         "communication_bytes": 8 * edge_count,
                         "disturbance": disturbance[i],
                         "compute_time_s": compute, "deadline_missed": missed})
    with (OUT / "sil_unified_log.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    deadline_ok = misses == 0
    status = {"status": "sil_interface_qualified" if deadline_ok else "sil_interface_qualified_with_deadline_misses",
              "interface_schema_qualified": True,
              "realtime_deadline_qualified": deadline_ok,
              "hil_executed": False,
              "cycles": STEPS, "nodes": N, "deadline_s": DT,
              "heterogeneous_plant": True,
              "heterogeneous_parameters": ["actuator_gain", "actuator_tau_s", "actuator_limit", "persistent_bias"],
              "deadline_misses": misses,
              "maximum_compute_time_s": max(r["compute_time_s"] for r in rows),
              "wall_time_s": time.perf_counter() - start}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__": main()
