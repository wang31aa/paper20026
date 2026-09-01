"""Deterministic V3 UAV surrogate shared by calibration and held-out runs."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "UAV_V3_PREREGISTRATION.json").read_text())
N = 5; DT = float(P["dt_s"]); STEPS = int(P["duration_s"] / DT)


def adjacency(drop_node: int | None = None) -> np.ndarray:
    A = np.zeros((N, N))
    for i in range(N): A[i, (i - 1) % N] = A[i, (i + 1) % N] = 1
    if drop_node is not None:
        A[drop_node, :] = 0; A[:, drop_node] = 0
        # deterministic connectivity restoration through the target-visible node 0
        A[0, drop_node] = A[drop_node, 0] = 1
    return A


def innovations(seed: int, env: dict) -> dict:
    g = np.random.default_rng(seed)
    return {"wind": g.normal(0, env["wind_sd"], (STEPS, N, 2)),
            "pn": g.normal(0, .005, (STEPS, N, 2)),
            "vn": g.normal(0, .008, (STEPS, N, 2)),
            "delivery": g.random((STEPS, N, N)) >= env["loss"]}


def environment(name: str) -> dict:
    return {
        "open_nominal": {"wind_sd": .012, "loss": .01, "delay": 0, "payload_scale": 1., "drop": None},
        "moderate_loss": {"wind_sd": .018, "loss": .08, "delay": 1, "payload_scale": 1., "drop": None},
        "moderate_crosswind": {"wind_sd": .026, "loss": .03, "delay": 1, "payload_scale": 1.05, "drop": None},
        "bursty_loss": {"wind_sd": .020, "loss": .16, "delay": 1, "payload_scale": 1., "drop": None},
        "delayed_crosswind": {"wind_sd": .035, "loss": .05, "delay": 3, "payload_scale": 1.05, "drop": None},
        "payload_shift": {"wind_sd": .018, "loss": .04, "delay": 1, "payload_scale": 1.18, "drop": None},
        "topology_dropout": {"wind_sd": .022, "loss": .10, "delay": 2, "payload_scale": 1., "drop": 3}
    }[name] | {"name": name}


def safety_projection(u: np.ndarray, p: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, float]:
    raw = u.copy(); dmin = P["task_contract"]["minimum_pair_clearance_m"]
    for _ in range(4):
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]; q = np.linalg.norm(d)
                if q < .52 and q > 1e-9:
                    n = d / q; need = 4.0 * (dmin - q) - 1.8 * np.dot(v[i] - v[j], n)
                    gap = need - np.dot(u[i] - u[j], n)
                    if gap > 0: u[i] += .5 * gap * n; u[j] -= .5 * gap * n
        W = P["task_contract"]["corridor_half_width_m"]
        for i in range(N):
            if abs(p[i, 1]) > .85 * W:
                u[i, 1] -= math.copysign(3.0 * (abs(p[i, 1]) - .85 * W) + 1.4 * abs(v[i, 1]), p[i, 1])
    lim = 2.; norms = np.linalg.norm(u, axis=1); over = norms > lim
    u[over] *= (lim / norms[over])[:, None]
    return u, float(np.max(np.linalg.norm(u - raw, axis=1)))


def simulate(policy: str, rho: float, env_name: str, seed: int, arm: str = "all_heterogeneous",
             observer_mode: str = "legacy", cancellation_reserve: float = 0.0,
             max_missed_updates: int | None = None,
             base_update_interval_s: float | None = None,
             target_visibility: str = "all",
             target_manoeuvre_amplitude_m: float = 0.0,
             target_observer_gain: float = 0.8) -> dict:
    env = environment(env_name); inn = innovations(seed, env)
    yref = np.linspace(-1.24, 1.24, N); p = np.c_[np.zeros(N), yref]; v = np.zeros((N, 2)); acc = np.zeros((N, 2))
    beta = np.array([.88, .94, 1., 1.07, 1.14]); drag = np.array([.04, .06, .08, .10, .12]); tau = np.array([.22, .30, .40, .56, .72])
    trim = np.asarray(P["declared_trim_acceleration_mps2"], float) * env["payload_scale"]
    if arm == "homogeneous": beta[:] = 1; drag[:] = .08; tau[:] = .40; trim[:] = 0
    elif arm == "effectiveness_only": drag[:] = .08; tau[:] = .40; trim[:] = 0
    elif arm == "drag_only": beta[:] = 1; tau[:] = .40; trim[:] = 0
    elif arm == "lag_only": beta[:] = 1; drag[:] = .08; trim[:] = 0
    elif arm == "trim_only": beta[:] = 1; drag[:] = .08; tau[:] = .40
    A = adjacency(env["drop"]); phat = np.repeat(p[None, :, :], N, axis=0); vhat = np.zeros((N, N, 2)); dhat = np.zeros((N, N, 2))
    ahat = np.zeros((N, N, 2)); uhat = np.zeros((N, N, 2)); previous_u = np.zeros((N, 2))
    missed_updates = np.zeros((N, N), dtype=int); maximum_scheduled_misses = 0
    target_position_hat = np.zeros(N); target_velocity_hat = np.zeros(N)
    base_update = P["base_update_interval_s"] if base_update_interval_s is None else base_update_interval_s
    residual = np.zeros(N); update_steps = max(1, round(base_update / (rho * DT)))
    p_hist = [p.copy() for _ in range(env["delay"] + 1)]; v_hist = [v.copy() for _ in range(env["delay"] + 1)]
    a_hist = [acc.copy() for _ in range(env["delay"] + 1)]; u_hist = [previous_u.copy() for _ in range(env["delay"] + 1)]
    min_pair = np.inf; max_corridor = peak_err = tail_err = 0.; energy = comm = safety_cost = cancellation_energy = 0.; gate_steps = 0
    peak_position_observer_error = 0.; peak_velocity_observer_error = 0.
    peak_target_observer_error = 0.; tail_target_observer_error = 0.
    peak_post_reset_position_error = 0.; peak_post_reset_velocity_error = 0.
    peak_position_growth_since_reset = 0.; peak_velocity_growth_since_reset = 0.
    reset_position_baseline = np.zeros((N, N)); reset_velocity_baseline = np.zeros((N, N))
    fault_end = P["fault"]["end_s"]; recovered_at = None; dwell = 0.; tail_start = int(.8 * STEPS)
    for k in range(STEPS):
        t = k * DT; s = min(t / P["duration_s"], 1.)
        base_speed = (P["task_contract"]["terminal_reference_overshoot_m"] + 16.) / P["duration_s"]
        refx = (P["task_contract"]["terminal_reference_overshoot_m"] + 16.) * s + target_manoeuvre_amplitude_m * math.sin(4 * math.pi * s)
        refvx = (base_speed + target_manoeuvre_amplitude_m * 4 * math.pi / P["duration_s"] * math.cos(4 * math.pi * s)) if t < P["duration_s"] else 0.
        pref = np.c_[np.full(N, refx), yref]; vref = np.c_[np.full(N, refvx), np.zeros(N)]
        if target_visibility == "target_rooted":
            target_position_hat += DT * target_velocity_hat
            target_position_hat[0] = refx; target_velocity_hat[0] = refvx
        else:
            target_position_hat[:] = refx; target_velocity_hat[:] = refvx
        if observer_mode == "model_based_bounded_age":
            for j in range(N):
                aj = math.exp(-DT / tau[j])
                ahat[:, j] = aj * ahat[:, j] + (1 - aj) * uhat[:, j]
                vhat[:, j] += DT * (beta[j] * ahat[:, j] - drag[j] * vhat[:, j] + dhat[:, j])
                phat[:, j] += DT * vhat[:, j]
        else:
            phat += DT * vhat
        p_hist.append(p + inn["pn"][k]); p_hist.pop(0); v_hist.append(v + inn["vn"][k]); v_hist.pop(0)
        a_hist.append(acc.copy()); a_hist.pop(0); u_hist.append(previous_u.copy()); u_hist.pop(0)
        if k % update_steps == 0:
            target_messages = [[] for _ in range(N)]
            for i in range(N):
                for j in np.where(A[i] > 0)[0]:
                    delivered = bool(inn["delivery"][k, i, j])
                    if max_missed_updates is not None and missed_updates[i, j] >= max_missed_updates:
                        delivered = True
                    if delivered:
                        missed_updates[i, j] = 0
                        ip = p_hist[0][j] - phat[i, j]; iv = v_hist[0][j] - vhat[i, j]
                        residual[j] = .72 * residual[j] + .28 * np.linalg.norm(np.r_[ip, iv])
                        if observer_mode == "model_based_bounded_age":
                            # Reset every transmitted physical state at its time stamp,
                            # then replay the known actuator and heterogeneous plant
                            # through the declared transport delay.
                            pp = p_hist[0][j].copy(); vv = v_hist[0][j].copy()
                            aa = a_hist[0][j].copy(); uu = u_hist[0][j].copy()
                            for _ in range(env["delay"]):
                                aj = math.exp(-DT / tau[j])
                                aa = aj * aa + (1 - aj) * uu
                                vv += DT * (beta[j] * aa - drag[j] * vv + trim[j])
                                pp += DT * vv
                            phat[i, j] = pp; vhat[i, j] = vv
                            ahat[i, j] = aa; uhat[i, j] = previous_u[j]; dhat[i, j] = trim[j]
                        elif observer_mode == "timestamp_full_reset":
                            # Propagate the timestamped delayed measurement to the
                            # current time before a full measurement reset.
                            phat[i, j] = p_hist[0][j] + env["delay"] * DT * v_hist[0][j]
                            vhat[i, j] = v_hist[0][j]
                        else:
                            phat[i, j] += .78 * ip; vhat[i, j] += .62 * iv
                        dhat[i, j] = .88 * dhat[i, j] + .12 * np.clip(iv / max(update_steps * DT, DT), -.4, .4)
                        rp = float(np.linalg.norm(phat[i, j] - p[j])); rv = float(np.linalg.norm(vhat[i, j] - v[j]))
                        reset_position_baseline[i, j] = rp; reset_velocity_baseline[i, j] = rv
                        peak_post_reset_position_error = max(peak_post_reset_position_error, rp)
                        peak_post_reset_velocity_error = max(peak_post_reset_velocity_error, rv)
                        comm += 1
                        if target_visibility == "target_rooted":
                            target_messages[i].append((target_position_hat[j], target_velocity_hat[j]))
                    else:
                        missed_updates[i, j] += 1
                        maximum_scheduled_misses = max(maximum_scheduled_misses, int(missed_updates[i, j]))
            if target_visibility == "target_rooted":
                old_p = target_position_hat.copy(); old_v = target_velocity_hat.copy()
                for i in range(1, N):
                    if target_messages[i]:
                        mp = float(np.mean([x[0] for x in target_messages[i]])); mv = float(np.mean([x[1] for x in target_messages[i]]))
                        gain = min(max(target_observer_gain * rho, 0.), 1.)
                        target_position_hat[i] = old_p[i] + gain * (mp - old_p[i])
                        target_velocity_hat[i] = old_v[i] + gain * (mv - old_v[i])
        for i in range(N):
            js = np.where(A[i] > 0)[0]
            if len(js):
                peak_position_observer_error = max(peak_position_observer_error,
                    float(np.max(np.linalg.norm(phat[i, js] - p[js], axis=1))))
                peak_velocity_observer_error = max(peak_velocity_observer_error,
                    float(np.max(np.linalg.norm(vhat[i, js] - v[js], axis=1))))
                for j in js:
                    ep = float(np.linalg.norm(phat[i, j] - p[j])); ev = float(np.linalg.norm(vhat[i, j] - v[j]))
                    peak_position_growth_since_reset = max(peak_position_growth_since_reset,
                        max(ep - reset_position_baseline[i, j], 0.))
                    peak_velocity_growth_since_reset = max(peak_velocity_growth_since_reset,
                        max(ev - reset_velocity_baseline[i, j], 0.))
        trust = np.ones(N)
        if policy in ("observer_residual_gate", "connectivity_constrained_observer_gate"):
            trust = np.clip(.09 / np.maximum(residual, .09), .08, 1.)
            if policy == "connectivity_constrained_observer_gate": trust = np.maximum(trust, .30)
            gate_steps += int(np.any(trust < .999))
        if policy == "heterogeneity_constrained_participation_gate":
            # Persistent mismatch, rather than state-estimation innovation,
            # determines whether a neighbour's physical coupling is trusted.
            node_trim = np.zeros((N, 2))
            for j in range(N):
                receivers = np.where(A[:, j] > 0)[0]
                if len(receivers): node_trim[j] = np.mean(dhat[receivers, j], axis=0)
            centre = np.mean(node_trim, axis=0)
            mismatch = np.linalg.norm(node_trim - centre, axis=1)
            trust = np.maximum(np.clip(.08 / np.maximum(mismatch, .08), .08, 1.), .30)
            gate_steps += int(np.any(trust < .999))
        if policy == "oracle_trim_gate_diagnostic": trust = np.clip(.10 / np.maximum(np.linalg.norm(trim, axis=1), .10), .08, 1.)
        local_pref = np.c_[target_position_hat, yref]
        local_vref = np.c_[target_velocity_hat, np.zeros(N)]
        u = 1.65 * (local_pref - p) + 1.45 * (local_vref - v)
        share = policy not in ("independent_constrained_tracking", "all_coupled_without_trim_sharing") and arm != "all_heterogeneous_without_feedforward_sharing"
        share_ff = np.zeros((N, 2))
        if policy != "independent_constrained_tracking":
            for i in range(N):
                js = np.where(A[i] > 0)[0]
                for j in js:
                    wij = rho * trust[j] / max(len(js), 1)
                    u[i] += wij * (.34 * (phat[i, j] - p[i] - np.array([0., yref[j] - yref[i]])) + .42 * (vhat[i, j] - v[i]))
                    if share:
                        term = wij * P["trim_sharing_gain"] * (dhat[i, j] - trim[i])
                        u[i] += term; share_ff[i] += term
        if cancellation_reserve > 0:
            mismatch_estimate = trim + share_ff
            for i in range(N):
                q = np.linalg.norm(mismatch_estimate[i])
                cancel = -mismatch_estimate[i] if q <= cancellation_reserve else -cancellation_reserve * mismatch_estimate[i] / q
                u[i] += cancel; cancellation_energy += float(np.dot(cancel, cancel)) * DT
        u, sc = safety_projection(u, p, v); safety_cost += sc * DT
        alpha = np.exp(-DT / tau); acc = alpha[:, None] * acc + (1 - alpha)[:, None] * u
        fault = np.zeros((N, 2))
        if P["fault"]["start_s"] <= t < P["fault"]["end_s"]: fault[P["fault"]["node"]] = P["fault"]["additional_trim_mps2"]
        v += DT * (beta[:, None] * acc - drag[:, None] * v + trim + fault + inn["wind"][k]); p += DT * v
        previous_u = u.copy()
        err = float(np.max(np.linalg.norm(p - pref, axis=1))); peak_err = max(peak_err, err)
        target_error = float(np.max(np.abs(target_position_hat - refx))); peak_target_observer_error = max(peak_target_observer_error, target_error)
        if k >= tail_start: tail_err = max(tail_err, err); tail_target_observer_error = max(tail_target_observer_error, target_error)
        pair = min(np.linalg.norm(p[i] - p[j]) for i in range(N) for j in range(i + 1, N)); min_pair = min(min_pair, pair)
        max_corridor = max(max_corridor, float(np.max(np.abs(p[:, 1])))); energy += DT * float(np.sum(acc ** 2))
        if t >= fault_end:
            if err <= P["task_contract"]["analytic_tracking_tolerance_m"]: dwell += DT
            else: dwell = 0
            if recovered_at is None and dwell >= P["fault"]["recovery_dwell_s"]: recovered_at = t - P["fault"]["recovery_dwell_s"]
    completion = float(np.mean(p[:, 0] >= 16.))
    physical = min_pair >= .34 and max_corridor <= 1.60 and completion >= .8
    tube = tail_err <= P["task_contract"]["analytic_tracking_tolerance_m"]
    return {"environment": env_name, "seed": seed, "rho": rho, "policy": policy, "arm": arm,
            "update_interval_s": update_steps * DT, "task_success": int(physical and tube), "physical_task_success": int(physical),
            "tube_success": int(tube), "completion_fraction": completion, "minimum_pair_clearance_m": min_pair,
            "maximum_corridor_excursion_m": max_corridor, "peak_tracking_error_m": peak_err, "tail_tracking_error_m": tail_err,
            "peak_observer_residual": float(np.max(residual)), "control_energy": energy, "communication_messages": comm,
            "peak_position_observer_error_m": peak_position_observer_error,
            "peak_velocity_observer_error_mps": peak_velocity_observer_error,
            "peak_target_observer_error_m": peak_target_observer_error,
            "tail_target_observer_error_m": tail_target_observer_error,
            "target_reachable": 1,
            "peak_post_reset_position_error_m": peak_post_reset_position_error,
            "peak_post_reset_velocity_error_mps": peak_post_reset_velocity_error,
            "peak_position_growth_since_reset_m": peak_position_growth_since_reset,
            "peak_velocity_growth_since_reset_mps": peak_velocity_growth_since_reset,
            "safety_filter_cost": safety_cost, "gate_active_steps": gate_steps,
            "cancellation_energy": cancellation_energy,
            "maximum_scheduled_misses": maximum_scheduled_misses,
            "message_age_bound_updates": -1 if max_missed_updates is None else max_missed_updates,
            "recovery_time_s": -1 if recovered_at is None else recovered_at - fault_end,
            "intrinsic_trim_norm": float(np.linalg.norm(trim)),
            "transmitted_trim_proxy": float(rho * P["trim_sharing_gain"] * np.linalg.norm(A @ trim))}
