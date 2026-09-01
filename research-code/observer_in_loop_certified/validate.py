#!/usr/bin/env python3
"""Fail-closed validation from raw observer-in-loop trajectories."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np

import run_benchmark as m
import test_equations

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_r3'
REL_TOL = 2e-8
RAW_TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_id(row: dict) -> str:
    return f"{row['topology']}_h{float(row['heterogeneity']):.2f}_seed{int(row['seed'])}"


def as_float(row: dict, key: str) -> float:
    return float(row[key])


def independent_certificate(W: np.ndarray, level: float) -> dict:
    """Recompute every analytic constant without production certificate()."""
    L = np.zeros((m.N + 1, m.N + 1)); L[:m.N, :] = -W
    L[np.arange(m.N), np.arange(m.N)] += W.sum(axis=1)
    L1 = L[:m.N, :m.N]
    g = np.linalg.solve(L1.T, np.ones(m.N)); G = np.diag(g)
    S = G @ L1 + L1.T @ G
    mu = float(np.linalg.eigvalsh(np.diag(g ** -.5) @ S @
                                  np.diag(g ** -.5)).min())
    pattern = np.array([-1., -.5, 0., .5, 1.])
    a_nodes = m.A0 * (1 + .45 * level * pattern)
    b_nodes = m.B0 * (1 + .70 * level * pattern[::-1])
    mismatch = (np.abs(a_nodes - m.A0) * m.R +
                np.abs(b_nodes - m.B0) * (m.R + 1))
    omega = float(np.sqrt(np.sum(g * mismatch ** 2)))
    r = float(np.max(-2 * (a_nodes - b_nodes)))
    d = m.ALPHA * mu - r; tracking_rate = d / 2
    observer_rate_proved = m.A0 - m.B0 + m.GAMMA_STATE * mu / 2
    observer_rate_used = min(m.OBSERVER_ENVELOPE_RATE_CAP, observer_rate_proved)
    delta = omega / tracking_rate / np.sqrt(np.min(g))
    follower_W = W[:, :m.N]
    follower_L = np.diag(follower_W.sum(axis=1)) - follower_W
    return dict(L=L, follower_L=follower_L, g=g, mu=mu,
        S_min=float(np.linalg.eigvalsh(S).min()), a=a_nodes, b=b_nodes,
        pin=W[:, m.N].copy(), mismatch=mismatch, omega=omega, r=r, d=d,
        tracking_rate=tracking_rate, observer_rate_proved=observer_rate_proved,
        observer_rate_used=observer_rate_used, delta=float(delta))


def main() -> int:
    required = ['raw_runs.csv', 'raw_trajectories.npz', 'dt_audit.csv',
        'matched_policy_runs.csv', 'matched_policy_trajectories.npz',
        'matched_policy_dt_audit.csv', 'metadata.json', 'RUN_COMPLETION.json']
    missing = [name for name in required if not (OUT / name).is_file()]
    if missing:
        print('FAIL missing result files:', ', '.join(missing)); return 1

    rows = list(csv.DictReader((OUT / 'raw_runs.csv').open()))
    dt_rows = list(csv.DictReader((OUT / 'dt_audit.csv').open()))
    policy_rows = list(csv.DictReader((OUT / 'matched_policy_runs.csv').open()))
    policy_dt_rows = list(csv.DictReader((OUT / 'matched_policy_dt_audit.csv').open()))
    raw = np.load(OUT / 'raw_trajectories.npz')
    policy_raw = np.load(OUT / 'matched_policy_trajectories.npz')
    meta = json.loads((OUT / 'metadata.json').read_text())
    checks = {}
    completion = json.loads((OUT / 'RUN_COMPLETION.json').read_text())
    completed_lock_path = Path(str(OUT) + '.completed.lock')
    completed_lock = json.loads(completed_lock_path.read_text()) \
        if completed_lock_path.is_file() else {}
    checks['atomic_single_run_completion_provenance'] = bool(
        not Path(str(OUT) + '.active.lock').exists()
        and not Path(str(OUT) + '.staging').exists()
        and completed_lock_path.is_file()
        and completion.get('protocol_id') == 'CERT-OIL-R3'
        and completed_lock.get('protocol_id') == 'CERT-OIL-R3'
        and completion.get('pid') == completed_lock.get('pid')
        and completion.get('started_utc') == completed_lock.get('started_utc')
        and completed_lock.get('target') == str(OUT.resolve())
        and isinstance(completion.get('completed_utc'), str))
    checks['formal_protocol_is_CERT_OIL_R3'] = meta.get('protocol_id') == 'CERT-OIL-R3'
    freeze_path = ROOT / 'SOURCE_FREEZE_R3.json'
    freeze = json.loads(freeze_path.read_text()) if freeze_path.is_file() else {}
    freeze_files = freeze.get('sha256', {})
    checks['frozen_protocol_and_source_hashes_match'] = bool(
        freeze.get('protocol_id') == 'CERT-OIL-R3' and freeze_files
        and all((ROOT / name).is_file() and sha256(ROOT / name) == digest
                for name, digest in freeze_files.items())
        and meta.get('source_freeze_sha256') == sha256(freeze_path))
    checks['completion_freeze_hash_matches_metadata'] = bool(
        completion.get('source_freeze_sha256') == meta.get('source_freeze_sha256'))
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_equations)
    equation_result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    checks['all_equation_and_rk4_tests_pass'] = bool(equation_result.wasSuccessful()
        and equation_result.testsRun == 7)
    expected_cases = {(name, level, seed) for name in m.topologies()
                      for level in m.LEVELS for seed in m.SEEDS}
    observed_cases = {(row['topology'], float(row['heterogeneity']), int(row['seed']))
                      for row in rows}
    checks['registered_120_unique_cases'] = (len(rows) == 120 and
        len(observed_cases) == 120 and observed_cases == expected_cases)
    checks['all_primary_runs_finite'] = all(row['finite'] == 'True' for row in rows)
    expected_dt = {(dt, name, seed) for dt in (.004, .002, .001)
                   for name in m.topologies() for seed in range(2)}
    observed_dt = {(float(row['dt']), row['topology'], int(row['seed']))
                   for row in dt_rows}
    checks['registered_24_unique_dt_cases'] = (len(dt_rows) == 24 and
        observed_dt == expected_dt and all(row['finite'] == 'True' for row in dt_rows))
    policies = ('observer_loop', 'oracle_target', 'no_target')
    expected_policy_cases = {(name, level, seed, policy) for name in m.topologies()
        for level in m.LEVELS for seed in m.SEEDS for policy in policies}
    observed_policy_cases = {(row['topology'], float(row['heterogeneity']),
        int(row['seed']), row['policy']) for row in policy_rows}
    checks['registered_360_unique_matched_policy_runs'] = bool(
        len(policy_rows) == 360 and len(observed_policy_cases) == 360
        and observed_policy_cases == expected_policy_cases
        and all(row['finite'] == 'True' for row in policy_rows))
    expected_policy_dt = {(dt, name, seed, policy) for dt in (.004, .002, .001)
        for name in m.topologies() for seed in range(2) for policy in policies}
    observed_policy_dt = {(float(row['dt']), row['topology'], int(row['seed']),
        row['policy']) for row in policy_dt_rows}
    checks['registered_72_unique_matched_policy_dt_runs'] = bool(
        len(policy_dt_rows) == 72 and observed_policy_dt == expected_policy_dt
        and all(row['finite'] == 'True' for row in policy_dt_rows))

    p = meta.get('preregistered', {})
    checks['metadata_identity'] = bool(
        p.get('N') == m.N and p.get('D') == m.D and p.get('seeds') == list(m.SEEDS)
        and np.allclose(p.get('levels'), m.LEVELS) and p.get('dt') == m.DT
        and p.get('t_end') == m.T_END and p.get('alpha') == m.ALPHA
        and p.get('gamma_state') == m.GAMMA_STATE
        and p.get('observer_envelope_rate_cap') == m.OBSERVER_ENVELOPE_RATE_CAP
        and p.get('gain_search') is False and p.get('clipping') is False
        and p.get('discarded_runs') == 0)
    checks['metadata_matched_policy_identity'] = bool(
        p.get('matched_policies') == list(policies)
        and p.get('primary_matched_cases') == 120
        and p.get('primary_policy_runs') == 360
        and p.get('dt_matched_cases') == 24 and p.get('dt_policy_runs') == 72)
    independent_constants_agree = True
    all_rates_positive = True
    for name, W in m.topologies().items():
        for level in m.LEVELS:
            independent = independent_certificate(W, level)
            production = m.certificate(W, level)
            archived = meta.get('certificates', {}).get(name, {}).get(str(level), {})
            for key in ('L', 'follower_L', 'g', 'mu', 'S_min', 'a', 'b', 'pin', 'mismatch',
                        'omega', 'r', 'd', 'tracking_rate',
                        'observer_rate_proved', 'observer_rate_used', 'delta'):
                independent_constants_agree &= bool(np.allclose(
                    independent[key], production[key], atol=2e-14, rtol=2e-14))
                independent_constants_agree &= bool(np.allclose(
                    independent[key], archived.get(key), atol=2e-14, rtol=2e-14))
            all_rates_positive &= bool(
                independent['mu'] > 0 and independent['S_min'] > 0
                and independent['d'] > 0
                and independent['observer_rate_proved'] >=
                    independent['observer_rate_used'] > 0)
    checks['independent_analytic_constants_match_code_and_metadata'] = \
        bool(independent_constants_agree)
    checks['all_graph_and_certificate_rates_positive'] = bool(all_rates_positive)
    checks['target_radius_formula'] = abs(m.R - m.B0 / (m.A0 - m.B0)) < 1e-14

    array_names = ('time', 'plant', 'observer', 'control', 'same_state_oracle_control',
        'tracking_weighted', 'observer_weighted', 'homogeneous_component',
        'mismatch_component', 'observer_component', 'tracking_envelope',
        'observer_envelope', 'tracking_envelope_ratio', 'observer_envelope_ratio')
    expected_keys = {f'{run_id(row)}__{name}' for row in rows for name in array_names}
    checks['raw_key_coverage_exact'] = set(raw.files) == expected_keys
    checks['all_raw_arrays_explicitly_finite'] = bool(
        set(raw.files) == expected_keys and
        all(np.all(np.isfinite(raw[key])) for key in raw.files))
    policy_array_names = ('time', 'plant', 'control')
    expected_policy_keys = {f"{row['topology']}_h{float(row['heterogeneity']):.2f}_seed{int(row['seed'])}__{row['policy']}__{key}"
        for row in policy_rows for key in policy_array_names}
    checks['matched_policy_raw_key_coverage_exact'] = set(policy_raw.files) == expected_policy_keys
    checks['all_matched_policy_arrays_explicitly_finite'] = bool(
        set(policy_raw.files) == expected_policy_keys
        and all(np.all(np.isfinite(policy_raw[key])) for key in policy_raw.files))

    max_raw_difference = 0.0; max_control_difference = 0.0
    max_control_gap_identity_difference = 0.0; max_tracking_ratio = 0.0
    max_observer_ratio = 0.0; max_node_ratio = 0.0; max_target_radius = 0.0
    max_positive_time_tracking_ratio = 0.0
    shapes_ok = True
    for row in rows:
        rid = run_id(row); name = row['topology']; level = float(row['heterogeneity'])
        c = independent_certificate(m.topologies()[name], level)
        t = raw[f'{rid}__time']; plant = raw[f'{rid}__plant']
        observer = raw[f'{rid}__observer']; control = raw[f'{rid}__control']
        oracle = raw[f'{rid}__same_state_oracle_control']
        expected_time = np.arange(round(m.T_END / m.SAVE_DT) + 1) * m.SAVE_DT
        shapes_ok &= (t.ndim == 1 and len(t) == int(row['samples']) and
            plant.shape == (len(t), m.N + 1, m.D) and
            observer.shape == control.shape == oracle.shape == (len(t), m.N, m.D)
            and abs(t[0]) < 1e-15 and abs(t[-1] - m.T_END) < 1e-14
            and np.all(np.diff(t) > 0) and t.shape == expected_time.shape
            and np.allclose(t, expected_time, atol=2e-14, rtol=0))
        diag = m.diagnostics(t, plant, observer, c)
        u_re = []; uo_re = []
        for k in range(len(t)):
            u, uo = m.controller_values(plant[k, :m.N], observer[k],
                c['L'][:m.N, :m.N], c['pin'], plant[k, m.N])
            u_re.append(u); uo_re.append(uo)
        u_re = np.asarray(u_re); uo_re = np.asarray(uo_re)
        max_control_difference = max(max_control_difference,
            float(np.max(np.abs(control - u_re))), float(np.max(np.abs(oracle - uo_re))))
        expected_gap = m.ALPHA * c['pin'][None, :, None] * (
            observer - plant[:, m.N, None, :])
        max_control_gap_identity_difference = max(max_control_gap_identity_difference,
            float(np.max(np.abs((control - oracle) - expected_gap))))
        for key in ('tracking_weighted', 'observer_weighted', 'homogeneous_component',
                    'mismatch_component', 'observer_component', 'tracking_envelope',
                    'observer_envelope', 'tracking_envelope_ratio',
                    'observer_envelope_ratio'):
            max_raw_difference = max(max_raw_difference,
                float(np.max(np.abs(raw[f'{rid}__{key}'] - diag[key]))))
        gap = np.linalg.norm((control - oracle).reshape(len(t), -1), axis=1)
        cnorm = np.linalg.norm(control.reshape(len(t), -1), axis=1)
        onorm = np.linalg.norm(oracle.reshape(len(t), -1), axis=1)
        tail = t >= .8 * m.T_END
        expected_summary = dict(
            initial_tracking=diag['tracking'][0], max_tracking=diag['tracking'].max(),
            tail20_max_tracking=diag['tracking'][tail].max(),
            final_tracking=diag['tracking'][-1],
            tail20_delta_ratio=diag['tracking'][tail].max() / c['delta'],
            initial_observer=diag['observer'][0], max_observer=diag['observer'].max(),
            tail20_max_observer=diag['observer'][tail].max(),
            final_observer=diag['observer'][-1],
            max_tracking_envelope_ratio=diag['tracking_envelope_ratio'].max(),
            max_observer_envelope_ratio=diag['observer_envelope_ratio'].max(),
            max_node_ratio=diag['max_node_ratio'].max(),
            target_radius_max=diag['target_radius'].max(),
            max_control_norm=cnorm.max(),
            max_same_state_oracle_control_norm=onorm.max(),
            max_same_state_control_gap=gap.max(),
            final_same_state_control_gap=gap[-1])
        for key, value in expected_summary.items():
            max_raw_difference = max(max_raw_difference, abs(as_float(row, key) - float(value)))
        max_raw_difference = max(max_raw_difference, abs(as_float(row, 'delta') - c['delta']))
        max_tracking_ratio = max(max_tracking_ratio,
                                 float(diag['tracking_envelope_ratio'].max()))
        max_positive_time_tracking_ratio = max(max_positive_time_tracking_ratio,
            float(diag['tracking_envelope_ratio'][t > 0].max()))
        max_observer_ratio = max(max_observer_ratio,
                                 float(diag['observer_envelope_ratio'].max()))
        max_node_ratio = max(max_node_ratio, float(diag['max_node_ratio'].max()))
        max_target_radius = max(max_target_radius, float(diag['target_radius'].max()))

    checks['raw_trajectory_shapes_complete'] = bool(shapes_ok)
    checks['raw_and_csv_recompute_within_1e_12'] = max_raw_difference < RAW_TOL
    checks['applied_and_oracle_controls_recompute_within_1e_12'] = \
        max_control_difference < RAW_TOL
    checks['control_gap_identity_within_1e_12'] = \
        max_control_gap_identity_difference < RAW_TOL
    checks['target_stayed_in_invariant_ball'] = max_target_radius <= m.R + RAW_TOL
    checks['new_full_time_tracking_envelope_covered'] = max_tracking_ratio <= 1 + REL_TOL
    checks['observer_envelope_covered'] = max_observer_ratio <= 1 + REL_TOL
    checks['node_envelopes_covered'] = max_node_ratio <= 1 + REL_TOL

    policy_summary_difference = 0.0; policy_control_difference = 0.0
    matched_initials = True; autonomous_targets_identical = True
    observer_raw_identity = True; actual_arms_distinct = True; policy_shapes_ok = True
    for name, level, seed in sorted(expected_cases):
        rid = f'{name}_h{level:.2f}_seed{seed}'
        c = independent_certificate(m.topologies()[name], level)
        arm = {}
        for policy in policies:
            prefix = f'{rid}__{policy}'
            t = policy_raw[f'{prefix}__time']; plant = policy_raw[f'{prefix}__plant']
            control = policy_raw[f'{prefix}__control']; arm[policy] = (t, plant, control)
            expected_time = np.arange(round(m.T_END / m.SAVE_DT) + 1) * m.SAVE_DT
            policy_shapes_ok &= bool(
                t.shape == expected_time.shape and plant.shape == (len(t), m.N + 1, m.D)
                and control.shape == (len(t), m.N, m.D) and np.all(np.diff(t) > 0)
                and np.allclose(t, expected_time, atol=2e-14, rtol=0))
            if policy == 'observer_loop':
                estimate = raw[f'{rid}__observer']
                recomputed = np.asarray([m.policy_control(
                    plant[k, :m.N], plant[k, m.N], c, policy, estimate[k])
                    for k in range(len(t))])
            else:
                recomputed = np.asarray([m.policy_control(
                    plant[k, :m.N], plant[k, m.N], c, policy)
                    for k in range(len(t))])
            policy_control_difference = max(policy_control_difference,
                float(np.max(np.abs(control - recomputed))))
            e = plant[:, :m.N] - plant[:, m.N, None, :]
            tracking = np.linalg.norm(e.reshape(len(t), -1), axis=1)
            target_radius = np.linalg.norm(plant[:, m.N], axis=1)
            control_norm = np.linalg.norm(control.reshape(len(t), -1), axis=1)
            tail = t >= .8 * m.T_END
            prow = next(row for row in policy_rows if row['topology'] == name
                and float(row['heterogeneity']) == level and int(row['seed']) == seed
                and row['policy'] == policy)
            expected_summary = dict(initial_tracking=tracking[0],
                max_tracking=tracking.max(), tail20_max_tracking=tracking[tail].max(),
                final_tracking=tracking[-1], target_radius_max=target_radius.max(),
                max_control_norm=control_norm.max())
            policy_summary_difference = max(policy_summary_difference,
                *(abs(float(prow[key]) - float(value))
                  for key, value in expected_summary.items()))
            policy_shapes_ok &= int(prow['samples']) == len(t) and prow['finite'] == 'True'
        initial_expected, _ = m.initial_conditions(seed)
        matched_initials &= all(np.array_equal(arm[policy][1][0], initial_expected)
                                for policy in policies)
        autonomous_targets_identical &= bool(
            np.array_equal(arm['observer_loop'][1][:, m.N], arm['oracle_target'][1][:, m.N])
            and np.array_equal(arm['observer_loop'][1][:, m.N], arm['no_target'][1][:, m.N]))
        observer_raw_identity &= bool(
            np.array_equal(arm['observer_loop'][0], raw[f'{rid}__time'])
            and np.array_equal(arm['observer_loop'][1], raw[f'{rid}__plant'])
            and np.array_equal(arm['observer_loop'][2], raw[f'{rid}__control']))
        positive_differences = [
            np.max(np.abs(arm['observer_loop'][1][1:] - arm['oracle_target'][1][1:])),
            np.max(np.abs(arm['observer_loop'][1][1:] - arm['no_target'][1][1:])),
            np.max(np.abs(arm['oracle_target'][1][1:] - arm['no_target'][1][1:]))]
        actual_arms_distinct &= all(float(value) > 1e-10 for value in positive_differences)
    checks['matched_policy_raw_shapes_and_time_grids'] = bool(policy_shapes_ok)
    checks['matched_policy_initial_plant_and_target_states_exact'] = bool(matched_initials)
    checks['autonomous_target_paths_identical_across_arms'] = bool(autonomous_targets_identical)
    checks['observer_arm_matches_certificate_raw_trajectory'] = bool(observer_raw_identity)
    checks['actual_policy_arms_not_copied_counterfactuals'] = bool(actual_arms_distinct)
    checks['matched_policy_controls_recompute_within_1e_12'] = \
        policy_control_difference < RAW_TOL
    checks['matched_policy_summaries_recompute_within_1e_12'] = \
        policy_summary_difference < RAW_TOL

    groups = {}
    for row in dt_rows:
        groups.setdefault((row['topology'], int(row['seed'])), {})[
            float(row['dt'])] = float(row['final_tracking'])
    dt_rel = [abs(values[.002] - values[.001]) / max(abs(values[.001]), 1e-12)
              for values in groups.values()]
    checks['dt_median_relative_difference_below_1pct'] = float(np.median(dt_rel)) < .01
    dt_policy_groups = {}
    for row in policy_dt_rows:
        dt_policy_groups.setdefault((row['policy'], row['topology'], int(row['seed'])), {})[
            float(row['dt'])] = float(row['final_tracking'])
    dt_policy_rel = {policy: [] for policy in policies}
    for (policy, _, _), values in dt_policy_groups.items():
        dt_policy_rel[policy].append(
            abs(values[.002] - values[.001]) / max(abs(values[.001]), 1e-12))
    checks['matched_policy_dt_medians_below_1pct'] = all(
        float(np.median(values)) < .01 for values in dt_policy_rel.values())
    checks['matched_policy_dt_initial_metrics_exactly_paired'] = all(
        len({float(row['initial_tracking']) for row in policy_dt_rows
             if float(row['dt']) == dt and row['topology'] == name
             and int(row['seed']) == seed}) == 1
        for dt in (.004, .002, .001) for name in m.topologies() for seed in range(2))

    # Array-exact deterministic replay of every primary arm. This is
    # intentionally expensive and prevents same-state diagnostics or copied
    # trajectories from masquerading as independently integrated controls.
    observer_replays_exact = True; comparator_replays_exact = True
    for name, level, seed in sorted(expected_cases):
        rid = f'{name}_h{level:.2f}_seed{seed}'
        replay_row, replay = m.simulate(
            name, m.topologies()[name], level, seed, store=True)
        stored_row = next(row for row in rows if run_id(row) == rid)
        observer_replays_exact &= all(np.array_equal(
            replay[key], raw[f'{rid}__{key}']) for key in array_names)
        observer_replays_exact &= all(
            (stored_row[key] == str(value) if isinstance(value, bool)
             else stored_row[key] == value if isinstance(value, str)
             else float(stored_row[key]) == float(value))
            for key, value in replay_row.__dict__.items())
        for policy in ('oracle_target', 'no_target'):
            replay_policy_row, replay_policy = m.simulate_plant_policy(
                name, m.topologies()[name], level, seed, policy, store=True)
            prefix = f'{rid}__{policy}'
            comparator_replays_exact &= all(np.array_equal(
                replay_policy[key], policy_raw[f'{prefix}__{key}'])
                for key in policy_array_names)
            stored = next(row for row in policy_rows if row['topology'] == name
                and float(row['heterogeneity']) == level and int(row['seed']) == seed
                and row['policy'] == policy)
            comparator_replays_exact &= all(
                (stored[key] == str(value) if isinstance(value, bool)
                 else stored[key] == value if isinstance(value, str)
                 else float(stored[key]) == float(value))
                for key, value in replay_policy_row.__dict__.items())
    checks['all_120_observer_primary_replays_exact'] = bool(observer_replays_exact)
    checks['all_240_oracle_and_no_target_primary_replays_exact'] = bool(
        comparator_replays_exact)

    # Deterministically replay all 72 policy-specific step-audit arms.
    dt_replays_exact = True
    for dt in (.004, .002, .001):
        for name, W in m.topologies().items():
            for seed in range(2):
                observer_row = m.simulate(name, W, .30, seed, dt=dt)[0]
                for policy, replay_row in [('observer_loop', observer_row),
                    ('oracle_target', m.simulate_plant_policy(
                        name, W, .30, seed, 'oracle_target', dt=dt)[0]),
                    ('no_target', m.simulate_plant_policy(
                        name, W, .30, seed, 'no_target', dt=dt)[0])]:
                    stored = next(row for row in policy_dt_rows
                        if float(row['dt']) == dt and row['topology'] == name
                        and int(row['seed']) == seed and row['policy'] == policy)
                    dt_replays_exact &= all(
                        (stored[key] == str(value) if isinstance(value, bool)
                         else stored[key] == value if isinstance(value, str)
                         else float(stored[key]) == float(value))
                        for key, value in replay_row.__dict__.items())
    checks['all_72_matched_policy_dt_replays_exact'] = bool(dt_replays_exact)

    hashes = {path.name: sha256(path) for path in sorted(OUT.iterdir())
              if path.is_file() and path.name != 'validation.json'}
    report = dict(passed=all(checks.values()), checks=checks,
        diagnostics=dict(max_raw_recompute_difference=max_raw_difference,
            max_control_recompute_difference=max_control_difference,
            max_control_gap_identity_difference=max_control_gap_identity_difference,
            max_tracking_envelope_ratio=max_tracking_ratio,
            max_positive_time_tracking_envelope_ratio=max_positive_time_tracking_ratio,
            max_observer_envelope_ratio=max_observer_ratio,
            max_node_ratio=max_node_ratio, max_target_radius=max_target_radius,
            matched_policy_summary_difference=policy_summary_difference,
            matched_policy_control_difference=policy_control_difference,
            median_dt_relative_difference=float(np.median(dt_rel)),
            matched_policy_median_dt_relative_difference={key: float(np.median(value))
                for key, value in dt_policy_rel.items()}), sha256=hashes)
    report_bytes = (json.dumps(report, indent=2) + '\n').encode()
    report_path = OUT / 'validation.json'
    if report_path.exists():
        if report_path.read_bytes() != report_bytes:
            raise RuntimeError('existing validation.json differs; overwrite refused')
    else:
        with report_path.open('xb') as handle:
            handle.write(report_bytes)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
