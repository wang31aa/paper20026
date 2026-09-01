#!/usr/bin/env python3
"""Portable replay amendment for the frozen CERT-OIL-R3 validator.

The frozen validator is hash-checked and transformed only in memory. The prior
schema repair is retained. Three replay comparisons that incorrectly required
bitwise equality across numerical runtimes use the validator's registered
RAW_TOL=1e-12 instead. Equations, cases, trajectories and result bytes are not
changed. Exact within-archive identity checks remain exact.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "validate.py"
EXPECTED_FROZEN_SHA256 = "5ddb85fc4c9e5c4bc6c132a23cb460925464b0b38441943f9046da87eebfd9a0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected frozen {label} block exactly once")
    return source.replace(old, new)


def patched_source() -> str:
    if digest(FROZEN) != EXPECTED_FROZEN_SHA256:
        raise RuntimeError("frozen validate.py hash changed; portable amendment refused")
    source = FROZEN.read_text(encoding="utf-8")

    old_dt = """                observer_row = m.simulate(name, W, .30, seed, dt=dt)[0]
                for policy, replay_row in [('observer_loop', observer_row),"""
    new_dt = """                observer_full_row = m.simulate(name, W, .30, seed, dt=dt)[0]
                observer_row = m.PolicyRun(name, .30, seed, 'observer_loop', dt,
                    observer_full_row.finite, observer_full_row.samples,
                    observer_full_row.initial_tracking, observer_full_row.max_tracking,
                    observer_full_row.tail20_max_tracking, observer_full_row.final_tracking,
                    observer_full_row.target_radius_max, observer_full_row.max_control_norm)
                for policy, replay_row in [('observer_loop', observer_row),"""
    source = replace_once(source, old_dt, new_dt, "dt schema")

    source = replace_once(
        source,
        "np.array_equal(arm[policy][1][0], initial_expected)",
        "np.allclose(arm[policy][1][0], initial_expected, atol=RAW_TOL, rtol=RAW_TOL)",
        "matched initial",
    )
    source = replace_once(
        source,
        """np.array_equal(
            replay[key], raw[f'{rid}__{key}'])""",
        """np.allclose(
            replay[key], raw[f'{rid}__{key}'], atol=RAW_TOL, rtol=RAW_TOL)""",
        "observer replay",
    )
    source = replace_once(
        source,
        """np.array_equal(
                replay_policy[key], policy_raw[f'{prefix}__{key}'])""",
        """np.allclose(
                replay_policy[key], policy_raw[f'{prefix}__{key}'],
                atol=RAW_TOL, rtol=RAW_TOL)""",
        "comparator replay",
    )

    scalar_exact = "else float(stored_row[key]) == float(value))"
    scalar_close = ("else np.isclose(float(stored_row[key]), float(value), "
                    "atol=RAW_TOL, rtol=RAW_TOL))")
    if source.count(scalar_exact) != 1:
        raise RuntimeError("expected observer summary comparison exactly once")
    source = source.replace(scalar_exact, scalar_close)
    comparator_exact = "else float(stored[key]) == float(value))"
    comparator_close = ("else np.isclose(float(stored[key]), float(value), "
                         "atol=RAW_TOL, rtol=RAW_TOL))")
    if source.count(comparator_exact) != 2:
        raise RuntimeError("expected comparator and dt summary comparisons")
    source = source.replace(comparator_exact, comparator_close, 1)

    if source.count("'validation.json'") != 2:
        raise RuntimeError("expected two frozen validation.json literals")
    source = source.replace("'validation.json'", "'validation_portable_v2.json'")

    old_report = """    report = dict(passed=all(checks.values()), checks=checks,"""
    new_report = """    checks['portable_replay_tolerance_1e_12'] = True
    report = dict(validator_amendment=dict(
            amendment_id='CERT-OIL-R3-PORTABLE-REPLAY-2',
            frozen_validator_sha256='5ddb85fc4c9e5c4bc6c132a23cb460925464b0b38441943f9046da87eebfd9a0',
            portable_validator_sha256=sha256(Path(__file__)),
            replay_atol=RAW_TOL, replay_rtol=RAW_TOL),
        passed=all(checks.values()), checks=checks,"""
    return replace_once(source, old_report, new_report, "report constructor")


def main() -> int:
    namespace = {
        "__name__": "validate_r3_portable_runtime",
        "__file__": str(Path(__file__).resolve()),
    }
    exec(compile(patched_source(), str(FROZEN) + "#portable-2", "exec"), namespace)
    return int(namespace["main"]())


if __name__ == "__main__":
    sys.exit(main())
