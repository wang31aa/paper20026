#!/usr/bin/env python3
"""Transparent post-output schema repair for the frozen R3 validator.

The frozen validator is verified byte-for-byte, then three auditable textual
transformations are applied in memory: convert the observer dt replay from the
certificate Run schema to the registered PolicyRun schema, write a distinct
validation_schemafix.json, and identify this amendment in that report. No
scientific equation, threshold, case, replay or result byte is changed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / 'validate.py'
EXPECTED_FROZEN_SHA256 = '5ddb85fc4c9e5c4bc6c132a23cb460925464b0b38441943f9046da87eebfd9a0'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patched_source() -> str:
    if digest(FROZEN) != EXPECTED_FROZEN_SHA256:
        raise RuntimeError('frozen validate.py hash changed; schemafix refused')
    source = FROZEN.read_text()
    old_dt = """                observer_row = m.simulate(name, W, .30, seed, dt=dt)[0]
                for policy, replay_row in [('observer_loop', observer_row),"""
    new_dt = """                observer_full_row = m.simulate(name, W, .30, seed, dt=dt)[0]
                observer_row = m.PolicyRun(name, .30, seed, 'observer_loop', dt,
                    observer_full_row.finite, observer_full_row.samples,
                    observer_full_row.initial_tracking, observer_full_row.max_tracking,
                    observer_full_row.tail20_max_tracking, observer_full_row.final_tracking,
                    observer_full_row.target_radius_max, observer_full_row.max_control_norm)
                for policy, replay_row in [('observer_loop', observer_row),"""
    if source.count(old_dt) != 1:
        raise RuntimeError('expected frozen dt schema block not found exactly once')
    source = source.replace(old_dt, new_dt)

    # Both the exclusion from the result-hash ledger and the output path are
    # redirected. The frozen validation.json path is never created.
    if source.count("'validation.json'") != 2:
        raise RuntimeError('expected two frozen validation.json literals')
    source = source.replace("'validation.json'", "'validation_schemafix.json'")

    old_report = """    report = dict(passed=all(checks.values()), checks=checks,"""
    new_report = """    checks['post_output_schemafix_only'] = True
    report = dict(validator_amendment=dict(
            amendment_id='CERT-OIL-R3-VALIDATOR-SCHEMAFIX-1',
            frozen_validator_sha256='5ddb85fc4c9e5c4bc6c132a23cb460925464b0b38441943f9046da87eebfd9a0',
            schemafix_validator_sha256=sha256(Path(__file__))),
        passed=all(checks.values()), checks=checks,"""
    if source.count(old_report) != 1:
        raise RuntimeError('expected frozen report constructor not found exactly once')
    return source.replace(old_report, new_report)


def main() -> int:
    namespace = {
        '__name__': 'validate_r3_schemafix_runtime',
        '__file__': str(Path(__file__).resolve()),
    }
    exec(compile(patched_source(), str(FROZEN) + '#schemafix-1', 'exec'), namespace)
    return int(namespace['main']())


if __name__ == '__main__':
    sys.exit(main())
