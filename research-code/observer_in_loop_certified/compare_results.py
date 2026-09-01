#!/usr/bin/env python3
"""Compare a clean CERT-OIL-R3 regeneration with the frozen result set."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CSV_FILES = (
    'raw_runs.csv', 'dt_audit.csv', 'matched_policy_runs.csv',
    'matched_policy_dt_audit.csv')
ARRAY_FILES = ('raw_trajectories.npz', 'matched_policy_trajectories.npz')


def compare_csv(expected: Path, actual: Path) -> float:
    with expected.open(newline='') as handle:
        exp = list(csv.DictReader(handle))
    with actual.open(newline='') as handle:
        act = list(csv.DictReader(handle))
    if len(exp) != len(act) or (exp and exp[0].keys() != act[0].keys()):
        raise ValueError(f'CSV structure differs: {expected.name}')
    maximum = 0.0
    for erow, arow in zip(exp, act):
        for key in erow:
            try:
                maximum = max(maximum, abs(float(erow[key]) - float(arow[key])))
            except ValueError:
                if erow[key] != arow[key]:
                    raise ValueError(f'CSV label differs in {expected.name}: {key}')
    return maximum


def compare_npz(expected: Path, actual: Path) -> float:
    exp = np.load(expected)
    act = np.load(actual)
    if set(exp.files) != set(act.files):
        raise ValueError(f'array keys differ: {expected.name}')
    return max(float(np.max(np.abs(exp[key] - act[key]))) for key in exp.files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-12)
    args = parser.parse_args()
    maxima = {name: compare_csv(args.expected / name, args.actual / name)
              for name in CSV_FILES}
    maxima.update({name: compare_npz(args.expected / name, args.actual / name)
                   for name in ARRAY_FILES})
    passed = max(maxima.values(), default=0.0) <= args.atol
    print(json.dumps({'passed': passed, 'absolute_tolerance': args.atol,
                      'maximum_absolute_differences': maxima}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
