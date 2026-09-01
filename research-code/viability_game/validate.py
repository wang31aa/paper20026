#!/usr/bin/env python3
"""Exact finite reach-and-stay game and certified set sandwich example."""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

State = tuple[int, int, int]  # two errors and previous graph mode (controller memory)

GRID = range(-5, 6)
STATES = {(x, y, m) for x, y, m in product(GRID, GRID, (0, 1))}
DIST = tuple(product((-1, 0, 1), repeat=2))
PEAK = 5
TARGET = {(x, y, m) for x, y, m in STATES if max(abs(x), abs(y)) <= 1}


def transition(s: State, action, w) -> State | None:
    x, y, _ = s
    mode, ux, uy = action
    wx, wy = w
    # Integer non-normal directed maps. Mode is included in the successor and
    # is therefore part of the Markov state rather than hidden policy memory.
    if mode == 0:
        xn, yn = x - (1 if x > 0 else -1 if x < 0 else 0) + ux + wx, y + x // 3 + uy + wy
    else:
        xn, yn = x + y // 3 + ux + wx, y - (1 if y > 0 else -1 if y < 0 else 0) + uy + wy
    if xn not in GRID or yn not in GRID:
        return None
    return xn, yn, mode


def excursion(s: State, successor: State | None) -> int:
    if successor is None:
        return 10**6
    return max(abs(s[0]), abs(s[1]), abs(successor[0]), abs(successor[1]))


def pre(c: set[State], peak: int, disturbances=DIST, authority=1) -> set[State]:
    actions = tuple(product((0, 1), range(-authority, authority + 1),
                            range(-authority, authority + 1)))
    out = set()
    for s in STATES:
        for a in actions:
            successors = [transition(s, a, w) for w in disturbances]
            if all(t in c and excursion(s, t) <= peak for t in successors):
                out.add(s); break
    return out


def greatest_invariant(target: set[State], peak: int, predecessor) -> set[State]:
    z = set(target)
    while True:
        new = target & predecessor(z, peak)
        if new == z:
            return z
        z = new


def reach_stay(target: set[State], peak: int, predecessor) -> set[State]:
    inv = greatest_invariant(target, peak, predecessor)
    safe = {s for s in STATES if s in pre(STATES, peak)}
    y = set(inv)
    while True:
        new = y | (safe & predecessor(y, peak))
        if new == y:
            return y
        y = new


def inner_pre(c: set[State], peak: int) -> set[State]:
    # More adversarial disturbance set and a one-unit threshold tightening.
    expanded = tuple(product((-2, -1, 0, 1, 2), repeat=2))
    return pre(c, peak - 1, expanded, authority=2)


def outer_pre(c: set[State], peak: int) -> set[State]:
    # Optimistic nominal disturbance and one-unit relaxation. This is an outer
    # operator for this declared finite example because exact feasible actions
    # remain feasible under a subset of disturbances and a relaxed threshold.
    return pre(c, peak + 1, ((0, 0),), authority=2)


def main() -> None:
    weak = lambda c, p: pre(c, p, DIST, authority=1)
    exact_pre = lambda c, p: pre(c, p, DIST, authority=2)
    infeasible = reach_stay(TARGET, PEAK, weak)
    exact = reach_stay(TARGET, PEAK, exact_pre)
    inner = reach_stay(TARGET, PEAK, inner_pre)
    outer = reach_stay(TARGET, PEAK, outer_pre)
    assert inner <= exact <= outer
    for c in (set(), TARGET, exact, STATES):
        assert inner_pre(c, PEAK) <= exact_pre(c, PEAK) <= outer_pre(c, PEAK)
    assert exact
    report = {
        "states": len(STATES), "target": len(TARGET),
        "weak_authority_winning": len(infeasible),
        "inner_winning": len(inner), "exact_winning": len(exact),
        "outer_winning": len(outer),
        "sandwich": True,
        "claim_boundary": "exact finite augmented-Markov abstraction; not a continuous-state Hausdorff certificate",
    }
    out = Path(__file__).with_name("results.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
