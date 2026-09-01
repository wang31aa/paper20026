#!/usr/bin/env python3
"""Exhaustive finite-state checks for the two-layer capability statements."""
from itertools import product


def gfp(states, task, actions, transition, admissible):
    kernel = set(task)
    while True:
        nxt = {
            x for x in kernel
            if any(admissible(x, a) and transition(x, a) in kernel
                   for a in actions[x])
        }
        if nxt == kernel:
            return kernel
        kernel = nxt


def main():
    # Counterexample: separate information/physical witnesses do not compose.
    states = {"x", "i", "p", "f"}
    actions = {s: ("I", "P") for s in states}
    trans = {
        ("x", "I"): "i", ("x", "P"): "p",
        ("i", "I"): "i", ("i", "P"): "f",
        ("p", "I"): "f", ("p", "P"): "p",
        ("f", "I"): "f", ("f", "P"): "f",
    }
    step = lambda x, a: trans[(x, a)]
    ki = gfp(states, {"x", "i"}, actions, step,
             lambda _x, a: a == "I")
    kp = gfp(states, {"x", "p"}, actions, step,
             lambda _x, a: a == "P")
    k2 = gfp(states, {"x"}, actions, step, lambda _x, _a: False)
    assert "x" in ki & kp and "x" not in k2

    # Selective-only region: all-on and all-off fail, selective succeeds.
    states2 = {"x", "s", "f"}
    actions2 = {s: ("all", "off", "selective") for s in states2}
    trans2 = {(x, a): ("s" if x in {"x", "s"} and a == "selective" else "f")
              for x, a in product(states2, actions2["x"])}
    trans2[("s", "selective")] = "s"
    step2 = lambda x, a: trans2[(x, a)]
    ksel = gfp(states2, {"x", "s"}, actions2, step2,
               lambda _x, a: a == "selective")
    kall = gfp(states2, {"x", "s"}, actions2, step2,
               lambda _x, a: a == "all")
    koff = gfp(states2, {"x", "s"}, actions2, step2,
               lambda _x, a: a == "off")
    assert "x" in ksel and "x" not in kall and "x" not in koff

    # Information-action enrichment cannot shrink a greatest fixed point.
    base_actions = {s: ("off",) for s in states2}
    rich_actions = {s: ("off", "selective") for s in states2}
    kbase = gfp(states2, {"x", "s"}, base_actions, step2,
                lambda _x, a: a == "selective")
    krich = gfp(states2, {"x", "s"}, rich_actions, step2,
                lambda _x, a: a == "selective")
    assert kbase <= krich
    print("PASS: composition counterexample, selective-only region, and information monotonicity")


if __name__ == "__main__":
    main()
