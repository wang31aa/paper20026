#!/usr/bin/env python3
"""Read-only source-semantic checks for unreproduced MATLAB branches.

This script does not execute or translate MATLAB. It verifies frozen hashes and
predeclared structural facts that a later Python port must preserve.
"""
from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "incoming/2026-07-28_author_matlab/source_tree/code"

HASHES = {
    "FourNodes_externl.m": "ae19aaab063e0ef4277ea12ae87612bcc4411ceff0e35b9f0d55bb4ed378a1d0",
    "FourNodes_full_robust.m": "4672a97f71da516c045fb1d08346a6baff5050eaa457f68c34992fd113cbf01b",
    "FourNodes_full_ADA_20220203.m": "185ab78666ee1ce6f56fe69a78d2feec5874d93bfa86771689de5fafcc0ca6f8",
    "FourNodes_full_ADA_20220204.m": "d4e561107aa61da18db77fcdf067971100a04e3176c84782d296261a6721c462",
    "FourNodes_full_ADA_robust_null.m": "bde0a50243fc96809f01450726d11878fde4190911e0d0dfab07df96f6681911",
    "safecommunciation.m": "a82b8596c198688bfae28b393805468c56584810e829b17e5610eacf3075fd9e",
}


def text(name: str) -> str:
    return (SRC / name).read_text(encoding="utf-8", errors="strict")


def sha256(name: str) -> str:
    return hashlib.sha256((SRC / name).read_bytes()).hexdigest()


def numerical_lines(s: str) -> list[str]:
    """Remove blank/comment/plot-only lines for the A03/A04 identity check."""
    out = []
    in_removed_plot = False
    for line in s.splitlines():
        stripped = line.strip()
        if stripped == "cc=hsv(8)":
            in_removed_plot = True
        if in_removed_plot and stripped.startswith("figure") and out:
            # The first figure starts the A03-only phase portrait; the next one
            # begins common output. A line-level diff below is safer, so this
            # helper is not used as an acceptance oracle.
            pass
        if stripped and not stripped.startswith("%"):
            out.append(stripped)
    return out


def switch_on(k: int, h: float = 0.001) -> bool:
    # Exact integer form of the ten source intervals at the archived h.
    for block in range(10):
        lo = 1 + int((10.0 * block - h) / h) if block else 1
        hi = 1 + int(((10.0 * block + 9.7) - h) / h)
        if lo <= k <= hi:
            return True
    return False


def main() -> None:
    checks: dict[str, object] = {}
    checks["hashes"] = {name: sha256(name) == expected for name, expected in HASHES.items()}
    sources = {name: text(name) for name in HASHES}
    checks["all_initialize_xx4_from_x3"] = all("xx4(:,1)=x3;" in s for s in sources.values())
    checks["external_has_unscaled_kick"] = "+H*(sin(60*(k-1)))" in sources["FourNodes_externl.m"]
    checks["safe_has_unscaled_kick"] = "+H*(sin(60*(k-1)))" in sources["safecommunciation.m"]
    checks["adaptive_cross_family_ccc2"] = "ccc2(:,k)=cc2(:,k-1)" in sources["FourNodes_full_ADA_20220203.m"]
    checks["robust_null_initializes_only_six_cc"] = (
        "cc6(:,1)=1;" in sources["FourNodes_full_ADA_robust_null.m"]
        and "cc7(:,1)=1;" not in sources["FourNodes_full_ADA_robust_null.m"]
    )

    a03 = sources["FourNodes_full_ADA_20220203.m"].splitlines(keepends=True)
    a04 = sources["FourNodes_full_ADA_20220204.m"].splitlines(keepends=True)
    diff = list(difflib.unified_diff(a03, a04))
    changed = [line for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
    checks["a03_a04_only_22_plot_statements"] = (
        len([x for x in changed if x.startswith("-")]) == 22
        and len([x for x in changed if x.startswith("+")]) == 1
        and not changed[-1][1:].strip()
    )

    on = sum(switch_on(k) for k in range(2, 100002))
    checks["safe_on_steps"] = on
    checks["safe_off_steps"] = 100000 - on
    checks["safe_switch_schedule_expected"] = (on, 100000 - on) == (97007, 2993)

    # A direct kick cannot have a fixed continuous-time amplitude under grid
    # refinement: max equivalent derivative is ||H||/h.
    h_norm = (2.0 ** 0.5) / 10.0
    rates = {str(h): h_norm / h for h in (0.001, 0.0005, 0.00025)}
    checks["kick_equivalent_derivative_norms"] = rates
    checks["kick_rate_doubles_under_halving_h"] = abs(rates["0.0005"] / rates["0.001"] - 2.0) < 1e-15

    failures = []
    for key, value in checks.items():
        if key == "hashes":
            failures.extend(f"hash:{n}" for n, ok in value.items() if not ok)
        elif key.endswith(("expected", "x3", "kick", "ccc2", "six_cc", "plot_block", "halving_h")) and value is not True:
            failures.append(key)

    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "checks": checks}
    out = HERE / "source_contract_results.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
