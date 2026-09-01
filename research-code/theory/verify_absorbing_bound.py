#!/usr/bin/env python3
"""Audit the infinite-time target-envelope gate for the archived Chua model.

The decisive stability test uses exact rational arithmetic.  Floating-point
eigenvalues are included only as diagnostics; they are not the proof.
"""
from __future__ import annotations

from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def target_outer_matrix() -> list[list[F]]:
    # Node 8: a=8.3, ell=13.586, gamma=0.0013.  In |s1|>1 the
    # saturation is constant, so this is the homogeneous outer matrix.
    return [
        [-F(83, 35), F(83, 10), F(0)],
        [F(1), -F(1), F(1)],
        [F(0), -F(6793, 500), F(13, 10000)],
    ]


def cubic_coefficients(A: list[list[F]]) -> tuple[F, F, F]:
    # det(lambda I-A)=lambda^3+a1 lambda^2+a2 lambda+a3.
    tr = sum(A[i][i] for i in range(3))
    a1 = -tr
    a2 = (
        A[0][0] * A[1][1] - A[0][1] * A[1][0]
        + A[0][0] * A[2][2] - A[0][2] * A[2][0]
        + A[1][1] * A[2][2] - A[1][2] * A[2][1]
    )
    det = (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )
    return a1, a2, -det


def frac_record(x: F) -> dict[str, object]:
    return {"exact": f"{x.numerator}/{x.denominator}", "decimal": float(x)}


def main() -> None:
    A_exact = target_outer_matrix()
    a1, a2, a3 = cubic_coefficients(A_exact)
    hurwitz_margin = a1 * a2 - a3

    # Exact cubic Routh-Hurwitz criterion: for positive a1,a2,a3, Hurwitz
    # stability requires a1*a2>a3.  Its strict failure here certifies two
    # open-right-half-plane roots (no rounding enters this conclusion).
    exact_rhp_pair = a1 > 0 and a2 > 0 and a3 > 0 and hurwitz_margin < 0
    routh_third = hurwitz_margin / a1

    A = np.array([[float(x) for x in row] for row in A_exact])
    eig = np.linalg.eigvals(A)

    theory = json.loads((HERE / "result.json").read_text())
    g = np.asarray(theory["G"], dtype=float)
    Q = np.asarray(theory["Q"], dtype=float)
    delta_a = np.array([0.1 * (8 - i) for i in range(1, 8)])
    weighted_delta_sq = float(np.sum(g * delta_a**2))
    conversion = float(
        2.0 / (theory["a"] * np.sqrt(g.min() * np.linalg.eigvalsh(Q).min()))
    )

    # If an independently proved ||s(t)||_2 <= S is supplied, this formula is
    # a rigorous mismatch envelope.  It follows from |sat(s1)|<=1 and the
    # exact node schedule; no target samples enter it.
    conditional = {
        "assumption": "limsup ||s(t)||_2 <= S, with a proved or validated S",
        "weighted_delta_sq": weighted_delta_sq,
        "Omega_upper_formula": (
            "sqrt(Cg*(q11*((sqrt(53)/7)*S+3/7)^2 "
            "+ q33*(sqrt(1+1e-6)*S)^2))"
        ),
        "Delta_upper_formula": "conversion_factor * Omega_upper(S)",
        "conversion_factor": conversion,
        "examples_not_certificates": {},
    }
    for S in (1.0, 2.0, 4.0, 5.0, 10.0):
        omega = np.sqrt(
            weighted_delta_sq
            * (
                Q[0, 0] * (np.sqrt(53.0) / 7.0 * S + 3.0 / 7.0) ** 2
                + Q[2, 2] * (np.sqrt(1.0 + 1e-6) * S) ** 2
            )
        )
        conditional["examples_not_certificates"][str(S)] = {
            "Omega_upper": float(omega),
            "Delta_upper": float(conversion * omega),
        }

    out = {
        "model": {
            "node": 8,
            "a": 8.3,
            "ell": 13.586,
            "gamma": 0.0013,
            "nonlinearity": "sat(s1)=0.5*(|s1+1|-|s1-1|)",
            "nonlinearity_bound": 1.0,
        },
        "exact_characteristic_coefficients": {
            "a1": frac_record(a1),
            "a2": frac_record(a2),
            "a3": frac_record(a3),
            "a1_a2_minus_a3": frac_record(hurwitz_margin),
        },
        "exact_routh_hurwitz_conclusion": {
            "first_column": [
                frac_record(F(1)), frac_record(a1),
                frac_record(routh_third), frac_record(a3),
            ],
            "sign_pattern": ["+", "+", "-", "+"],
            "sign_changes": 2,
            "right_half_plane_pair": exact_rhp_pair,
            "outer_matrix_hurwitz": not exact_rhp_pair,
        },
        "floating_point_diagnostic_eigenvalues": [
            {"real": float(z.real), "imag": float(z.imag)} for z in eig
        ],
        "global_absorbing_set_gate": {
            "passed": False,
            "reason": (
                "The bounded saturation cannot yield a global absorbing set "
                "when the outer homogeneous matrix has an open-right-half-plane "
                "mode; a left-eigenvector projection gives escaping solutions."
            ),
        },
        "specific_archived_orbit_gate": {
            "passed": False,
            "reason": (
                "Finite integrations do not prove that the particular chaotic "
                "orbit remains in a trapping region for all future time."
            ),
        },
        "conditional_bound": conditional,
    }
    if not exact_rhp_pair:
        raise RuntimeError("Exact Routh-Hurwitz audit no longer matches the archived model")
    (HERE / "absorbing_bound_result.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
