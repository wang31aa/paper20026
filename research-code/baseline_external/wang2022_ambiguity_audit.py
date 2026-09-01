#!/usr/bin/env python3
"""Executable audit of equation/example inconsistencies in Wang (2022).

This is not a performance implementation. It freezes two interpretations of
the Perron-like vector used in printed equation (11) and in the numerical
example/code, and verifies the resulting spectral claims without silently
choosing one branch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).with_name("wang2022_ambiguity_results.json")

# Equation (51), identical to the augmented graph literal in the supplied
# Graph-1 source. The follower block is the leading 7x7 block.
L_TILDE = np.array([
    [1, 0, 0, 0, 0, 0, 0, -1],
    [-2, 2, 0, 0, 0, 0, 0, 0],
    [0, -2, 2, 0, 0, 0, 0, 0],
    [-2, 0, -1, 3, 0, 0, 0, 0],
    [0, 0, -3, 0, 3, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=float)


def branch(matrix: np.ndarray) -> dict[str, object]:
    theta = np.linalg.solve(matrix, np.ones(7))
    G = np.diag(1.0 / theta)
    eig = np.linalg.eigvalsh(G @ L_TILDE[:7, :7] + L_TILDE[:7, :7].T @ G)
    return {
        "theta": theta.tolist(),
        "symmetric_part_eigenvalues": eig.tolist(),
        "lambda_min": float(eig[0]),
        "positive_definite": bool(eig[0] > 0),
    }


L1 = L_TILDE[:7, :7]
printed = branch(L1.T)
example = branch(L1)
result = {
    "purpose": "equation/example ambiguity audit; not a performance score",
    "printed_equation_11_theta_solve": "solve(L1.T, ones)",
    "example_code_theta_solve": "solve(L1, ones)",
    "printed_branch": printed,
    "example_branch": example,
    "reported_lambda0": 0.1246,
    "example_lambda_matches_reported": abs(example["lambda_min"] - 0.1246) < 1e-4,
    "printed_branch_contradicts_positive_definiteness": not printed["positive_definite"],
    "gain_inequality_conflict": {
        "theorem_text": "c > alpha/lambda0",
        "numerical_example": "alpha > c/lambda0",
        "c": 2.0,
        "alpha": 16.1,
        "c_over_lambda0": 2.0 / example["lambda_min"],
    },
    "published_vs_snapshot_certificate": {
        "published_static_bound": 0.0520,
        "published_adaptive_bound": 0.1808,
        "unpublished_draft_snapshot_bound": 0.0255,
        "must_not_be_conflated": True,
    },
}
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

assert result["example_lambda_matches_reported"]
assert result["printed_branch_contradicts_positive_definiteness"]
assert result["gain_inequality_conflict"]["c_over_lambda0"] < 16.1
print("PASS: Wang (2022) printed/example graph branches frozen; contradiction retained")
