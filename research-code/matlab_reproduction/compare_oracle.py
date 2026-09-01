#!/usr/bin/env python3
"""Compare a literal Python run with the archived MATLAB workspace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


CHECKPOINTS = (0, 1, 2, 9, 99, 999, 9999, 49999, 100000)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--implementation", default="literal-static-core-v1")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    names = [*(f"xx{i}" for i in range(1, 9)), *(f"xg{i}" for i in range(1, 8)),
             *(f"MA{i}" for i in range(1, 8)), *(f"MB{i}" for i in range(1, 8)),
             "Error", *(f"Error{i}" for i in range(1, 8))]
    m = loadmat(args.mat, variable_names=names)
    py = np.load(args.npz)
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for i in range(8): pairs.append((f"xx{i+1}", m[f"xx{i+1}"], py["xx"][i]))
    for i in range(7): pairs.append((f"xg{i+1}", m[f"xg{i+1}"], py["xg"][i]))
    for family in ("MA", "MB"):
        for i in range(7): pairs.append((f"{family}{i+1}", m[f"{family}{i+1}"], py[family][i]))
    pairs.append(("Error", m["Error"], py["Error"]))
    for i in range(7): pairs.append((f"Error{i+1}", m[f"Error{i+1}"], py["Error_agents"][i]))

    rows = []
    all_early = True
    all_full = True
    raw_points = []
    for name, matlab, python in pairs:
        matlab = np.asarray(matlab).squeeze()
        python = np.asarray(python).squeeze()
        # Short-horizon selection runs compare against the corresponding
        # prefix of the archived full trajectory.
        if matlab.ndim == python.ndim and matlab.shape[:-1] == python.shape[:-1] and matlab.shape[-1] >= python.shape[-1]:
            matlab = matlab[..., :python.shape[-1]]
        # MATLAB 3x3xT aligns directly; Python matrix history is 3x3xT.
        shape_ok = matlab.shape == python.shape
        if not shape_ok:
            rows.append({"variable": name, "shape_match": False})
            all_early = all_full = False
            continue
        diff = np.abs(python - matlab)
        denom = np.maximum(np.abs(matlab), 1e-12)
        rel = diff / denom
        early_slice = (..., slice(0, min(100, matlab.shape[-1])))
        early_abs = float(np.max(diff[early_slice]))
        early_rel = float(np.max(rel[early_slice]))
        max_abs = float(np.max(diff))
        nrmse = float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(np.abs(matlab) ** 2)) + 1e-12))
        finite_match = bool(np.array_equal(np.isfinite(matlab), np.isfinite(python)))
        early_pass = finite_match and early_abs <= 1e-11 and early_rel <= 1e-9
        full_pass = finite_match and max_abs <= 1e-6 and nrmse <= 1e-7
        all_early &= early_pass
        all_full &= full_pass
        exceed = np.argwhere(diff > 1e-6)
        rows.append({"variable": name, "shape_match": True, "finite_mask_match": finite_match,
                     "early_max_abs": early_abs, "early_max_rel": early_rel,
                     "max_abs": max_abs, "nrmse": nrmse,
                     "first_full_abs_exceed_flat": int(np.ravel_multi_index(tuple(exceed[0]), diff.shape)) if exceed.size else None,
                     "early_pass": early_pass, "full_pass": full_pass})
        for idx in CHECKPOINTS:
            if idx < matlab.shape[-1]:
                raw_points.append({"variable": name, "time_index": idx,
                                   "matlab_flat": json.dumps(np.asarray(matlab[..., idx]).ravel().tolist()),
                                   "python_flat": json.dumps(np.asarray(python[..., idx]).ravel().tolist()),
                                   "max_abs": float(np.max(diff[..., idx]))})

    with (args.outdir / "pointwise_errors.csv").open("w", newline="") as f:
        fields = sorted({k for row in rows for k in row})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (args.outdir / "raw_checkpoints.csv").open("w", newline="") as f:
        fields = ("variable", "time_index", "matlab_flat", "python_flat", "max_abs")
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(raw_points)
    summary = {"graph": args.graph, "implementation": args.implementation,
               "early_gate_pass": all_early, "full_gate_pass": all_full,
               "variables_compared": len(rows), "mat_sha256": sha256(args.mat),
               "npz_sha256": sha256(args.npz), "tolerances_frozen": True}
    (args.outdir / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
