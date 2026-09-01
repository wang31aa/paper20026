#!/usr/bin/env python3
"""Analyse preregistered independent Rössler-circuit time series."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PATTERN = re.compile(r"ST_(\d+)_(\d+)\.dat$")


def metrics(path: Path) -> dict[str, float | int | str]:
    match = PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"unexpected filename: {path.name}")
    coupling_index, repeat = map(int, match.groups())
    values = np.loadtxt(path)
    if values.shape != (30000, 28) or not np.isfinite(values).all():
        raise ValueError(f"invalid data shape/values: {path} {values.shape}")
    values = values[6000:, :]
    disagreement = values - values.mean(axis=1, keepdims=True)
    node_rms = np.sqrt(np.mean(disagreement**2, axis=0))
    rms = float(np.sqrt(np.mean(disagreement**2)))
    centred = values - values.mean(axis=0, keepdims=True)
    amplitude = float(np.sqrt(np.mean(centred**2)))
    corr = np.corrcoef(values, rowvar=False)
    offdiag = corr[~np.eye(corr.shape[0], dtype=bool)]
    return {
        "file": path.name,
        "coupling_index": coupling_index,
        "repeat": repeat,
        "rms_disagreement": rms,
        "normalized_disagreement": rms / amplitude,
        "max_node_rms_disagreement": float(node_rms.max()),
        "mean_abs_offdiag_correlation": float(np.mean(np.abs(offdiag))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    files = sorted(args.input.rglob("ST_*_*.dat"))
    rows = [metrics(path) for path in files]
    frame = pd.DataFrame(rows).sort_values(["coupling_index", "repeat"])
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "physical_metrics.csv", index=False)

    complete = len(frame) == 303 and set(frame["coupling_index"]) == set(range(101)) and set(frame["repeat"]) == {1, 2, 3}
    medians = frame.groupby("coupling_index")["normalized_disagreement"].median()
    rho, pvalue = spearmanr(medians.index.to_numpy(), medians.to_numpy())
    endpoint_by_repeat = {}
    for repeat, group in frame.groupby("repeat"):
        indexed = group.set_index("coupling_index")["normalized_disagreement"]
        endpoint_by_repeat[str(int(repeat))] = bool(indexed.loc[100] < indexed.loc[0])
    result = {
        "input_file_count": int(len(frame)),
        "complete_303_file_grid": bool(complete),
        "endpoint_median_x0": float(medians.loc[0]),
        "endpoint_median_x100": float(medians.loc[100]),
        "endpoint_direction_pass": bool(medians.loc[100] < medians.loc[0]),
        "spearman_rho": float(rho),
        "spearman_two_sided_p": float(pvalue),
        "endpoint_direction_by_repeat": endpoint_by_repeat,
        "all_repeat_endpoint_pass": bool(all(endpoint_by_repeat.values())),
        "claim_boundary": "measurement-side physical residual only; no controller, observer, theorem or HIL validation",
    }
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
