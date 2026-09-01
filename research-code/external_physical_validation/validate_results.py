#!/usr/bin/env python3
"""Validate released physical-data derivatives without redistributing raw data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

root = Path(__file__).resolve().parent
frame = pd.read_csv(root / "results/physical_metrics.csv")
saved = json.loads((root / "results/validation.json").read_text())
assert len(frame) == 303
assert set(frame.coupling_index) == set(range(101))
assert set(frame.repeat) == {1, 2, 3}
assert np.isfinite(frame.select_dtypes(include=["number"]).to_numpy()).all()
med = frame.groupby("coupling_index").normalized_disagreement.median()
rho, pvalue = spearmanr(med.index.to_numpy(), med.to_numpy())
assert abs(saved["endpoint_median_x0"] - med.loc[0]) < 1e-12
assert abs(saved["endpoint_median_x100"] - med.loc[100]) < 1e-12
assert abs(saved["spearman_rho"] - rho) < 1e-12
assert abs(saved["spearman_two_sided_p"] - pvalue) < 1e-105
assert all(frame.set_index(["repeat", "coupling_index"]).loc[(r, 100), "normalized_disagreement"] <
           frame.set_index(["repeat", "coupling_index"]).loc[(r, 0), "normalized_disagreement"]
           for r in (1, 2, 3))
print("PASS: 303 released physical-data metrics and registered summaries validated")

top = pd.read_csv(root / "results/topology_metrics.csv")
top_saved = json.loads((root / "results/topology_validation.json").read_text())
assert len(top) == 303 and np.isfinite(top.select_dtypes(include=["number"]).to_numpy()).all()
assert top.identity_abs_difference.max() <= 1e-10
top_med = top.groupby("coupling_index").normalized_edge_residual.median()
top_rho, top_p = spearmanr(top_med.index.to_numpy(), top_med.to_numpy())
assert abs(top_saved["endpoint_median_x0"] - top_med.loc[0]) < 1e-12
assert abs(top_saved["endpoint_median_x100"] - top_med.loc[100]) < 1e-12
assert abs(top_saved["spearman_rho"] - top_rho) < 1e-12
assert abs(top_saved["spearman_two_sided_p"] - top_p) < 1e-154
assert top_saved["connected_components"] == 1 and top_saved["undirected_edges"] == 42
print("PASS: 303 topology-aware metrics, graph identity and registered summaries validated")
