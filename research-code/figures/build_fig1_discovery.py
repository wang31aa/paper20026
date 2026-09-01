#!/usr/bin/env python3
"""Build Figure 1 with the bundled LaTeX vector backend (no Python plots)."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "figures"
SRC = HERE / "source_data"
SRC.mkdir(exist_ok=True)

copies = {
    ROOT / "theory/universal_capability_functional_validation.json": SRC / "Fig1_panel_b_capability_kernels.json",
    ROOT / "uav_causal_validation/results/uav_v8_curve_summary.csv": SRC / "Fig1_panel_c_uav_frozen_curve.csv",
    ROOT / "cross_domain_v18/results/V18_QUALIFICATION_REGISTRY.json": SRC / "Fig1_panel_d_out_of_grid_prediction.json",
    ROOT / "cross_domain_v19/results/V19_QUALIFICATION_REGISTRY.json": SRC / "Fig1_panel_d_system_specific_prediction.json",
}
for source, destination in copies.items():
    shutil.copyfile(source, destination)

subprocess.run(["pdftex", "--fmt=pdflatex", "-interaction=batchmode", "-halt-on-error",
                "Fig1_two_network_discovery.tex"], cwd=HERE, check=True)
subprocess.run(["pdftocairo", "-svg", "Fig1_two_network_discovery.pdf",
                "Fig1_two_network_discovery.svg"], cwd=HERE, check=True)
assert (HERE/"Fig1_two_network_discovery.pdf").stat().st_size > 10_000
assert (HERE/"Fig1_two_network_discovery.svg").stat().st_size > 10_000
print("PASS: theory-first Figure 1 PDF/SVG and source-data snapshots rebuilt")
