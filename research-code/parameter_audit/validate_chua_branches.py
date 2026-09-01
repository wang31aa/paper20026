#!/usr/bin/env python3
"""Validate the immutable Chua branch ledger against independent artefacts.

This checker reads the rebuilt Python implementation, recovered MATLAB text,
archived MAT workspaces, result metadata and manuscript parameter table.  It
does not alter or evaluate any experimental trajectory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = Path(__file__).with_name("chua_branch_ledger.json")
REPORT_PATH = Path(__file__).with_name("chua_branch_validation.json")
BASE_H = np.diag([12.0, 10.0, 11.0])
ATOL = 1e-12


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("rebuild_chua_parameters", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_assignment(text: str, name: str) -> float:
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*([-+0-9.eE]+)\s*;?", text)
    if not match:
        raise ValueError(f"missing scalar assignment {name}")
    return float(match.group(1))


def matlab_parameters(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    a33 = []
    h_offsets = []
    compact_base = "[12,0,0;0,10,0;0,0,11]"
    for i in range(1, 9):
        am = re.search(rf"(?m)^\s*A{i}{i}\s*=\s*\[(.*?)\]\s*;?\s*$", text)
        if not am:
            raise ValueError(f"missing A{i}{i} in {path}")
        last_entry = am.group(1).split(";")[-1].split(",")[-1].strip()
        a33.append(float(last_entry))

        hm = re.search(rf"(?m)^\s*H{i}{i}\s*=\s*(\[[^\n]+?\])\s*(?:\+\s*([-+0-9.eE]+))?\s*;?\s*$", text)
        if not hm:
            raise ValueError(f"missing H{i}{i} in {path}")
        if re.sub(r"\s+", "", hm.group(1)) != compact_base:
            raise ValueError(f"unexpected H{i}{i} base matrix in {path}")
        h_offsets.append(float(hm.group(2) or 0.0))
    return {
        "A33": a33,
        "H_offsets": h_offsets,
        "coupling": scalar_assignment(text, "Coup"),
        "base_dt": scalar_assignment(text, "h"),
        "horizon": scalar_assignment(text, "Ftime"),
    }


def workspace_parameters(path: Path) -> dict:
    data = loadmat(path, squeeze_me=True)
    return {
        "A33": [float(np.asarray(data[f"A{i}{i}"])[2, 2]) for i in range(1, 9)],
        "H": [np.asarray(data[f"H{i}{i}"], dtype=float) for i in range(1, 9)],
        "coupling": float(np.asarray(data["Coup"]).item()),
        "base_dt": float(np.asarray(data["h"]).item()),
        "horizon": float(np.asarray(data["Ftime"]).item()),
        "stored_samples": int(np.asarray(data["Len"]).item()),
    }


def main() -> int:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    diagnostics: dict[str, object] = {}

    def check(name: str, value: bool) -> None:
        checks[name] = bool(value)

    rebuild = ledger["branches"]["REBUILD-R"]
    module = load_module(ROOT / rebuild["equation_source"])
    expected_a33 = np.asarray(rebuild["A33_values_nodes_1_to_8"], dtype=float)
    actual_a33 = np.asarray([module.chua_matrices(i)[0][2, 2] for i in range(1, 9)])
    check("REBUILD-R_A33_code", np.allclose(actual_a33, expected_a33, atol=ATOL, rtol=0))

    expected_h = np.stack([BASE_H + 0.1 * i * np.eye(3) for i in range(7)])
    check("REBUILD-R_H_diagonal_identity_semantics", np.allclose(module.H_LIST, expected_h, atol=ATOL, rtol=0))
    check("REBUILD-R_H_offdiagonal_zero", np.allclose(module.H_LIST - np.stack([np.diag(np.diag(x)) for x in module.H_LIST]), 0.0, atol=ATOL, rtol=0))
    check("REBUILD-R_coupling_code", abs(float(module.ALPHA) - rebuild["coupling"]) <= ATOL)
    check("REBUILD-R_dt_code", abs(float(module.BASE_DT) - rebuild["base_dt"]) <= ATOL)
    check("REBUILD-R_horizon_code", abs(float(module.T_END) - rebuild["horizon"]) <= ATOL)

    metadata = json.loads((ROOT / rebuild["result_metadata"]).read_text(encoding="utf-8"))
    check("REBUILD-R_coupling_metadata", abs(float(metadata["alpha_fixed"]) - rebuild["coupling"]) <= ATOL)
    check("REBUILD-R_dt_metadata", abs(float(metadata["base_dt"]) - rebuild["base_dt"]) <= ATOL)
    check("REBUILD-R_horizon_metadata", abs(float(metadata["t_end"]) - rebuild["horizon"]) <= ATOL)

    theory = json.loads((ROOT / rebuild["theory_result"]).read_text(encoding="utf-8"))
    check("REBUILD-R_common_theorem_H", np.allclose(theory["H"], rebuild["theorem_H"]["value"], atol=ATOL, rtol=0))
    check("REBUILD-R_theory_coupling", abs(float(theory["alpha"]) - rebuild["coupling"]) <= ATOL)
    check("REBUILD-R_L_phi_export", abs(float(theory["L_phi"]) - 1.0) <= ATOL)
    static_source = (ROOT / "theory/solve_static.py").read_text(encoding="utf-8")
    check("REBUILD-R_secondary_horizon_source", "T=200." in static_source)
    check("REBUILD-R_secondary_tail_source", "tail=50." in static_source)
    check("REBUILD-R_secondary_max_step_source", "max_step=.005" in static_source)

    diagnostics["REBUILD-R"] = {
        "A33_extracted": actual_a33.tolist(),
        "controller_H2_extracted": np.asarray(module.H_LIST[1]).tolist(),
        "controller_H7_extracted": np.asarray(module.H_LIST[6]).tolist(),
        "common_theorem_H_extracted": theory["H"],
        "coupling": float(module.ALPHA),
        "base_dt": float(module.BASE_DT),
        "horizon": float(module.T_END),
    }

    for branch_id in ("MATLAB-L/G1", "MATLAB-L/G2"):
        branch = ledger["branches"][branch_id]
        source_path = ROOT / branch["matlab_source"]
        workspace_path = ROOT / branch["mat_workspace"]
        key = branch_id.replace("/", "_")
        check(f"{key}_source_hash", sha256(source_path) == branch["source_sha256"])
        check(f"{key}_workspace_hash", sha256(workspace_path) == branch["workspace_sha256"])

        source = matlab_parameters(source_path)
        workspace = workspace_parameters(workspace_path)
        expected_a33 = np.asarray(branch["A33_values_nodes_1_to_8"], dtype=float)
        expected_offsets = np.arange(8, dtype=float) * 0.1
        expected_h = [BASE_H + offset * np.ones((3, 3)) for offset in expected_offsets]

        check(f"{key}_A33_source", np.allclose(source["A33"], expected_a33, atol=ATOL, rtol=0))
        check(f"{key}_A33_workspace", np.allclose(workspace["A33"], expected_a33, atol=ATOL, rtol=0))
        check(f"{key}_H_source_scalar_offsets", np.allclose(source["H_offsets"], expected_offsets, atol=ATOL, rtol=0))
        check(f"{key}_H_workspace_scalar_fill", all(np.allclose(x, y, atol=ATOL, rtol=0) for x, y in zip(workspace["H"], expected_h)))
        check(f"{key}_H2_offdiagonal_nonzero", abs(float(workspace["H"][1][0, 1]) - 0.1) <= ATOL)
        for field in ("coupling", "base_dt", "horizon"):
            expected = float(branch[field])
            check(f"{key}_{field}_source", abs(float(source[field]) - expected) <= ATOL)
            check(f"{key}_{field}_workspace", abs(float(workspace[field]) - expected) <= ATOL)
        check(f"{key}_stored_samples_workspace", workspace["stored_samples"] == int(branch["stored_samples"]))
        check(f"{key}_sample_identity", workspace["stored_samples"] == round(workspace["horizon"] / workspace["base_dt"]) + 1)

        diagnostics[branch_id] = {
            "A33_source": source["A33"],
            "A33_workspace": workspace["A33"],
            "H2_workspace": workspace["H"][1].tolist(),
            "H8_workspace": workspace["H"][7].tolist(),
            "coupling": workspace["coupling"],
            "base_dt": workspace["base_dt"],
            "horizon": workspace["horizon"],
            "stored_samples": workspace["stored_samples"],
        }

    main_tex = (ROOT / "manuscript/NC_Rebuilt_Main.tex").read_text(encoding="utf-8")
    si_tex = (ROOT / "manuscript/NC_Rebuilt_SI.tex").read_text(encoding="utf-8")
    adaptive_tex = (ROOT / "theory/adaptive_theorem.tex").read_text(encoding="utf-8")
    claim_ledger = (ROOT / "manuscript/CLAIM_LEDGER.md").read_text(encoding="utf-8")
    main_words = " ".join(main_tex.split())
    si_words = " ".join(si_tex.split())
    adaptive_words = " ".join(adaptive_tex.split())

    for token in ("\\texttt{REBUILD-R}", "\\texttt{MATLAB-L/G1}", "\\texttt{MATLAB-L/G2}"):
        check(f"SI_branch_token_{token}", token in si_tex)
    check("SI_A33_REBUILD_expression", "$0.0005+0.0001i$" in si_tex)
    check("SI_A33_MATLAB_expression", "$-(0.004+0.001i)$" in si_tex)
    check("SI_coupling_G1", "& 16.1 & 0.001 & 100" in si_tex)
    check("SI_coupling_G2", "& 10.8 & 0.001 & 100" in si_tex)
    check("manuscript_Chua_symbol_beta_A", "\\beta_i^A" in main_tex and "\\beta_i^A" in si_tex)
    check("manuscript_no_ell_i_collision", "\\ell_i" not in main_tex and "\\ell_i" not in si_tex)
    check("manuscript_L_phi_i_declared", "L_{\\phi,i}" in main_tex and "L_{\\phi,i}" in si_tex)
    check("manuscript_L_phi_0_declared", "L_{\\phi,0}" in main_tex and "L_{\\phi,0}" in si_tex)
    check("adaptive_main_not_archived_counterexample", "not claimed to refute the archived theorem" in main_words)
    check("adaptive_SI_not_archived_counterexample", "not presented as a counterexample to the archived theorem" in si_words)
    check("adaptive_theory_not_archived_counterexample", "not asserted to refute the archived theorem" in adaptive_words)
    check("adaptive_claim_ledger_downgraded", "no counterexample to the archived theorem is claimed" in claim_ledger)
    check("adaptive_pinning_symbol_unique", "b_i^{\\rm pin}" in adaptive_tex and "c_id_iF_i" not in adaptive_tex and "+d_i(x_i-\\hat s_i)" not in adaptive_tex)
    check("adaptive_observer_error_symbol_unique", "r_{\\rm obs}" in adaptive_tex and "\\mathcal R(c)r." not in adaptive_tex and "\\mathcal R(c(t))r(t)" not in si_tex)

    failed = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "scope": ledger["scope"],
        "checks": checks,
        "failed_checks": failed,
        "diagnostics": diagnostics,
        "ledger_sha256": sha256(LEDGER_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
