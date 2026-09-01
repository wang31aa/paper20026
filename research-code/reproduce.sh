#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MPLCONFIGDIR="/private/tmp/nature-v6-mpl-${USER:-runner}"
export XDG_CACHE_HOME="/private/tmp/nature-v6-xdg-${USER:-runner}"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$XDG_CACHE_HOME"

cd "$ROOT_DIR"
if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  PY="$(command -v python3)"
fi
"$PY" -c 'import numpy, pandas, scipy, matplotlib' 2>/dev/null || {
  echo "ERROR: selected Python lacks required packages; create/activate the environment from requirements.txt or set PYTHON_BIN." >&2
  exit 2
}

# Check the frozen archive before builders can refresh derived files.
if [ -f package_manifest.json ]; then
  "$PY" tools/validate_package_manifest.py package_manifest.json
fi
if [ -f AUDIT_SHA256.txt ]; then
  (cd "$ROOT_DIR" && shasum -a 256 -c AUDIT_SHA256.txt)
fi

"$PY" code/validate_results.py
"$PY" observer_rebuild/test_graph_and_equations.py
"$PY" observer_rebuild/validate_results.py observer_rebuild/results
"$PY" observer_in_loop_certified/test_equations.py
"$PY" observer_in_loop_certified/validate_r3_portable.py
"$PY" observer_in_loop_certified/validate_r4.py
"$PY" observer_in_loop_certified/validate_cluster_safety_budget.py
"$PY" boundary_law/derive_boundary_law.py
"$PY" boundary_law/validate.py
"$PY" cross_domain_v10/validate_v10.py
"$PY" cross_domain_v11/validate_v111.py
"$PY" cross_domain_v11/validate_v12.py
"$PY" cross_domain_v11/validate_v121.py
"$PY" cross_domain_v17/validate_v17.py
"$PY" cross_domain_v18/validate_v18.py
"$PY" parameter_qualification/validate_parameter_families.py
"$PY" cross_domain_v20/validate_v20.py
"$PY" cross_domain_v21/validate_v21.py
"$PY" cross_domain_v22/validate_v22.py
"$PY" audit/validate_central_evidence.py
"$PY" cross_domain_v10/analyse_v10.py
MPLBACKEND=Agg "$PY" cross_domain_v10/plot_v10.py
MPLBACKEND=Agg "$PY" cross_domain_v11/plot_v111.py
MPLBACKEND=Agg "$PY" cross_domain_v11/plot_v12.py
MPLBACKEND=Agg "$PY" cross_domain_v20/plot_v20.py
MPLBACKEND=Agg "$PY" figures/build_fig1_conditional_criticality.py
"$PY" discovery_extension/run_extension.py
"$PY" discovery_extension/validate.py
"$PY" physical_e2e_round34/validate_evidence_status.py
"$PY" physical_e2e_round34/validate_parameters.py \
  physical_e2e_round34/PARAMETER_TEMPLATE.json --draft-lint
PYTHONPATH=physical_e2e_round34 "$PY" -m unittest discover -v \
  -s physical_e2e_round34 -p 'test_*.py'
PYTHONPATH=physical_e2e_round34 "$PY" -m unittest discover -v \
  -s physical_e2e_round34/clean_room_host/tests -p 'test_*.py'
PYTHONPATH=physical_e2e_round34 "$PY" -m unittest discover -v \
  -s physical_e2e_round34/tests -p 'test_*.py'
"$PY" baseline_ablation/validate_results.py baseline_ablation/results
"$PY" extension_study/validate_results.py extension_study/results
"$PY" external_physical_validation/validate_results.py
"$PY" public_data/validate.py
"$PY" cps_transfer_benchmark/validate_results.py
(cd cps_transfer_benchmark && "$PY" -m unittest -v test_openmct_qualification.py)
(cd cps_transfer_benchmark && "$PY" -m unittest -v test_rths_design_identity.py)
"$PY" openmct_greybox_round42/validate.py \
  --module openmct_greybox_round42 --results openmct_greybox_round42/results
(cd openmct_greybox_round42 && "$PY" -m unittest -v test_greybox.py)
"$PY" openmct_modelset_round46/validate.py
(cd openmct_modelset_round46 && "$PY" -m unittest -v test_preflight.py)
(cd openmct_modelset_round46 && "$PY" reproduce_temp.py)
if [ -f public_cluster_cases/cache/robot_swarm_validation.csv ] && \
   [ -f public_cluster_cases/cache/uav_swarm_synthetic.csv ] && \
   [ -f public_cluster_cases/cache/adas_two_vehicle_sample.csv ]; then
  "$PY" public_cluster_cases/run_cases.py
  "$PY" public_cluster_cases/validate.py
else
  echo 'SKIP: public mobile-cluster raw caches are opt-in third-party downloads'
fi
"$PY" figures/build_cps_transfer_figure.py
MPLBACKEND=Agg "$PY" figures/build_fig1_discovery.py
"$PY" figures/build_openmct_figure.py
"$PY" figures/build_observer_in_loop_certified_figure.py
"$PY" figures/build_r4_mechanism.py
"$PY" figures/build_openmct_greybox_bridge.py
"$PY" tools/validate_observer_in_loop_figure.py
if [ "${FULL_CPS_REBUILD:-0}" = "1" ]; then
  : "${RTHS_RAW_DIR:?set RTHS_RAW_DIR to Data.v1.0.0/02_SMAWD}"
  : "${WATER_HIL_RAW_DIR:?set WATER_HIL_RAW_DIR to the water-HIL CSV directory}"
  : "${OPENMCT_RAW_DIR:?set OPENMCT_RAW_DIR to the extracted OpenMCT dataset root}"
  "$PY" cps_transfer_benchmark/rebuild_from_raw_and_compare.py \
    --rths-raw "$RTHS_RAW_DIR" --water-raw "$WATER_HIL_RAW_DIR" \
    --openmct-raw "$OPENMCT_RAW_DIR"
fi
if [ "${FULL_OPENMCT_GREYBOX_REBUILD:-0}" = "1" ]; then
  : "${OPENMCT_ARCHIVE:?set OPENMCT_ARCHIVE to the official checksum-bound V1 ZIP}"
  : "${OPENMCT_RAW_DIR:?set OPENMCT_RAW_DIR to the extracted OpenMCT dataset root}"
  "$PY" openmct_greybox_round42/validate.py \
    --module openmct_greybox_round42 --results openmct_greybox_round42/results \
    --archive "$OPENMCT_ARCHIVE" --raw "$OPENMCT_RAW_DIR"
fi
if [ "${FULL_EXTERNAL_REBUILD:-0}" = "1" ]; then
  : "${EXTERNAL_R1_DIR:?set EXTERNAL_R1_DIR to the extracted Zenodo R1 directory}"
  : "${EXTERNAL_R1_EDGES:?set EXTERNAL_R1_EDGES to Structure/Net_1.dat}"
  "$PY" external_physical_validation/rebuild_from_raw_and_compare.py \
    --raw-dir "$EXTERNAL_R1_DIR" --edges "$EXTERNAL_R1_EDGES"
fi
"$PY" theory/verify_absorbing_bound.py
"$PY" certified_benchmark/validate.py certified_benchmark/results
"$PY" matlab_reproduction/validate.py
"$PY" matlab_reproduction/test_intended_continuous_identity.py
"$PY" matlab_reproduction/validate_phase2.py
"$PY" matlab_branch_reproduction/source_contract_harness.py
"$PY" baseline_external/wang2022_ambiguity_audit.py

# Both frozen decision phases must remain structurally reproducible, and the
# registered Phase B peak-frequency gate must remain an explicit failure.
DECISION_LOG="$(mktemp "${TMPDIR:-/tmp}/nature-v6-decision.XXXXXX")"
if (cd decision_benchmark && "$PY" validate.py raw phase_b_raw >"$DECISION_LOG" 2>&1); then
  echo "ERROR: decision Phase B unexpectedly passed its frozen numerical gate" >&2
  cat "$DECISION_LOG" >&2
  exit 1
fi
grep -q "phase_b_raw: dt peak gate FAIL" "$DECISION_LOG"
grep -q "OVERALL: FAIL (pre-registered numerical gate)" "$DECISION_LOG"
cat "$DECISION_LOG"
rm -f "$DECISION_LOG"
echo 'PASS: registered decision-benchmark failure reproduced and retained'
"$PY" decision_benchmark/validate_posthoc_numerical.py \
  decision_benchmark/posthoc_numerical_raw

"$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np
p=Path('theory/result.json')
r=json.loads(p.read_text())
assert r['G_status']=='optimal' and r['Q_status']=='optimal'
assert r['d']>0 and r['a']>0
assert max(r['iqc_lmi_max_eigs']) <= 1e-7
assert abs(r['theorem_bound_using_finite_tail_envelope']-1.8076207696557125)<1e-10
print('PASS: archived static-theory values and numerical residuals validated')
PY

"$PY" tools/validate_original_figure_snapshot.py
"$PY" figures/build_figures.py
"$PY" figures/build_observer_figure.py
"$PY" figures/build_matlab_reproduction_figure.py
"$PY" tools/build_panel_provenance.py
"$PY" tools/validate_panel_provenance.py
"$PY" tools/validate_figure_fonts.py

if [ "${RUN_V23:-0}" = "1" ]; then
  "$PY" cross_domain_v23/run_v23.py
  "$PY" cross_domain_v23/validate_v23.py
  "$PY" cross_domain_v23/analyse_plot_v23.py
else
  "$PY" cross_domain_v23/validate_v23.py
fi

if [ "${RUN_V24:-0}" = "1" ]; then
  "$PY" cross_domain_v24/run_v24.py
  "$PY" cross_domain_v24/validate_v24.py
  "$PY" cross_domain_v24/analyse_v24.py
elif [ -f cross_domain_v24/results/V24_VALIDATION.json ]; then
  "$PY" cross_domain_v24/validate_v24.py
  "$PY" cross_domain_v24/analyse_v24.py
fi

if [ "${RUN_V25:-0}" = "1" ]; then
  "$PY" cross_domain_v25/run_v25.py
fi
if [ -f cross_domain_v25/results/V25_VALIDATION_ANALYSIS.json ]; then
  "$PY" cross_domain_v25/validate_analyse_v25.py
fi

if [ "${RUN_V26:-0}" = "1" ]; then
  "$PY" cross_domain_v26/run_v26.py
fi
if [ -f cross_domain_v26/results/V26_VALIDATION.json ]; then
  "$PY" cross_domain_v26/validate_v26.py
fi
"$PY" audit/parameter_flow_audit_v26.py

if [ "${RUN_V27:-0}" = "1" ]; then
  "$PY" cross_domain_v27/run_v27.py
fi
if [ -f cross_domain_v27/results/V27_VALIDATION.json ]; then
  "$PY" cross_domain_v27/validate_v27.py
fi
if [ "${RUN_V28:-0}" = "1" ]; then
  "$PY" cross_domain_v28/run_v28.py
fi
if [ -f cross_domain_v28/results/v28_runs.csv ]; then
  "$PY" cross_domain_v28/validate_v28.py
fi
if [ "${RUN_V29:-0}" = "1" ]; then
  "$PY" cross_domain_v28/fit_v28_critical.py
  "$PY" cross_domain_v29/run_v29.py
fi
if [ -f cross_domain_v29/results/v29_runs.csv ]; then
  "$PY" cross_domain_v29/validate_v29.py
  (cd figures && latexmk -pdf -interaction=nonstopmode -halt-on-error Fig_V28_V29_switching_ood.tex)
fi

if [ "${RUN_V30_V39:-0}" = "1" ]; then
  "$PY" cross_domain_v30/run_v30.py
  "$PY" cross_domain_v34/fit_v34_kernel.py
  "$PY" cross_domain_v35/run_validate_v35.py
  "$PY" cross_domain_v36/fit_v36.py
  "$PY" cross_domain_v37/run_validate_v37.py
  "$PY" cross_domain_v38/fit_v38.py
  "$PY" cross_domain_v39/run_validate_v39.py
fi
if [ -f cross_domain_v39/results/V39_VALIDATION.json ]; then
  "$PY" - <<'PY'
import json
x=json.load(open('cross_domain_v39/results/V39_VALIDATION.json'))
assert x['metrics']['water']['positives']>=40 and x['metrics']['water']['negatives']>=40
assert x['metrics']['circuit']['positives']>=40 and x['metrics']['circuit']['negatives']>=40
assert x['metrics']['water']['sensitivity']>=.8 and x['metrics']['water']['specificity']>=.8
assert not x['all_acceptance_pass']
print('PASS: V39 result and retained circuit non-promotion verified')
PY
fi

if [ "${RUN_VHIL_V3:-0}" = "1" ]; then
  "$PY" virtual_hil_v3/run_vhil_v3.py
fi
if [ -f virtual_hil_v3/results/VHIL_V3_QUALIFICATION.json ]; then
  "$PY" virtual_hil_v3/validate_vhil_v3.py
fi

MPLBACKEND=Agg "$PY" figures/build_fig1_v25.py

"$PY" audit/validate_active_parameter_observer_contract.py
"$PY" parameter_qualification/validate_parameter_families.py
"$PY" audit/validate_all_active_model_parameters.py
"$PY" audit/check_circuit_water_substeps.py

# Decisive two-layer theory and prospective prediction gates.  These checks
# deliberately preserve the distinction between theorem-class computation,
# within-domain prediction and physical/HIL evidence.
"$PY" theory/validate_kernel_approximation.py
"$PY" theory/validate_universal_capability_functional.py
"$PY" theory/validate_two_layer_capability.py
"$PY" theory/validate_general_participation_regime.py
"$PY" cross_domain_v44_frozen_prediction/validate_v44.py
"$PY" cross_domain_v45_unseen_thermal/validate_v45.py
"$PY" vehicle_v47_dense/validate_v47.py
"$PY" vehicle_v49_warning/validate_v49.py
"$PY" vehicle_v50_exact_kernel/validate_v50.py
"$PY" vehicle_v51_lag_refinement/validate_v51.py
"$PY" vehicle_v52_two_layer_platoon/validate_v52.py
"$PY" vehicle_v53_finite_game/run_v53.py
"$PY" vehicle_v53_finite_game/validate_v53.py
"$PY" uav_v54_sixdof/run_v54.py
"$PY" uav_v54_sixdof/validate_v54.py
"$PY" cross_domain_v55_transfer/run_v55.py
"$PY" cross_domain_v55_transfer/validate_v55.py
MPLBACKEND=Agg "$PY" figures/build_fig1_v55_decisive_evidence.py

if [ "${FULL_REBUILD:-0}" = "1" ]; then
  REBUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nature-v6-rebuild.XXXXXX")"
  trap 'rm -rf "$REBUILD_DIR"' EXIT
  "$PY" code/run_experiments.py --out "$REBUILD_DIR"
  "$PY" - "$REBUILD_DIR" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
root=Path('.')
new=Path(sys.argv[1])
for name in ('raw_runs.csv','summary.csv','dt_convergence.csv'):
    a=pd.read_csv(root/'code/results'/name)
    b=pd.read_csv(new/name)
    assert list(a.columns)==list(b.columns) and len(a)==len(b), name
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float),
                               rtol=2e-10, atol=1e-12, equal_nan=True), (name,col)
        else:
            assert a[col].astype(str).tolist()==b[col].astype(str).tolist(), (name,col)
assert json.loads((root/'code/results/metadata.json').read_text()) == json.loads((new/'metadata.json').read_text())
with np.load(root/'code/results/topologies.npz') as a, np.load(new/'topologies.npz') as b:
    assert a.files==b.files
    assert all(np.array_equal(a[k],b[k]) for k in a.files)
with np.load(root/'code/results/raw_representative_trajectories.npz') as a, np.load(new/'raw_representative_trajectories.npz') as b:
    assert a.files==b.files
    assert all(np.allclose(a[k],b[k],rtol=2e-10,atol=1e-12,equal_nan=True) for k in a.files)
print('PASS: clean simulation rerun matches frozen tables, arrays, topologies and metadata')
PY
fi

(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_Main.tex >/dev/null)
(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_Main.tex >/dev/null)
(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_SI_Submission.tex >/dev/null)
(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_SI_Submission.tex >/dev/null)
(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_Extended_Data.tex >/dev/null)
(cd manuscript && pdflatex -recorder -interaction=nonstopmode -halt-on-error NC_Rebuilt_Extended_Data.tex >/dev/null)

"$PY" tools/build_provenance_graph.py
"$PY" tools/build_submission_bundle.py
"$PY" tools/validate_submission_bundle.py
"$PY" tools/build_package_manifest.py
"$PY" tools/validate_package_manifest.py package_manifest.json
(cd "$ROOT_DIR" && shasum -a 256 -c AUDIT_SHA256.txt >/dev/null)

echo 'PASS: validation, figures, manuscript/SI compilation and checksum audit complete'
