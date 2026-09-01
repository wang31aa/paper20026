#!/bin/sh
set -eu
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" uav_causal_validation/validate.py
"$PYTHON_BIN" uav_causal_validation/validate_uav_v8.py
"$PYTHON_BIN" uav_causal_validation/analyze_uav_v8.py
if [ "${RERUN_UAV_V8:-0}" = "1" ]; then
  "$PYTHON_BIN" uav_causal_validation/run_uav_v8_dense_validate.py
  "$PYTHON_BIN" uav_causal_validation/validate_uav_v8.py
fi
if command -v /Volumes/wbh/opt/miniconda3/bin/python3.13 >/dev/null 2>&1; then
  /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/audit_official_records.py
  /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/test_double_integrator_translation.py
  /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/test_python_translation.py
  /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/test_heterogeneous_plant.py
  /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/test_participation_gate.py
  if [ -f /private/tmp/epfl_swarm_dataset.zip ]; then
    /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/qualify_archived_nmpc_causal_rollout.py /private/tmp/epfl_swarm_dataset.zip --output uav_causal_validation/results/archived_nmpc_causal_rollout_qualification.json
    /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/validate_uav_evidence_tiers.py
  fi
  if [ -n "${EPFL_PF_WORKSPACE:-}" ]; then
    /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/qualify_pf_replay.py "$EPFL_PF_WORKSPACE"
  fi
  if [ -n "${EPFL_NMPC_WORKSPACE:-}" ]; then
    /Volumes/wbh/opt/miniconda3/bin/python3.13 uav_causal_validation/qualify_translation.py "$EPFL_NMPC_WORKSPACE"
  fi
else
  echo "SciPy runtime unavailable: L0 metric rebuild skipped; qualification remains fail-closed" >&2
fi
"$PYTHON_BIN" uav_causal_validation/validate.py
