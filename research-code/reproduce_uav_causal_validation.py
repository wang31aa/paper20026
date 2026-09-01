#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def run(*cmd): subprocess.run(cmd,cwd=ROOT,check=True)
run(sys.executable,'uav_causal_validation/validate.py')
run(sys.executable,'uav_causal_validation/validate_uav_v8.py')
run(sys.executable,'uav_causal_validation/analyze_uav_v8.py')
scipy_python=Path('/Volumes/wbh/opt/miniconda3/bin/python3.13')
if scipy_python.exists():
    run(str(scipy_python),'uav_causal_validation/audit_official_records.py')
    run(str(scipy_python),'uav_causal_validation/test_double_integrator_translation.py')
    archive=Path('/private/tmp/epfl_swarm_dataset.zip')
    if archive.exists():
        run(str(scipy_python),'uav_causal_validation/qualify_archived_nmpc_causal_rollout.py',str(archive),'--output','uav_causal_validation/results/archived_nmpc_causal_rollout_qualification.json')
        run(str(scipy_python),'uav_causal_validation/validate_uav_evidence_tiers.py')
else:
    print('SciPy runtime missing; L0 metrics not rebuilt and status remains fail-closed',file=sys.stderr)
run(sys.executable,'uav_causal_validation/validate.py')
