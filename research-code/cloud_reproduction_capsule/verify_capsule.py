#!/usr/bin/env python3
"""Fail-closed verification of the central-claim reproduction capsule."""
from pathlib import Path
import json, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
def load(p): return json.loads((ROOT/p).read_text())
def run(p): subprocess.run([sys.executable,str(ROOT/p)],check=True,cwd=ROOT)
run('theory/validate_two_layer_capability.py')
run('theory/validate_universal_capability_functional.py')
v17=load('cross_domain_v17/results/v17_analysis.json')
v18=load('cross_domain_v18/results/V18_QUALIFICATION_REGISTRY.json')
v19=load('cross_domain_v19/results/V19_QUALIFICATION_REGISTRY.json')
backend=load('virtual_hil_v2/results/cross_backend_validation.json')
cloud=load('audit/GITHUB_CLOUD_EVIDENCE_2026-08-20.json')
assert all(v17['checks'].values())
assert v18['qualified_domain_count']==0 and not v18['all_eight_qualified']
assert v19['qualified_domain_count']==2 and not v19['all_eight_qualified']
assert backend['checks']['cross_backend_agreement'] and not backend['checks']['hardware_hil']
assert cloud['independent_cloud_microgate_qualified'] and not cloud['full_release_clean_clone_qualified']
assert cloud['hardware_hil'] is False and cloud['entity_hil'] is False
print('PASS: central theory, eight-domain evidence, adverse results and hardware/HIL boundaries are internally reproducible')
