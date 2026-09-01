#!/usr/bin/env python3
"""Fail-closed entry point for the confirmatory heterogeneous-UAV experiment."""
from __future__ import annotations
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
qualification=json.loads((HERE/'qualification_registry.json').read_text())
experiment=json.loads((HERE/'HETEROGENEOUS_EXPERIMENT_REGISTRY.json').read_text())
blocks=[]
if not qualification['levels']['L1_source_replay_qualified']:
    blocks.append('L1 source-controller replay has not passed every frozen criterion')
if not experiment['heterogeneous_plant_parameters']['identified_and_frozen']:
    blocks.append('per-agent heterogeneous plant parameters are not identified and frozen')
if not experiment['prediction_frozen']:
    blocks.append('the theory-derived participation prediction is not frozen')
if not experiment['paired_innovation_manifest']:
    blocks.append('the confirmatory paired-innovation manifest is absent')
if blocks:
    print(json.dumps({"status":"BLOCKED_FAIL_CLOSED","reasons":blocks,
                      "results_generated":False},indent=2))
    raise SystemExit(2)
raise SystemExit('Registry gates passed, but the official-source intervention adapter must be selected explicitly.')
