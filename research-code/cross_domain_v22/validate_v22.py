#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
HERE=Path(__file__).resolve().parent; C=json.load(open(HERE/'V22_REGIME_FROZEN_CONTRACT.json')); R=HERE/'results'
dev=list(csv.DictReader(open(R/'v22_development.csv'))); held=list(csv.DictReader(open(R/'v22_heldout.csv'))); a=json.load(open(R/'v22_analysis.json'))
ed=len(C['regimes'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*len(C['development_seeds'])
eh=len(C['regimes'])*len(C['sizes'])*len(C['topologies'])*len(C['rho_grid'])*len(C['heldout_seeds'])
assert len(dev)==ed and len(held)==eh and a['rows']==ed+eh
assert len({(r['regime'],r['paired_replay_id']) for r in dev+held})==ed+eh
assert all(math.isfinite(float(r['minimum_physical_margin'])) and float(r['minimum_parameter_spread'])>0 for r in dev+held)
print(f'PASS: V22 complete ({ed} development, {eh} held-out); all adverse classes retained')
