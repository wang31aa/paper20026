#!/usr/bin/env python3
"""Validate frozen Phase 2 files and nonconvergence disclosures."""
from __future__ import annotations
import csv, hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 errors=[]; count=0
 for line in (HERE/'PHASE2_SHA256.txt').read_text().splitlines():
  if not line.strip(): continue
  expected,raw=line.split('  ',1); p=ROOT/raw; count+=1
  if not p.is_file(): errors.append(f'missing {raw}')
  elif sha(p)!=expected: errors.append(f'hash mismatch {raw}')
 for graph in ('graph1','graph2'):
  d=HERE/'phase2'/f'{graph}_summary'
  rows=list(csv.DictReader((d/'run_metrics.csv').open()))
  if len(rows)!=3 or any(r['finite']!='True' for r in rows): errors.append(f'{graph}: runs not 3 finite')
  grids=list(csv.DictReader((d/'grid_differences.csv').open()))
  if len(grids)!=8: errors.append(f'{graph}: grid rows != 8')
  ab=list(csv.DictReader((d/'legacy_ab.csv').open()))
  if len(ab)!=3: errors.append(f'{graph}: A/B rows != 3')
  tags={'0.001':'h1e-3','0.0005':'h5e-4','0.00025':'h2p5e-4'}
  for row in rows:
   tag=tags.get(row['h'])
   if tag is None:
    errors.append(f"{graph}: unexpected h={row['h']}"); continue
   z=np.load(HERE/'phase2'/f'{graph}_{tag}.npz')
   if z['Error'].shape!=(100001,) or not np.isfinite(z['Error']).all(): errors.append(f'{graph}_{tag}: invalid error')
   # Independent observable identities: per-agent Euclidean norms and the
   # reported aggregate RMS across followers must agree with archived states.
   e_agents=np.linalg.norm(z['xx'][:7]-z['xx'][7][None,:,:],axis=1)
   r_agents=np.linalg.norm(z['xg']-z['xx'][7][None,:,:],axis=1)
   if not np.allclose(e_agents,z['Error_agents'],rtol=1e-12,atol=1e-13): errors.append(f'{graph}_{tag}: Error_agents identity')
   if not np.allclose(r_agents,z['ReconstructionError_agents'],rtol=1e-12,atol=1e-13): errors.append(f'{graph}_{tag}: ReconstructionError_agents identity')
   if not np.allclose(np.sqrt(np.mean(e_agents**2,axis=0)),z['Error'],rtol=1e-12,atol=1e-13): errors.append(f'{graph}_{tag}: Error aggregate identity')
   if not np.allclose(np.sqrt(np.mean(r_agents**2,axis=0)),z['ReconstructionError'],rtol=1e-12,atol=1e-13): errors.append(f'{graph}_{tag}: ReconstructionError aggregate identity')
   q=max(1,z['Error'].size//10); tail=slice(-q,None)
   expected={
    'error_final':z['Error'][-1], 'error_max':z['Error'].max(),
    'error_tail_rms':np.sqrt(np.mean(z['Error'][tail]**2)),
    'recon_final':z['ReconstructionError'][-1],
    'recon_max':z['ReconstructionError'].max(),
    'recon_tail_rms':np.sqrt(np.mean(z['ReconstructionError'][tail]**2)),
   }
   for key,value in expected.items():
    if not np.isclose(float(row[key]),value,rtol=2e-12,atol=1e-14): errors.append(f'{graph}_{tag}: summary {key}')
  runs={h:np.load(HERE/'phase2'/f'{graph}_{tag}.npz') for h,tag in tags.items()}
  grid_index={(r['coarse_h'],r['fine_h'],r['variable']):r for r in grids}
  for h1,h2 in (('0.001','0.0005'),('0.0005','0.00025')):
   for key in ('xx','xg','Error','ReconstructionError'):
    diff=runs[h1][key]-runs[h2][key]
    expected=(float(np.max(np.abs(diff))),float(np.sqrt(np.mean(diff*diff))))
    row=grid_index.get((h1,h2,key))
    if row is None: errors.append(f'{graph}: missing grid {h1}/{h2}/{key}'); continue
    if not np.isclose(float(row['max_abs']),expected[0],rtol=2e-12,atol=1e-14): errors.append(f'{graph}: grid max {h1}/{h2}/{key}')
    if not np.isclose(float(row['rms']),expected[1],rtol=2e-12,atol=1e-14): errors.append(f'{graph}: grid rms {h1}/{h2}/{key}')
  source=ROOT/'incoming/2026-07-28_author_matlab/source_tree/code'/f'exam1_{graph}.mat'
  legacy=loadmat(source,variable_names=[*(f'xx{i}' for i in range(1,9)),*(f'xg{i}' for i in range(1,8)),'Error'])
  old={'xx':np.stack([legacy[f'xx{i}'] for i in range(1,9)]),'xg':np.stack([legacy[f'xg{i}'] for i in range(1,8)]),'Error':legacy['Error'].squeeze()}
  ab_index={r['variable']:r for r in ab}
  for key in ('xx','xg','Error'):
   diff=runs['0.001'][key]-old[key]
   expected=(float(np.max(np.abs(diff))),float(np.sqrt(np.mean(diff*diff))))
   row=ab_index[key]
   if not np.isclose(float(row['max_abs']),expected[0],rtol=2e-12,atol=1e-14): errors.append(f'{graph}: A/B max {key}')
   if not np.isclose(float(row['rms']),expected[1],rtol=2e-12,atol=1e-14): errors.append(f'{graph}: A/B rms {key}')
 # Preserve the material negative result rather than accepting a rewritten summary.
 g2=list(csv.DictReader((HERE/'phase2/graph2_summary/grid_differences.csv').open()))
 vals={(r['coarse_h'],r['fine_h'],r['variable']):float(r['max_abs']) for r in g2}
 if vals[('0.0005','0.00025','ReconstructionError')] < 1.0: errors.append('Graph2 reconstruction nonconvergence disclosure altered')
 if errors:
  print('PHASE2_VALIDATION: FAIL'); [print('-',e) for e in errors]; raise SystemExit(1)
 print('PHASE2_VALIDATION: PASS')
 print(f'{count} hashes PASS; six finite runs; state identities and NPZ-to-summary/grid/legacy metrics PASS; Graph2 reconstruction nonconvergence preserved; corrected-init no-op and physical-time N/A documented.')
if __name__=='__main__': main()
