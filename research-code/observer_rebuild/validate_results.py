#!/usr/bin/env python3
"""Independent checker that recomputes reported endpoints from raw arrays."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import run_observer_static as m


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main(out: Path, write_report: bool = False):
    rows={int(r['seed']):r for r in csv.DictReader((out/'raw_runs.csv').open())}
    z=np.load(out/'raw_trajectories.npz'); checks=[]
    for seed,row in rows.items():
        ys=z[f'seed_{seed}_state']; stored=z[f'seed_{seed}_metrics']
        recomputed=np.stack([m.metrics(y) for y in ys])
        diff=float(np.max(np.abs(stored-recomputed)))
        endpoints=[('final_observer_state_error',0),('final_model_error',1),
                   ('final_mismatch_estimation_error',2),('final_tracking_error',3)]
        endpoint_diff=max(abs(float(row[name])-recomputed[-1,j]) for name,j in endpoints)
        checks.append(dict(seed=seed,metric_array_max_abs_diff=diff,
                           csv_endpoint_max_abs_diff=float(endpoint_diff)))
    md=json.loads((out/'metadata.json').read_text()); ga=m.graph_audit()
    L0=np.linalg.norm(m.chua_matrices(8)[0],2)+np.linalg.norm(m.chua_matrices(8)[1],2)
    conv=list(csv.DictReader((out/'dt_convergence.csv').open()))
    by={(float(r['dt']),int(r['seed'])):r for r in conv}; rel=[]
    for seed in range(5):
        coarse=by[(.001,seed)]; fine=by[(.0005,seed)]
        for name in ('final_observer_state_error','final_model_error',
                     'final_mismatch_estimation_error','final_tracking_error'):
            a,b=float(coarse[name]),float(fine[name]); rel.append(abs(a-b)/max(abs(b),1e-8))
    report=dict(n_runs=len(rows), all_finite=all(r['finite']=='True' for r in rows.values()),
        raw_recompute_max_abs_diff=max(x['metric_array_max_abs_diff'] for x in checks),
        csv_recompute_max_abs_diff=max(x['csv_endpoint_max_abs_diff'] for x in checks),
        state_gain_margin=dict(gamma_mu=m.GAMMA_STATE*ga['generalized_margin_mu'],
                               required_2L0=2*L0,
                               pass_=bool(m.GAMMA_STATE*ga['generalized_margin_mu']>2*L0)),
        dt_convergence_max_relative_001_vs_0005=float(max(rel)),
        dt_convergence_all_finite=all(r['finite']=='True' for r in conv),
        metadata_identity=bool(md['dt']==m.DT and md['alpha']==m.ALPHA and
                               np.allclose(md['common_H'],m.H)), per_seed=checks)
    report['pass']=bool(report['all_finite'] and report['raw_recompute_max_abs_diff']<1e-12
        and report['csv_recompute_max_abs_diff']<1e-12 and report['state_gain_margin']['pass_']
        and report['dt_convergence_all_finite'] and report['dt_convergence_max_relative_001_vs_0005']<.02
        and report['metadata_identity'])
    if write_report:
        (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        files=['raw_runs.csv','raw_trajectories.npz','dt_convergence.csv','metadata.json','validation.json']
        (out/'SHA256SUMS.json').write_text(json.dumps({x:sha256(out/x) for x in files},indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='per_seed'},indent=2))
    if not report['pass']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('--write-report', action='store_true', help='refresh validation.json and SHA256SUMS.json')
    args=p.parse_args(); main(args.out, args.write_report)
