#!/usr/bin/env python3
"""Create frozen Phase 2 A/B and grid summaries plus plot-ready data."""

from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def metrics(d):
    e=d["Error"]; r=d["ReconstructionError"]
    q=max(1, e.size//10)
    return {"finite":bool(np.isfinite(e).all() and np.isfinite(r).all()),
            "error_final":float(e[-1]),"error_max":float(e.max()),"error_tail_rms":float(np.sqrt(np.mean(e[-q:]**2))),
            "recon_final":float(r[-1]),"recon_max":float(r.max()),"recon_tail_rms":float(np.sqrt(np.mean(r[-q:]**2)))}


def delta(a,b,key):
    x=a[key]; y=b[key]; d=x-y
    return {"max_abs":float(np.max(np.abs(d))),"rms":float(np.sqrt(np.mean(d*d)))}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--source',type=Path,required=True); p.add_argument('--graph',required=True); p.add_argument('--outdir',type=Path,required=True); a=p.parse_args()
    runs={h:np.load(a.root/f'{a.graph}_h{tag}.npz') for h,tag in [(0.001,'1e-3'),(0.0005,'5e-4'),(0.00025,'2p5e-4')]}
    legacy=loadmat(a.source,variable_names=[*(f'xx{i}' for i in range(1,9)),*(f'xg{i}' for i in range(1,8)),'Error'])
    lxx=np.stack([legacy[f'xx{i}'] for i in range(1,9)]); lxg=np.stack([legacy[f'xg{i}'] for i in range(1,8)])
    rows=[]
    for h,d in runs.items(): rows.append({'graph':a.graph,'h':h,**metrics(d)})
    grids=[]
    for h1,h2 in [(0.001,0.0005),(0.0005,0.00025)]:
        for key in ['xx','xg','Error','ReconstructionError']: grids.append({'graph':a.graph,'coarse_h':h1,'fine_h':h2,'variable':key,**delta(runs[h1],runs[h2],key)})
    ab=[]
    base=runs[0.001]
    for key,old in [('xx',lxx),('xg',lxg),('Error',legacy['Error'].squeeze())]: ab.append({'graph':a.graph,'variable':key,**delta(base,{key:old},key)})
    a.outdir.mkdir(parents=True,exist_ok=True)
    for name,data in [('run_metrics.csv',rows),('grid_differences.csv',grids),('legacy_ab.csv',ab)]:
        with (a.outdir/name).open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    t=np.arange(base['Error'].size)*0.001
    figure = np.column_stack((t, legacy['Error'].squeeze(), runs[0.001]['Error'],
                              runs[0.0005]['Error'], runs[0.00025]['Error']))
    np.savetxt(a.outdir/'figure_data.csv', figure, delimiter=',', fmt='%.17g',
               header='t,legacy_error,intended_h1e-3,intended_h5e-4,intended_h2p5e-4', comments='')
    print(json.dumps({'graph':a.graph,'runs':rows,'grids':grids,'legacy_ab':ab},indent=2))
if __name__=='__main__':main()
