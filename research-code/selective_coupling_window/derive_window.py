#!/usr/bin/env python3
"""Exploratory fixed-matrix calculation of two opposing exclusions.

This is not the variable-matrix calculation in the earlier frozen protocol;
see audit/SELECTIVE_COUPLING_PROTOCOL_DEVIATION.md.
"""
from pathlib import Path
import csv, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finite_resource_boundary"))
from derive_bounds import pinned_laplacians, graph_metric

OUT = Path(__file__).resolve().parent / "results"

def main():
    lam, alpha = 0.2, 4.0
    d0, db, authority = 0.1, 1.0, 0.1
    vs, h = 0.15, 0.2
    eps_peak, eps_ultimate = 0.20, 0.15
    data=[]
    for name, lap in pinned_laplacians().items():
        g, mu = graph_metric(lap)
        A = lam*np.eye(len(lap)) + alpha*lap
        gain = float(np.linalg.norm(np.linalg.solve(A, np.diag(g**-0.5)),2))
        for rho in np.arange(.05,1.0001,.05):
            load=d0+rho*db
            lower_u=max(load-authority,0)*gain
            exact_policy_upper=load*gain
            lyapunov_upper=2*load/((alpha*mu+2*lam)*np.sqrt(g.min()))
            lower_peak=vs*h/rho
            data.append(dict(topology=name,rho=float(rho),
                             exact_policy_upper=exact_policy_upper,
                             lyapunov_certificate_upper=lyapunov_upper,
                             lower_ultimate=lower_u,lower_peak_information=lower_peak,
                             exact_policy_certified=int(exact_policy_upper<=eps_ultimate),
                             lyapunov_certified=int(lyapunov_upper<=eps_ultimate),
                             ultimate_impossible=int(lower_u>eps_ultimate),
                             peak_impossible=int(lower_peak>eps_peak),
                             joint_not_ruled_out=int(lower_u<=eps_ultimate and lower_peak<=eps_peak)))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'selective_window.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0]); w.writeheader(); w.writerows(data)
    summary={"analysis_status": "exploratory_protocol_deviation",
             "model": "fixed graph matrix; rho scales candidate load only",
             "logical_scope": "ultimate-certified and peak-not-ruled-out; not a joint guarantee",
             "topologies": {}}
    for name in sorted({x['topology'] for x in data}):
        r=[x for x in data if x['topology']==name]
        summary["topologies"][name]={
          'peak_impossible_max_rho':round(max([x['rho'] for x in r if x['peak_impossible']],default=0),2),
          'exact_policy_certified_max_rho':round(max([x['rho'] for x in r if x['exact_policy_certified']],default=0),2),
          'lyapunov_certified_max_rho':round(max([x['rho'] for x in r if x['lyapunov_certified']],default=0),2),
          'ultimate_impossible_min_rho':round(min([x['rho'] for x in r if x['ultimate_impossible']],default=0),2),
          'ultimate_feasible_peak_not_ruled_out_rho':[round(x['rho'],2) for x in r if x['joint_not_ruled_out']],
        }
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
