#!/usr/bin/env python3
import csv,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run_v11 as engine
C=json.loads((R/'V12_CONFIRMATORY_CONTRACT.json').read_text())
P=json.loads((R/'V12_FROZEN_PARAMETERS.json').read_text());pars={x['domain']:x for x in P['parameters']}
O=R/'results'
def one(t):
    x=engine.task(t);q=pars[x['domain']]
    x['psi_prediction']=(q['A_information']/(x['rho']*x['gamma'])+
                         q['B_mismatch']*x['rho']*x['heterogeneous_fraction']+
                         q['C_constraint']*x['rho']**2*x['heterogeneous_fraction']+
                         q['D_scale']*np.log(x['n']/5))
    x['predicted_success']=int(x['psi_prediction']<=1);return x
def main():
    tasks=[(d,n,k,r,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for r in C['rho_grid'] for s in C['new_seeds'] for p in C['policies']]
    with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(one,tasks,chunksize=40))
    path=O/'v12_confirmatory_runs.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    meta={'contract_sha256':hashlib.sha256((R/'V12_CONFIRMATORY_CONTRACT.json').read_bytes()).hexdigest(),
          'parameter_sha256':hashlib.sha256((R/'V12_FROZEN_PARAMETERS.json').read_bytes()).hexdigest(),
          'rows':len(rows),'unique_keys':len({(x['domain'],x['n'],x['topology'],x['rho'],x['seed'],x['policy']) for x in rows}),
          'scope':C['claim_boundary']}
    (O/'v12_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
