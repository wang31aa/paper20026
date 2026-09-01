#!/usr/bin/env python3
import csv,hashlib,json,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run_v11 as engine
C=json.loads((R/'V11_1_CONFIRMATORY_CONTRACT.json').read_text());P=json.loads((R/'V11_1_FROZEN_PARAMETERS.json').read_text())
O=R/'results';pars={x['domain']:x for x in P['parameters']}
def one(t):
 d,n,k,r,s,p=t;x=engine.task(t);q=pars[d];x['psi_prediction']=q['A_information']/(x['rho']*x['gamma'])+q['B_mismatch']*x['rho']*x['heterogeneous_fraction'];x['predicted_success']=int(x['psi_prediction']<=1);return x
def main():
 tasks=[(d,n,k,r,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for r in C['rho_grid'] for s in C['new_seeds'] for p in C['policies']]
 with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,tasks,chunksize=40))
 with (O/'v11_1_confirmatory_runs.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 meta={'contract_sha256':hashlib.sha256((R/'V11_1_CONFIRMATORY_CONTRACT.json').read_bytes()).hexdigest(),'parameter_sha256':hashlib.sha256((R/'V11_1_FROZEN_PARAMETERS.json').read_bytes()).hexdigest(),'rows':len(rows),'scope':C['claim_boundary']};(O/'v11_1_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
if __name__=='__main__':main()
