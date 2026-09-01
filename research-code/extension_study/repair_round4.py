#!/usr/bin/env python3
"""Round-4 repair: rerun only clean cells under continuous stage semantics.

The v1 directory is retained. No gains, seeds, horizon, graph, or acceptance
rules are changed. Non-clean sampled-data rows are migrated losslessly with
new derived normalization/failure-time fields.
"""
import csv, json, hashlib, math, os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import numpy as np
import run_extension as re

ROOT=Path(__file__).parent; OUT=ROOT/'results'; V1=ROOT/'results_v1_pre_round4'

def read(path): return list(csv.DictReader(path.open()))
def migrate(r):
    r=dict(r); n=int(r['n'])
    r['tail20_max_tracking_per_sqrt_n']=float(r['tail20_max_tracking'])/math.sqrt(n)
    r['final_tracking_per_sqrt_n']=float(r['final_tracking'])/math.sqrt(n)
    r['failure_time']=float(r['completed_steps'])*float(r['dt']) if r['failure_reason'] else re.T_END
    return r
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    workers=min(4,os.cpu_count() or 1)
    primary=[(m,n,'clean',s,re.DT,s==0) for m in re.MODELS for n in re.SIZES for s in re.SEEDS]
    clean_rows=[]; replacement={}
    with ThreadPoolExecutor(max_workers=workers) as ex:
      for task,(row,a,x) in ex.map(re.run_task,primary):
        clean_rows.append({k:str(v) if isinstance(v,bool) else v for k,v in asdict(row).items()})
        m,n,scenario,seed,_,_=task
        if seed==0:
          replacement[f'{m}_N{n}_clean_metrics']=a; replacement[f'{m}_N{n}_clean_states']=x
    old=[migrate(r) for r in read(V1/'raw_runs.csv') if r['scenario']!='clean']
    merged=old+clean_rows
    order={(m,n,s,k):i for i,(m,n,s,k) in enumerate((m,n,s,k) for m in re.MODELS for n in re.SIZES for s in re.SCENARIOS for k in re.SEEDS)}
    merged.sort(key=lambda r:order[(r['model'],int(r['n']),r['scenario'],int(r['seed']))])
    write(OUT/'raw_runs.csv',merged)
    oldnpz=np.load(V1/'raw_representative_trajectories.npz'); arrays={k:oldnpz[k] for k in oldnpz.files if '_clean_' not in k}; arrays.update(replacement)
    np.savez_compressed(OUT/'raw_representative_trajectories.npz',**arrays)

    audits=[(m,n,'clean',0,dt,False) for m in re.MODELS for n in re.SIZES for dt in (.004,.002,.001)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
      clean_audit=[asdict(result[0]) for _,result in ex.map(re.run_task,audits)]
    olda=[migrate(r) for r in read(V1/'dt_audit.csv') if r['scenario']!='clean']
    all_a=olda+clean_audit
    all_a.sort(key=lambda r:(re.MODELS.index(r['model']),int(r['n']),re.SCENARIOS.index(r['scenario']),float(r['dt'])))
    write(OUT/'dt_audit.csv',all_a)

    meta=json.loads((V1/'metadata.json').read_text()); meta['protocol_version']='1.1-round4-repair'
    meta['clean_semantics']='continuous equations: communication recomputed at every RK4 stage'
    meta['impaired_semantics']='sampled packets frozen over one RK4 step'
    meta['protocol_amendment']='A2 reviewer-round4: clean stage consistency; no gains or conditions changed'
    meta['normalization']='tracking norms additionally divided by sqrt(N); raw aggregate norms retained'
    meta['supersedes']='results_v1_pre_round4 (retained verbatim)'
    meta.pop('protocol_sha256',None); meta['protocol_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'recomputed_primary_clean':len(clean_rows),'recomputed_dt_clean':len(clean_audit),'merged_primary':len(merged),'merged_dt':len(all_a)},indent=2))
if __name__=='__main__': main()
