#!/usr/bin/env python3
"""Frozen process-isolated virtual-HIL exercise for eight heterogeneous domains."""
from pathlib import Path
import csv, hashlib, json, math, multiprocessing as mp, time
import numpy as np

HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'VHIL_V1_FROZEN_CONTRACT.json').read_text())
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)

SPECS={
 'uav6dof': dict(p1=(.82,1.23),p2=(.10,.27),limit=(.68,1.05),target=.8,margin=.42,kind='second'),
 'vehicle': dict(p1=(.34,.88),p2=(.015,.046),limit=(1.4,2.5),target=14.,margin=2.2,kind='second'),
 'motor': dict(p1=(.026,.071),p2=(.018,.057),limit=(.75,1.18),target=1.2,margin=.36,kind='second'),
 'robot': dict(p1=(.72,1.31),p2=(.11,.29),limit=(.65,1.08),target=.7,margin=.38,kind='second'),
 'microgrid': dict(p1=(2.4,5.7),p2=(.55,1.35),limit=(.35,.72),target=0.,margin=.18,kind='second'),
 'circuit': dict(p1=(.78,1.36),p2=(.42,.93),limit=(.55,.94),target=.45,margin=.31,kind='first'),
 'water': dict(p1=(.76,1.42),p2=(.035,.092),limit=(.18,.39),target=1.1,margin=.28,kind='first'),
 'structure': dict(p1=(.76,1.38),p2=(.12,.36),limit=(.42,.86),target=0.,margin=.24,kind='second')}

def parameters(domain):
    rng=np.random.default_rng(C['seeds'][domain]); s=SPECS[domain]; n=C['nodes']
    return dict(p1=rng.uniform(*s['p1'],n),p2=rng.uniform(*s['p2'],n),limit=rng.uniform(*s['limit'],n))

def innovations(domain):
    rng=np.random.default_rng(C['seeds'][domain]+99)
    return rng.normal(0,.025,(C['cycles'],C['nodes']))

def plant(conn,domain,pars,x0,v0,noise):
    dt=C['sample_period_s']; s=SPECS[domain]; x=x0.copy(); v=v0.copy()
    for k in range(C['cycles']):
        conn.send({'kind':'measurement','cycle':k,'x':x.tolist(),'v':v.tolist(),'wall_ns':time.perf_counter_ns()})
        msg=conn.recv(); u=np.asarray(msg['applied'],float)
        bias=.018*np.sin(.071*k+np.arange(C['nodes']))+noise[k]
        if s['kind']=='second':
            acc=(u-pars['p2']*v-bias)/pars['p1']; v=v+dt*acc; x=x+dt*v
        else:
            flow=(pars['p1']*u-pars['p2']*x+bias); x=x+dt*flow; v=flow
        conn.send({'kind':'state','cycle':k,'x':x.tolist(),'v':v.tolist(),'wall_ns':time.perf_counter_ns()})
    conn.close()

def controller(conn,domain,policy,pars,outq):
    dt=C['sample_period_s']; s=SPECS[domain]; rows=[]; misses=0; n=C['nodes']
    for k in range(C['cycles']):
        start=time.perf_counter(); m=conn.recv(); x=np.asarray(m['x']); v=np.asarray(m['v'])
        residual=x-s['target']; trusted=np.abs(residual)<=s['margin']
        A=np.zeros((n,n),int)
        for i in range(1,n):
            if policy=='all_coupled' or trusted[i-1] or i-1==0: A[i,i-1]=1
        neighbour=np.array([sum(A[i,j]*(x[j]-x[i]) for j in range(n)) for i in range(n)])
        requested=-1.05*residual-.42*v+.34*neighbour
        if policy=='two_layer_gate': requested-=.18*residual*trusted
        if policy=='physical_filter': requested-=.32*np.tanh(residual/max(s['margin'],1e-9))
        applied=np.clip(requested,-pars['limit'],pars['limit'])
        conn.send({'kind':'control','cycle':k,'applied':applied.tolist()})
        nxt=conn.recv(); elapsed=time.perf_counter()-start; missed=int(elapsed>dt); misses+=missed
        margin=s['margin']-float(np.max(np.abs(np.asarray(nxt['x'])-s['target'])))
        rows.append(dict(domain=domain,policy=policy,cycle=k,time_s=k*dt,
          state_json=json.dumps(nxt['x']),velocity_json=json.dumps(nxt['v']),
          requested_control_json=json.dumps(requested.tolist()),applied_control_json=json.dumps(applied.tolist()),
          saturation_count=int(np.sum(np.abs(requested-applied)>1e-12)),adjacency_json=json.dumps(A.tolist()),
          edge_count=int(A.sum()),target_reachable=int(all(i==0 or A[i].any() for i in range(n))),
          task_margin=margin,communication_bytes=int(16*A.sum()),message_age_s=max(0.,(time.perf_counter_ns()-m['wall_ns'])/1e9),
          compute_time_s=elapsed,deadline_missed=missed))
        remain=dt-elapsed
        if remain>0: time.sleep(remain)
    outq.put((domain,policy,rows,misses)); conn.close()

def episode(domain,policy,outq):
    pars=parameters(domain); noise=innovations(domain); rng=np.random.default_rng(C['seeds'][domain]+7)
    target=SPECS[domain]['target']; x0=target+rng.normal(0,.16,C['nodes']); v0=rng.normal(0,.04,C['nodes'])
    a,b=mp.Pipe(); pp=mp.Process(target=plant,args=(a,domain,pars,x0,v0,noise)); cp=mp.Process(target=controller,args=(b,domain,policy,pars,outq))
    pp.start(); cp.start(); pp.join(); cp.join()
    if pp.exitcode or cp.exitcode: raise RuntimeError((domain,policy,pp.exitcode,cp.exitcode))

def main():
    q=mp.Queue(); results=[]
    # Keep the eight domain platforms independent and concurrent, but execute
    # policy batches sequentially so host oversubscription is not mistaken for
    # a plant/controller deadline failure.
    for p in C['policies']:
        for d in C['domains']:
            proc=mp.Process(target=episode,args=(d,p,q)); proc.start()
            results.append(q.get()); proc.join()
            if proc.exitcode: raise RuntimeError('episode process failed')
    rows=[r for _,_,rr,_ in results for r in rr]
    fields=list(rows[0]);
    with (OUT/'vhil_cycles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    total=len(rows); misses=sum(m for *_,m in results)
    domain_summary={}
    for d in C['domains']:
        dr=[r for r in rows if r['domain']==d]
        pars=parameters(d)
        domain_summary[d]={'rows':len(dr),'minimum_parameter_spread':min(float(np.ptp(v)) for v in pars.values()),
          'deadline_miss_fraction':sum(r['deadline_missed'] for r in dr)/len(dr),
          'policy_final_margins':{p:[r for r in dr if r['policy']==p][-1]['task_margin'] for p in C['policies']}}
    status={'schema':C['schema'],'contract_sha256':hashlib.sha256((HERE/'VHIL_V1_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
      'qualification':'process_isolated_virtual_hil','hardware_hil':False,'physical_experiment':False,'source_controller_replay':False,
      'separate_process_io':True,'domains':domain_summary,'rows':total,'deadline_misses':misses,
      'deadline_miss_fraction':misses/total,'claim_boundary':C['claim_boundary']}
    (OUT/'qualification_registry.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))

if __name__=='__main__': mp.set_start_method('spawn'); main()
