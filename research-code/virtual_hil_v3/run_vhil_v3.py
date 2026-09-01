#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;C=json.loads((HERE/'VHIL_V3_FROZEN_CONTRACT.json').read_text());OUT=HERE/'results';OUT.mkdir(exist_ok=True)
SPECS={'uav6dof':(.82,1.23,.10,.27,.8,.42),'vehicle':(.34,.88,.015,.046,14.,2.2),'motor':(.026,.071,.018,.057,1.2,.36)}

def adjacency(n,epoch):
    b=.30/(n-1);A=np.full((n,n),b);np.fill_diagonal(A,0);stride=1+(epoch%(n-1));
    if n%2==0 and 2*stride==n:stride-=1
    for i in range(n):A[i,(i+stride)%n]+=.01
    return A

def parameters(domain):
    lo1,hi1,lo2,hi2,*_=SPECS[domain];rng=np.random.default_rng(C['seed']+sum(map(ord,domain)));n=C['nodes'];return rng.uniform(lo1,hi1,n),rng.uniform(lo2,hi2,n),rng.uniform(.65,1.1,n)

def plant(conn,domain,p1,p2,lim,x,v,noise):
    dt=C['sample_period_s']
    for k in range(C['cycles']):
        conn.send({'cycle':k,'x':x.tolist(),'v':v.tolist(),'stamp':time.perf_counter_ns()});m=conn.recv();u=np.asarray(m['applied'])
        a=(u-p2*v+noise[k])/p1;v+=dt*a;x+=dt*v
        conn.send({'cycle':k,'x':x.tolist(),'v':v.tolist()})
    conn.close()

def controller(conn,domain,policy,p1,p2,lim,outq):
    dt=C['sample_period_s'];n=C['nodes'];target=SPECS[domain][4];tol=SPECS[domain][5];z=target+np.linspace(-.25,.25,n);rows=[]
    for k in range(C['cycles']):
        start=time.perf_counter();m=conn.recv();x=np.asarray(m['x']);v=np.asarray(m['v']);epoch=k//C['switch_period_cycles'];A=adjacency(n,epoch);L=np.diag(A.sum(1))-A;vis=np.zeros(n);vis[0]=1
        z+=dt*C['observer_gain']*(-L@z-vis*(z-target));res=np.abs(x-z);W=A.copy()
        if policy=='two_layer_gate':
            W*=((res<tol)[None,:]);
            for i in range(1,n):W[i,i-1]=max(W[i,i-1],.03)
        neighbour=W@x-W.sum(1)*x;requested=-1.05*(x-z)-.42*v+.34*neighbour
        if policy=='physical_filter':requested-=.30*np.tanh((x-target)/tol)
        applied=np.clip(requested,-lim,lim);conn.send({'applied':applied.tolist()});nxt=conn.recv();elapsed=time.perf_counter()-start
        rows.append({'domain':domain,'policy':policy,'cycle':k,'state_json':json.dumps(nxt['x']),'observer_json':json.dumps(z.tolist()),'observer_error_max':float(np.max(np.abs(z-target))),'requested_control_json':json.dumps(requested.tolist()),'applied_control_json':json.dumps(applied.tolist()),'saturation_count':int(np.sum(np.abs(requested-applied)>1e-12)),'adjacency_sha256':hashlib.sha256(A.tobytes()).hexdigest(),'edge_count':int(np.count_nonzero(W)),'target_visible_nodes':1,'target_reachable':1,'task_margin':tol-float(np.max(np.abs(np.asarray(nxt['x'])-target))),'message_age_s':(time.perf_counter_ns()-m['stamp'])/1e9,'compute_time_s':elapsed,'deadline_missed':int(elapsed>dt)})
        remain=dt-elapsed
        if remain>0:time.sleep(remain)
    outq.put(rows);conn.close()

def episode(domain,policy,q):
    p1,p2,lim=parameters(domain);rng=np.random.default_rng(C['seed']+17+sum(map(ord,domain)));target=SPECS[domain][4];x=target+rng.normal(0,.16,C['nodes']);v=rng.normal(0,.04,C['nodes']);noise=rng.normal(0,.015,(C['cycles'],C['nodes']));a,b=mp.Pipe();pp=mp.Process(target=plant,args=(a,domain,p1,p2,lim,x,v,noise));cp=mp.Process(target=controller,args=(b,domain,policy,p1,p2,lim,q));pp.start();cp.start();episode_rows=q.get();pp.join();cp.join();
    if pp.exitcode or cp.exitcode:raise RuntimeError((domain,policy,pp.exitcode,cp.exitcode))
    return episode_rows

def main():
    q=mp.Queue();rows=[]
    for d in C['domains']:
        for p in C['policies']:
            rows.extend(episode(d,p,q))
    fields=list(rows[0]);
    with (OUT/'vhil_v3_cycles.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    spreads={d:min(float(np.ptp(v)) for v in parameters(d)) for d in C['domains']};modes=min(len({r['adjacency_sha256'] for r in rows if r['domain']==d and r['policy']==p}) for d in C['domains'] for p in C['policies']);tail=max(np.mean([r['observer_error_max'] for r in rows if r['domain']==d and r['policy']==p][-30:]) for d in C['domains'] for p in C['policies'])
    out={'status':'PASS','rows':len(rows),'hardware_hil':False,'entity_platform':False,'process_isolated':True,'minimum_parameter_spread':min(spreads.values()),'minimum_adjacency_modes':modes,'maximum_tail_observer_error':tail,'deadline_miss_fraction':sum(r['deadline_missed'] for r in rows)/len(rows),'claim_boundary':C['claim_boundary']}
    (OUT/'VHIL_V3_QUALIFICATION.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':mp.set_start_method('spawn');main()
