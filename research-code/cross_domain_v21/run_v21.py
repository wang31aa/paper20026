#!/usr/bin/env python3
"""Frozen time-scale-aware heterogeneous multi-motor computation."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, math, sys
import numpy as np
from functools import lru_cache

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
C=json.loads((HERE/'V21_MOTOR_FROZEN_CONTRACT.json').read_text())
sys.path.insert(0,str(ROOT/'parameter_qualification'))
from validate_parameter_families import sample_motor

DT=float(C['sample_period_s']); SUB=int(C['plant_substeps']); STEPS=round(C['duration_s']/DT)
OUT=HERE/'results'; OUT.mkdir(exist_ok=True)
CORRECTED_DISTRIBUTED_OBSERVER=False
DYNAMIC_SWITCHING=False
SWITCH_PERIOD_STEPS=round(0.6/DT)

def expm(a):
    """Deterministic scaling-and-squaring Taylor exponential for small matrices."""
    a=np.asarray(a,dtype=float); norm=float(np.linalg.norm(a,ord=np.inf))
    scale=max(0,int(np.ceil(np.log2(norm/.5)))) if norm>.5 else 0
    x=a/(2**scale); result=np.eye(len(a)); term=np.eye(len(a))
    for k in range(1,81):
        term=term@x/k; result+=term
        if np.linalg.norm(term,ord=np.inf)<=1e-15*max(1.,np.linalg.norm(result,ord=np.inf)): break
    for _ in range(scale): result=result@result
    return result

def adjacency(n,kind,seed):
    rng=np.random.default_rng(seed); A=np.zeros((n,n))
    if kind=='certified_switching':
        backbone=.30/max(n-1,1)
        A[:]=backbone; np.fill_diagonal(A,0)
        direction=1 if int(seed)%2 else -1
        stride=1+((int(seed)//2)%max(1,n-1))
        if n%2==0 and 2*stride==n: stride=max(1,stride-1)
        for i in range(n):
            j=(i+direction*stride)%n
            if i!=j: A[i,j]+=.01
        return A
    for i in range(1,n): A[i,i-1]=1
    if kind in ('random_directed','switching'):
        A=np.maximum(A,(rng.random((n,n))<min(.16,3/n)).astype(float)); np.fill_diagonal(A,0)
        for i in range(1,n): A[i,i-1]=1
    rows=A.sum(1); A[rows>0]/=rows[rows>0,None]
    return A

@lru_cache(None)
def observer_data(n,top,rho,epoch):
    offset=7919*epoch if DYNAMIC_SWITCHING and top in ('switching','certified_switching') else 0
    A=adjacency(n,top,7703+n+offset); visible=np.zeros(n); visible[:max(1,math.ceil(.2*n))]=1
    L=np.diag(A.sum(1))-A if CORRECTED_DISTRIBUTED_OBSERVER else np.eye(n)-A
    H=L+np.diag(visible); T=expm(-rho*H*DT)
    pin_gain=np.linalg.solve(H,(np.eye(n)-T)@visible) if CORRECTED_DISTRIBUTED_OBSERVER else None
    return A,visible,H,T,pin_gain

def execute(task):
    if len(task)==5:
        n,top,rho,seed,policy=task; exposure_scale=1.; authority_scale=1.
    else:
        n,top,rho,seed,policy,exposure_scale,authority_scale=task
    rng=np.random.default_rng(seed+113*n+19*C['topologies'].index(top))
    A,visible,H,T,pin_gain=observer_data(n,top,rho,0)
    ratios=sample_motor(rng,n)
    R=ratios['resistance_ratio']; L=.1*ratios['inductance_ratio']; J=.04*ratios['inertia_ratio']
    B=.03*ratios['friction_ratio']; Kt=ratios['torque_constant_ratio']; Ke=.08*np.ones(n)
    node_bias=rng.choice([-1.,1.],n)*rng.uniform(.08,.24,n)
    innovations=rng.normal(0,.015,(STEPS,n)); load_phase=rng.uniform(0,2*np.pi,n)
    omega=12+rng.normal(0,.08,n); load0=.16+.035*np.sin(load_phase)
    current=(B*omega+load0)/Kt; z=np.full(n,12.0); integ=np.zeros(n)
    kp=C['controller']['kp']; ki=C['controller']['ki']; imax=C['controller']['current_limit_a']*authority_scale; vmax=C['controller']['voltage_limit_v']*authority_scale
    margins=[]; errors=[]; energy=0.; messages=0.; first=C['duration_s']; recovery=0.
    for k in range(STEPS):
        epoch=k//SWITCH_PERIOD_STEPS if DYNAMIC_SWITCHING and top in ('switching','certified_switching') else 0
        A,visible,H,T,pin_gain=observer_data(n,top,rho,epoch)
        t=k*DT; target=12+2*np.sin(.13*t)
        if CORRECTED_DISTRIBUTED_OBSERVER:
            z=T@z+pin_gain*target
        else:
            z=target+T@(z-target)
        residual=np.abs(node_bias)+.035*np.abs(omega-z)
        if policy in ('connectivity_gate_pi','two_layer_pi'):
            W=A*(residual<.22)[None,:]
            for i in range(1,n): W[i,i-1]=max(W[i,i-1],.18)
        elif policy=='global_gain_reduction': W=.55*A
        elif policy=='decentralized_pi': W=np.zeros_like(A)
        else: W=A
        messages+=int(np.count_nonzero(W)); coupling=W@(omega+exposure_scale*node_bias)-W.sum(1)*omega
        err=z-omega; integ=np.clip(integ+DT*err,-C['controller']['antiwindup_limit'],C['controller']['antiwindup_limit'])
        load=.16+.055*np.sin(.20*t+load_phase)
        current_ff=(B*z+load)/Kt
        desired_current=current_ff+kp*err+ki*integ
        if policy in ('current_limited_pi','two_layer_pi'): desired_current=np.clip(desired_current,-.85*imax,.85*imax)
        cg=C['controller']['coupling_gain']*rho
        if policy=='global_gain_reduction': cg*=.55
        voltage=R*desired_current+Ke*omega+cg*coupling+innovations[k]
        voltage=np.clip(voltage,-vmax,vmax)
        for _ in range(SUB):
            h=DT/SUB
            current+=h*(voltage-R*current-Ke*omega)/L
            omega+=h*(Kt*current-B*omega-load)/J
        energy+=float(np.sum(np.abs(voltage*current))*DT)
        e=float(np.max(np.abs(omega-target))); margin=min(C['task']['speed_error_limit']-e,imax-float(np.max(np.abs(current))))
        errors.append(e); margins.append(margin)
        if t>=C['task']['settling_exclusion_s'] and margin<0 and first==C['duration_s']: first=t
        if margin>=0 and first<C['duration_s']: recovery+=DT
    start=round(C['task']['settling_exclusion_s']/DT); evaluated=margins[start:]
    spread={k:float(np.ptp(v)) for k,v in ratios.items()}
    return dict(domain='motor',n=n,topology=top,rho=rho,seed=seed,policy=policy,
        split='development' if seed in C['development_seeds'] else 'heldout',task_success=int(min(evaluated)>=0),
        minimum_physical_margin=min(evaluated),first_failure_time=first,recovery_dwell=recovery,
        tail_error=float(np.quantile(errors[-round(2/DT):],.95)),control_energy=energy,
        communication_messages=messages,minimum_parameter_spread=min(spread.values()),
        parameter_spread=json.dumps(spread,sort_keys=True),sample_period=DT,plant_substeps=SUB,
        exposure_scale=exposure_scale,authority_scale=authority_scale,
        paired_replay_id=f'motor|{n}|{top}|{rho}|{seed}',evidence_class=C['evidence_class'])

def main():
    seeds=C['development_seeds']+C['heldout_seeds']
    tasks=[(n,t,r,s,p) for n in C['sizes'] for t in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    with ProcessPoolExecutor(max_workers=6) as pool: rows=list(pool.map(execute,tasks,chunksize=18))
    fields=sorted({k for row in rows for k in row})
    with (OUT/'v21_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
    manifest={'contract_sha256':hashlib.sha256((HERE/'V21_MOTOR_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),
              'rows':len(rows),'unique_rows':len({(x['paired_replay_id'],x['policy']) for x in rows}),
              'evidence_class':C['evidence_class']}
    (OUT/'v21_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
