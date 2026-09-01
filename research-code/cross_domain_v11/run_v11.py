#!/usr/bin/env python3
"""Frozen heterogeneous three-domain causal computation for V11.

The models are deliberately independent and retain domain units.  They share
only the target observer, graph contract and policy interface.  Nothing in this
module is an official controller replay or a physical/HIL experiment.
"""
from __future__ import annotations
import csv, hashlib, json, math, time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'V11_FROZEN_CONTRACT.json').read_text())
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
DT=.08; STEPS=180
PARAMETER_MODE='legacy_independent'
# Later prospective experiments may enable these two corrections.  Defaults
# remain false so the frozen V11--V20 archives are byte-for-byte reproducible.
CORRECTED_DISTRIBUTED_OBSERVER=False
DYNAMIC_SWITCHING=False
CORRECTED_INDEPENDENT_TRACKING=False
SWITCH_PERIOD_STEPS=30

def domain_parameters(domain,rng,n,het):
    """Return plant parameters without changing the frozen V11 default.

    Later prospective experiments may set ``PARAMETER_MODE='coherent'``.  The
    legacy branch is byte-for-byte algebraically identical to V11--V12.
    """
    if PARAMETER_MODE=='coherent':
        import sys
        qroot=ROOT.parent/'parameter_qualification'
        if str(qroot) not in sys.path: sys.path.insert(0,str(qroot))
        from validate_parameter_families import sample_motor, sample_uav, sample_vehicle
        ratios={'uav6dof':sample_uav,'vehicle':sample_vehicle,'motor':sample_motor}[domain](rng,n)
        if domain=='uav6dof':
            p=dict(mass=ratios['mass_ratio'],drag=.16*ratios['drag_ratio'],bandwidth=3.8*ratios['bandwidth_ratio'],reserve=ratios['actuator_reserve_ratio'])
        elif domain=='vehicle':
            p=dict(tau=.5*ratios['lag_ratio'],authority=ratios['authority_ratio'],drag=.025*ratios['drag_ratio'])
        else:
            # V20 retains the nondimensional V11 motor equations.  These are
            # ratios around the declared computational baseline, not SI values.
            p=dict(R=ratios['resistance_ratio'],L=.1*ratios['inductance_ratio'],J=.04*ratios['inertia_ratio'],B=.03*ratios['friction_ratio'],Kt=ratios['torque_constant_ratio'])
        nominal={'uav6dof':dict(mass=1.,drag=.16,bandwidth=3.8,reserve=1.),'vehicle':dict(tau=.5,authority=1.,drag=.025),'motor':dict(R=1.,L=.1,J=.04,B=.03,Kt=1.)}[domain]
        return {k:np.where(het,v,nominal[k]) for k,v in p.items()}
    if domain=='uav6dof':
        return dict(mass=np.where(het,rng.uniform(.75,1.35,n),1.0),drag=np.where(het,rng.uniform(.10,.28,n),.16),bandwidth=np.where(het,rng.uniform(2.2,4.8,n),3.8),reserve=np.where(het,rng.uniform(.72,1.0,n),1.0))
    if domain=='vehicle':
        return dict(tau=np.where(het,rng.uniform(.35,.9,n),.5),authority=np.where(het,rng.uniform(.65,1.05,n),1),drag=np.where(het,rng.uniform(.015,.045,n),.025))
    return dict(R=np.where(het,rng.uniform(.7,1.5,n),1),L=np.where(het,rng.uniform(.06,.16,n),.1),J=np.where(het,rng.uniform(.025,.07,n),.04),B=np.where(het,rng.uniform(.018,.055,n),.03),Kt=np.where(het,rng.uniform(.75,1.2,n),1))

def adjacency(n,kind,seed):
    r=np.random.default_rng(seed); A=np.zeros((n,n))
    if kind=='certified_switching':
        # A dense symmetric backbone supplies a mode-independent contraction
        # margin.  A small directed circulation changes with ``seed`` and
        # makes every mode genuinely non-symmetric without removing that
        # margin.  Do not row-normalise this weighted family: the row
        # Laplacian below is defined for general non-negative weights.
        backbone=.30/max(n-1,1)
        A[:]=backbone; np.fill_diagonal(A,0)
        direction=1 if int(seed)%2 else -1
        stride=1+((int(seed)//2)%max(1,n-1))
        if n%2==0 and 2*stride==n: stride=max(1,stride-1)
        for i in range(n):
            j=(i+direction*stride)%n
            if i!=j: A[i,j]+=.01
        return A.astype(float)
    for i in range(1,n): A[i,i-1]=1
    if kind=='ring': A[0,-1]=1
    elif kind=='clustered':
        b=max(2,n//5)
        for i in range(n):
            for j in range(max(0,(i//b)*b),min(n,(i//b+1)*b)):
                if i!=j:A[i,j]=1
        for i in range(b,n,b):A[i,i-1]=1
    elif kind in ('random_directed','switching'):
        A=np.maximum(A,(r.random((n,n))<min(.16,3/n)).astype(float)); np.fill_diagonal(A,0)
        for i in range(1,n):A[i,i-1]=1
    s=A.sum(1); A[s>0]/=s[s>0,None]; return A.astype(float)

@lru_cache(None)
def graph_data(n,kind,visible_fraction,epoch=0):
    dynamic_offset=7919*epoch if DYNAMIC_SWITCHING and kind in ('switching','certified_switching') else 0
    A=adjacency(n,kind,11003+37*n+C['topologies'].index(kind)+dynamic_offset)
    vis=np.zeros(n); vis[:max(1,math.ceil(visible_fraction*n))]=1
    if CORRECTED_DISTRIBUTED_OBSERVER:
        L=np.diag(A.sum(1))-A
        H=L+np.diag(vis)
    else:
        H=np.eye(n)-A+np.diag(vis)
    P=solve_continuous_lyapunov(H.T,np.eye(n)); P=(P+P.T)/2
    gamma=1/(2*np.max(np.linalg.eigvalsh(P)))
    if gamma<=0: raise RuntimeError('uncertified target-rooted graph')
    return A,vis,H,float(gamma)

def condition(seed):
    if seed in C['heldout_seeds']: j=C['heldout_seeds'].index(seed)
    elif seed in C['development_seeds']: j=C['development_seeds'].index(seed)
    else: j=int(seed)%12
    vf=C['target_visible_fractions'][j%3]; hf=C['heterogeneous_fractions'][(2*j+1)%3]
    fault=C['heldout_faults'][j%4] if seed in C['heldout_seeds'] else 'development'
    return vf,hf,fault

def policy_edges(A,residual,policy):
    if CORRECTED_INDEPENDENT_TRACKING and policy=='independent_tracking':
        return np.zeros_like(A)
    if policy=='global_gain_reduction': return .55*A
    if policy=='residual_gate': return A*(residual<.45)[None,:]
    if policy in ('connectivity_gate','two_layer_gate'):
        W=A*(residual<.45)[None,:]
        # Preserve the frozen target-rooted predecessor chain as information backbone.
        for i in range(1,len(A)): W[i,i-1]=max(W[i,i-1],.22)
        return W
    return A

@lru_cache(None)
def observer_transition(n,topology,visible_fraction,rho,observer_mode,fault,epoch=0):
    _,_,H,_=graph_data(n,topology,visible_fraction,epoch)
    boost=1.35 if observer_mode=='boost' else 1.0
    delay=.72 if fault=='delay' else 1.0
    loss=.78 if fault=='packet_loss' else 1.0
    return expm(-rho*boost*delay*loss*H*DT)

def observer(z,target,T,H=None,vis=None,pin_gain=None):
    if CORRECTED_DISTRIBUTED_OBSERVER:
        if H is None or vis is None: raise ValueError('corrected observer requires H and visibility')
        if pin_gain is None: pin_gain=np.linalg.solve(H,(np.eye(len(H))-T)@vis)
        return T@z+pin_gain*target
    return target+T@(z-target)

@lru_cache(None)
def graph_for_epoch(n,topology,visible_fraction,rho,observer_mode,fault,epoch):
    A,vis,H,gamma=graph_data(n,topology,visible_fraction,epoch)
    T=observer_transition(n,topology,visible_fraction,rho,observer_mode,fault,epoch)
    pin_gain=np.linalg.solve(H,(np.eye(n)-T)@vis) if CORRECTED_DISTRIBUTED_OBSERVER else None
    return A,vis,H,gamma,T,epoch,pin_gain

def graph_at_step(n,topology,visible_fraction,rho,observer_mode,fault,k):
    epoch=k//SWITCH_PERIOD_STEPS if DYNAMIC_SWITCHING and topology in ('switching','certified_switching') else 0
    return graph_for_epoch(n,topology,visible_fraction,rho,observer_mode,fault,epoch)

def common(seed,n,topology,rho,hf,vf):
    rng=np.random.default_rng(seed+101*n+17*C['topologies'].index(topology))
    A,vis,H,gamma=graph_data(n,topology,vf)
    het=np.zeros(n,bool); het[rng.choice(n,max(1,math.ceil(hf*n)),False)]=1
    bias=np.where(het,rng.choice([-1.,1.],n)*rng.uniform(.18,.42,n),0)
    innovations=rng.normal(0,1,(STEPS,n,4))
    return rng,A,H,gamma,het,bias,innovations

def simulate_uav(n,topology,rho,seed,policy):
    vf,hf,fault=condition(seed); rng,A,H,gamma,het,bias,inn=common(seed,n,topology,rho,hf,vf)
    T=observer_transition(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault)
    pars=domain_parameters('uav6dof',rng,n,het); mass=pars['mass'];drag=pars['drag'];bandwidth=pars['bandwidth'];reserve=pars['reserve']
    offsets=-.5*np.arange(n,dtype=float)
    p=np.c_[offsets,np.zeros(n),1.5*np.ones(n)]; v=np.zeros((n,3)); ang=np.zeros((n,3)); rate=np.zeros((n,3)); z=np.zeros(n)
    tail=[]; margins=[]; energy=0.; messages=0; first=STEPS*DT; recovery=0.; t0=time.perf_counter()
    for k in range(STEPS):
        A,vis,H,_,T,_,pin_gain=graph_at_step(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault,k)
        target=.8*np.sin(.18*k*DT); z=observer(z,target,T,H,vis,pin_gain); residual=np.abs(bias)+np.linalg.norm(v,axis=1)*.05
        W=policy_edges(A,residual,policy); messages+=int(np.count_nonzero(W)); desired=np.c_[z+offsets,.3*np.sin(.11*k*DT+np.arange(n)/n),1.5*np.ones(n)]
        coupling=W@(p+bias[:,None])-W.sum(1)[:,None]*p
        acc=1.5*(desired-p)-1.1*v+.28*rho*coupling+.035*inn[k,:,:3]
        if policy in ('physical_filter','two_layer_gate'): # analytic separation/tilt filter
            order=np.argsort(p[:,0]); gaps=np.diff(p[order,0]);
            for q,g in enumerate(gaps):
                if g<.28: acc[order[q],0]-=.5; acc[order[q+1],0]+=.5
        if policy=='predictive_filter': acc-=.18*v
        if policy=='fault_tolerant_control': acc*=np.where(het[:,None],.82,1)
        lim=2.8*reserve[:,None]; acc=np.clip(acc,-lim,lim); v+=DT*(acc/mass[:,None]-drag[:,None]*v); p+=DT*v
        desired_ang=np.c_[np.clip(-acc[:,1]/9.81,-.55,.55),np.clip(acc[:,0]/9.81,-.55,.55),np.zeros(n)]
        rate+=DT*bandwidth[:,None]*(desired_ang-ang-rate/.8); ang+=DT*rate
        energy+=float(np.sum(np.linalg.norm(acc,axis=1)**2)*DT)
        truth=np.c_[target+offsets,.3*np.sin(.11*k*DT+np.arange(n)/n),1.5*np.ones(n)]
        e=float(np.max(np.linalg.norm(p-truth,axis=1))); sep=float(np.min(np.linalg.norm(p[:,None,:]-p[None,:,:]+np.eye(n)[:,:,None]*1e6,axis=2)))
        margin=min(.55-e,sep-.22,.65-float(np.max(np.abs(ang[:,:2]))),8-energy/(n*STEPS))
        tail.append(e); margins.append(margin)
        if k>=30 and margin<0 and first==STEPS*DT:first=k*DT
        if margin>=0 and first<STEPS*DT:recovery+=DT
    return outcome('uav6dof',n,topology,rho,seed,policy,gamma,vf,hf,fault,tail,margins,energy,messages,first,recovery,time.perf_counter()-t0)

def simulate_vehicle(n,topology,rho,seed,policy):
    vf,hf,fault=condition(seed); rng,A,H,gamma,het,bias,inn=common(seed,n,topology,rho,hf,vf)
    T=observer_transition(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault)
    pars=domain_parameters('vehicle',rng,n,het);tau=pars['tau'];authority=pars['authority'];drag=pars['drag']
    x=-40*np.arange(n,dtype=float); v=18*np.ones(n); acc=np.zeros(n); z=18*np.ones(n); tail=[];margins=[];energy=0;messages=0;first=STEPS*DT;recovery=0;t0=time.perf_counter()
    for k in range(STEPS):
        A,vis,H,_,T,_,pin_gain=graph_at_step(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault,k)
        target=18+1.8*np.sin(.09*k*DT); z=observer(z,target,T,H,vis,pin_gain); residual=np.abs(bias)+.04*np.abs(v-z)
        W=policy_edges(A,residual,policy);messages+=int(np.count_nonzero(W));coupling=W@(v+bias)-W.sum(1)*v
        exposure=bias if policy in ('all_coupled','global_gain_reduction','broadcast_boost') else bias*(residual<.45)
        cmd=.9*(z-v)+.80*rho*coupling+4.0*rho*rho*exposure+.04*inn[k,:,0]
        gaps=x[:-1]-x[1:]; safe=7+1.1*v[1:]
        if policy in ('physical_filter','two_layer_gate','predictive_filter'):
            risk=np.maximum(0,safe-gaps);cmd[1:]-=.32*risk
        if policy=='fault_tolerant_control':cmd*=np.where(het,.8,1)
        cmd=np.clip(cmd,-4.2*authority,2.5*authority);acc+=DT*(cmd-acc)/tau;v=np.maximum(0,v+DT*(acc-drag*v));x+=DT*v
        energy+=float(np.sum(cmd*cmd)*DT);gaps=x[:-1]-x[1:];head=gaps-5-.8*v[1:];amp=float(np.max(np.abs(v-v[0])))
        e=float(np.max(np.abs(v-target))); margin=min(float(np.min(head)),1.80-e,3.8-float(np.max(np.abs(acc))),2.5-amp)
        tail.append(e);margins.append(margin)
        if k>=30 and margin<0 and first==STEPS*DT:first=k*DT
        if margin>=0 and first<STEPS*DT:recovery+=DT
    return outcome('vehicle',n,topology,rho,seed,policy,gamma,vf,hf,fault,tail,margins,energy,messages,first,recovery,time.perf_counter()-t0)

def simulate_motor(n,topology,rho,seed,policy):
    vf,hf,fault=condition(seed); rng,A,H,gamma,het,bias,inn=common(seed,n,topology,rho,hf,vf)
    T=observer_transition(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault)
    pars=domain_parameters('motor',rng,n,het);R=pars['R'];L=pars['L'];J=pars['J'];B=pars['B'];Kt=pars['Kt']
    omega=12*np.ones(n);current=(B*omega+.16)/Kt;z=12*np.ones(n);tail=[];margins=[];energy=0;messages=0;first=STEPS*DT;recovery=0;t0=time.perf_counter()
    for k in range(STEPS):
        A,vis,H,_,T,_,pin_gain=graph_at_step(n,topology,vf,rho,'boost' if policy=='broadcast_boost' else 'base',fault,k)
        target=12+2*np.sin(.13*k*DT);z=observer(z,target,T,H,vis,pin_gain);residual=np.abs(bias)+.025*np.abs(omega-z)
        W=policy_edges(A,residual,policy);messages+=int(np.count_nonzero(W));coupling=W@(omega+bias)-W.sum(1)*omega
        voltage=1.5*(z-omega)+.18*rho*coupling+.03*inn[k,:,0]
        if policy in ('physical_filter','two_layer_gate','predictive_filter'):voltage-=.08*current
        if policy=='fault_tolerant_control':voltage*=np.where(het,.82,1)
        voltage=np.clip(voltage,-24,24);load=np.where(het,.2+.08*np.sin(.2*k*DT),.16)
        substeps=16 if PARAMETER_MODE=='coherent' else 4
        for _ in range(substeps):
            h=DT/substeps; current+=h*(voltage-R*current-.08*omega)/L;omega+=h*(Kt*current-B*omega-load)/J
        energy+=float(np.sum(np.abs(voltage*current))*DT);e=float(np.max(np.abs(omega-target)));margin=min(3.08-e,18-float(np.max(np.abs(current))),150-energy/n)
        tail.append(e);margins.append(margin)
        if k>=30 and margin<0 and first==STEPS*DT:first=k*DT
        if margin>=0 and first<STEPS*DT:recovery+=DT
    return outcome('motor',n,topology,rho,seed,policy,gamma,vf,hf,fault,tail,margins,energy,messages,first,recovery,time.perf_counter()-t0)

def outcome(domain,n,topology,rho,seed,policy,gamma,vf,hf,fault,tail,margins,energy,messages,first,recovery,wall):
    evaluated=margins[30:]
    return dict(domain=domain,n=n,topology=topology,rho=rho,gamma=gamma,eta=rho*gamma,seed=seed,policy=policy,visible_fraction=vf,heterogeneous_fraction=hf,fault=fault,task_success=int(min(evaluated)>=0),first_failure_time=first,minimum_physical_margin=min(evaluated),tail_error=float(np.quantile(tail[-45:],.95)),control_energy=energy,communication_messages=messages,wall_clock_compute_time=wall,recovery_dwell=recovery)

def task(t):
    domain,n,topology,rho,seed,policy=t
    return {'uav6dof':simulate_uav,'vehicle':simulate_vehicle,'motor':simulate_motor}[domain](n,topology,rho,seed,policy)

def write(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

def declare_parameters():
    # These coefficients are author-declared model parameters. The development
    # draws document the design grid; they do not statistically identify A,
    # D0, Db, U, decay or epsilon.
    rows=[]
    for domain in C['domains']:
        vals=[]
        for seed in C['development_seeds']:
            vf,hf,_=condition(seed);rng=np.random.default_rng(seed+len(domain));vals.append((vf,hf,float(rng.uniform(.92,1.08))))
        base={'uav6dof':(.42,.22,.48,.18,.52),'vehicle':(.34,.18,.40,.16,.58),'motor':(.28,.16,.36,.14,.62)}[domain]
        A,D0,Db,U,eps=base;rows.append(dict(domain=domain,A=A,D0=D0,Db=Db,U=U,decay=1.0,epsilon=eps,development_draws=vals))
    payload={'contract_sha256':hashlib.sha256((ROOT/'V11_FROZEN_CONTRACT.json').read_bytes()).hexdigest(),'parameters':rows,'status':'AUTHOR_DECLARED_BEFORE_HELDOUT','identification_performed':False}
    (ROOT/'V11_DECLARED_PARAMETERS.json').write_text(json.dumps(payload,indent=2)+'\n');return payload

def main():
    declared=declare_parameters();seeds=C['heldout_seeds'];tasks=[(d,n,k,r,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    with ProcessPoolExecutor(max_workers=6) as pool:rows=list(pool.map(task,tasks,chunksize=40))
    write(OUT/'v11_heldout_runs.csv',rows)
    meta={'contract_sha256':declared['contract_sha256'],'parameter_sha256':hashlib.sha256((ROOT/'V11_DECLARED_PARAMETERS.json').read_bytes()).hexdigest(),'parameter_status':declared['status'],'rows':len(rows),'unique_keys':len({(r['domain'],r['n'],r['topology'],r['rho'],r['seed'],r['policy']) for r in rows}),'scope':C['claim_boundary']}
    (OUT/'v11_run_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
