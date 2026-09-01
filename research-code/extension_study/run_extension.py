#!/usr/bin/env python3
"""Preregistered extension: common-H pinning plus distributed target observer.

All parameters are constants below.  The program never reads prior results,
never searches gains, never clips states, and writes every run including
non-finite and domain-exit runs.  Followers are N and the leader is index N.
Receiver-row convention: L[i,j] < 0 means j sends to i.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

SIZES = (8, 16, 32)
MODELS = ("chua", "lorenz")
SCENARIOS = ("clean", "noise", "delay", "dropout")
SEEDS = tuple(range(10))
DT, T_END, SAVE_DT = 0.002, 6.0, 0.02
ALPHA, GAMMA_MODEL, GAMMA_STATE = 12.0, 5.0, 35.0
NOISE_SD, DELAY_SECONDS, KEEP_PROB = 0.02, 0.04, 0.90
DOMAIN_RADIUS = {"chua": 50.0, "lorenz": 60.0}


def matrices(model: str, u: float) -> tuple[np.ndarray, np.ndarray]:
    """Return A and B; vector field is A x + B phi(x). u in [0,1]."""
    if model == "chua":
        a, ell = 9.0 - .7*u, 14.286 - .7*u
        A = np.array([[-2*a/7, a, 0], [1, -1, 1], [0, -ell, .0012]], float)
        B = np.diag([3*a/7, 0, 0])
    elif model == "lorenz":
        # leader u=1: sigma=10, rho=28, beta=8/3; followers differ <=5%.
        sigma, rho, beta = 9.5 + .5*u, 26.6 + 1.4*u, 2.8 - (2.8-8/3)*u
        A = np.array([[-sigma, sigma, 0], [rho, -1, 0], [0, 0, -beta]], float)
        B = np.eye(3)
    else: raise ValueError(model)
    return A, B


def phi(model: str, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    if model == "chua":
        out[..., 0] = .5*(np.abs(x[..., 0]+1)-np.abs(x[..., 0]-1))
    else:
        out[..., 1] = -x[..., 0]*x[..., 2]
        out[..., 2] = x[..., 0]*x[..., 1]
    return out


def graph(n: int) -> np.ndarray:
    """Deterministic balanced-depth rooted binary arborescence, weight 1."""
    L = np.zeros((n+1, n+1)); leader = n
    for i in range(n):
        parent = leader if i == 0 else (i-1)//2
        L[i, i] = 1.; L[i, parent] = -1.
    return L


def initial(seed: int, n: int, model: str):
    # Seed mapping is fixed and independent of scenario.
    rng = np.random.default_rng(1_000_000 + 10_000*n + 100*SEEDS.index(seed) + MODELS.index(model))
    if model == "chua": X = rng.uniform(-.3, .3, (n+1,3))
    else:
        X = np.tile(np.array([1.,1.,1.]), (n+1,1)) + rng.normal(0,.12,(n+1,3))
    A0,B0=matrices(model,1.); Ah=A0+rng.normal(0,.3,(n,3,3)); Bh=np.tile(B0,(n,1,1))
    sh=X[-1]+rng.normal(0,.25,(n,3))
    return X,Ah,Bh,sh


def pack(X,Ah,Bh,sh): return np.concatenate([X.ravel(),Ah.ravel(),Bh.ravel(),sh.ravel()])
def unpack(y,n):
    p=0; X=y[p:p+(n+1)*3].reshape(n+1,3); p+=(n+1)*3
    Ah=y[p:p+n*9].reshape(n,3,3); p+=n*9
    Bh=y[p:p+n*9].reshape(n,3,3); p+=n*9
    return X,Ah,Bh,y[p:].reshape(n,3)


def communicated(current, history, scenario, rng, dt):
    z = current.copy()
    if scenario == "delay":
        lag=max(1,round(DELAY_SECONDS/dt)); z=history[max(0,len(history)-1-lag)].copy()
    if scenario == "noise": z += rng.normal(0,NOISE_SD,z.shape)
    if scenario == "dropout":
        mask=rng.random(z.shape[0]) < KEEP_PROB
        held=history[-1] if history else current
        z=np.where(mask[:,None],z,held)  # lost broadcast: one-step held sample
    return z


def rhs(y,n,model,L,commX,commS,commA,commB,As,Bs,A0,B0):
    X,Ah,Bh,sh=unpack(y,n)
    intrinsic=np.einsum('nij,nj->ni',As,X[:n])+np.einsum('nij,nj->ni',Bs,phi(model,X[:n]))
    dX=np.empty_like(X); dX[:n]=intrinsic-ALPHA*(L[:n]@commX); dX[n]=A0@X[n]+B0@phi(model,X[n])
    dAh=-GAMMA_MODEL*np.einsum('ij,jkl->ikl',L[:n],commA)
    dBh=-GAMMA_MODEL*np.einsum('ij,jkl->ikl',L[:n],commB)
    internal=np.einsum('nij,nj->ni',Ah,sh)+np.einsum('nij,nj->ni',Bh,phi(model,sh))
    dsh=internal-GAMMA_STATE*(L[:n]@commS)
    return pack(dX,dAh,dBh,dsh)


@dataclass
class Row:
    model:str; n:int; scenario:str; seed:int; dt:float; planned_steps:int; completed_steps:int
    finite:bool; domain_exit:bool; max_state_norm:float; initial_tracking:float; final_tracking:float
    tail20_max_tracking:float; initial_observer:float; final_observer:float; tail20_max_observer:float
    final_model_error:float; tail20_max_tracking_per_sqrt_n:float; final_tracking_per_sqrt_n:float
    failure_time:float; failure_reason:str


def simulate(model,n,scenario,seed,dt=DT,save=False):
    L=graph(n); X,Ah,Bh,sh=initial(seed,n,model); y=pack(X,Ah,Bh,sh)
    A0,B0=matrices(model,1.)
    As=np.stack([matrices(model,(i+1)/(n+1))[0] for i in range(n)])
    Bs=np.stack([matrices(model,(i+1)/(n+1))[1] for i in range(n)])
    rng=np.random.default_rng(9_000_000+100_000*n+1000*seed+100*MODELS.index(model)+SCENARIOS.index(scenario))
    steps=round(T_END/dt); stride=max(1,round(SAVE_DT/dt)); histX=[X.copy()]; histS=[np.vstack([sh,X[-1]])]
    histA=[np.concatenate([Ah,matrices(model,1.)[0][None]]).reshape(n+1,-1)]
    histB=[np.concatenate([Bh,matrices(model,1.)[1][None]]).reshape(n+1,-1)]
    rec=[]; snapshots=[]; finite=True; domain=False; reason=""
    for k in range(steps+1):
        X,Ah,Bh,sh=unpack(y,n)
        tr=float(np.linalg.norm((X[:n]-X[n]).ravel())); ob=float(np.linalg.norm((sh-X[n]).ravel()))
        me=float(np.sqrt(np.sum((Ah-A0)**2)+np.sum((Bh-B0)**2))); mx=float(np.max(np.linalg.norm(X,axis=1)))
        if k%stride==0 or k==steps: rec.append((k*dt,tr,ob,me,mx))
        if save and (k%stride==0 or k==steps): snapshots.append(X.copy())
        if not np.all(np.isfinite(y)): finite=False; reason="non_finite"; break
        if mx>DOMAIN_RADIUS[model]: domain=True
        if k==steps: break
        curX=X.copy(); curS=np.vstack([sh,X[-1]]); curA=np.concatenate([Ah,A0[None]]); curB=np.concatenate([Bh,B0[None]])
        cx=communicated(curX,histX,scenario,rng,dt); cs=communicated(curS,histS,scenario,rng,dt)
        ca=communicated(curA.reshape(n+1,-1),histA,scenario,rng,dt).reshape(n+1,3,3)
        cb=communicated(curB.reshape(n+1,-1),histB,scenario,rng,dt).reshape(n+1,3,3)
        if scenario == 'clean':
            # Continuous C1/O1--O3: every RK4 stage evaluates communication
            # from that stage's state, rather than freezing the step-start packet.
            def f(z):
                zX,zA,zB,zS=unpack(z,n)
                return rhs(z,n,model,L,zX,np.vstack([zS,zX[-1]]),
                    np.concatenate([zA,A0[None]]),np.concatenate([zB,B0[None]]),As,Bs,A0,B0)
        else:
            # Impaired links are explicitly sampled-data: one packet realization per step.
            f=lambda z: rhs(z,n,model,L,cx,cs,ca,cb,As,Bs,A0,B0)
        k1=f(y); k2=f(y+.5*dt*k1); k3=f(y+.5*dt*k2); k4=f(y+dt*k3); y=y+dt*(k1+2*k2+2*k3+k4)/6
        histX.append(curX); histS.append(curS); histA.append(curA.reshape(n+1,-1)); histB.append(curB.reshape(n+1,-1))
        keep=max(2,round(DELAY_SECONDS/dt)+2)
        if len(histX)>keep:
            histX.pop(0); histS.pop(0); histA.pop(0); histB.pop(0)
    a=np.asarray(rec); tail=max(0,int(.8*len(a)))
    if domain and not reason: reason="finite_domain_exit"
    tail_track=float(a[tail:,1].max()); final_track=float(a[-1,1])
    row=Row(model,n,scenario,seed,dt,steps,k,finite,domain,float(a[:,4].max()),float(a[0,1]),final_track,
            tail_track,float(a[0,2]),float(a[-1,2]),float(a[tail:,2].max()),float(a[-1,3]),
            tail_track/np.sqrt(n),final_track/np.sqrt(n),float(k*dt if reason else T_END),reason)
    return row,a,np.asarray(snapshots) if save else None


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rows[0]))); w.writeheader(); w.writerows(asdict(r) for r in rows)

def run_task(task):
    model,n,scenario,seed,dt,save=task
    return task,simulate(model,n,scenario,seed,dt,save)


def main(out, audit_only=False):
    out.mkdir(parents=True,exist_ok=True); rows=[]; arrays={}
    tasks=[(m,n,s,k,DT,k==0) for m in MODELS for n in SIZES for s in SCENARIOS for k in SEEDS]
    workers=min(8,os.cpu_count() or 1)
    if not audit_only:
      with ThreadPoolExecutor(max_workers=workers) as ex:
       for task,(r,a,x) in ex.map(run_task,tasks):
        rows.append(r); model,n,scenario,seed,_,_=task
        if seed==0: arrays[f'{model}_N{n}_{scenario}_metrics']=a; arrays[f'{model}_N{n}_{scenario}_states']=x
      write_csv(out/'raw_runs.csv',rows); np.savez_compressed(out/'raw_representative_trajectories.npz',**arrays)
    audit_tasks=[(m,n,s,0,dt,False) for m in MODELS for n in SIZES for s in SCENARIOS for dt in (.004,.002,.001)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
      audit=[result[0] for _,result in ex.map(run_task,audit_tasks)]
    write_csv(out/'dt_audit.csv',audit)
    meta=dict(protocol_version='1.0-frozen-before-run',models=MODELS,sizes=SIZES,scenarios=SCENARIOS,seeds=SEEDS,
      dt=DT,t_end=T_END,save_dt=SAVE_DT,alpha=ALPHA,gamma_model=GAMMA_MODEL,gamma_state=GAMMA_STATE,
      common_H=np.eye(3).tolist(),noise_sd=NOISE_SD,delay_seconds=DELAY_SECONDS,packet_keep_probability=KEEP_PROB,
      gain_search=False,clipping=False,discarded_runs=0,primary_metrics=['final_tracking','tail20_max_tracking','final_observer'],
      applicability={'chua':'globally Lipschitz piecewise-linear nonlinearity',
       'lorenz':'quadratic field is not globally Lipschitz; finite-domain empirical extrapolation only, never a certificate'},
      failure_rule='report non-finite and any state norm above declared domain radius',domain_radius=DOMAIN_RADIUS,
      clean_semantics='continuous equations: communication recomputed at every RK4 stage',
      impairment_semantics={'noise':'independent Gaussian receiver packet noise per integration step',
       'delay':'constant 0.04 s delay with initial-history hold','dropout':'independent broadcast keep probability 0.90; loss uses one-step held sample'},
      rng='NumPy PCG64; deterministic formulas in source',parallel_workers=workers,
      protocol_amendment='A2 reviewer-round4: clean stage consistency; no gains or conditions changed')
    meta['protocol_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'runs_written_this_call':len(rows),'failures_in_primary_call':sum(bool(r.failure_reason) for r in rows),
      'nonfinite_in_primary_call':sum(not r.finite for r in rows),'domain_exits_in_primary_call':sum(r.domain_exit for r in rows),
      'dt_audit_runs':len(audit)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--audit-only',action='store_true'); a=p.parse_args(); main(a.out,a.audit_only)
