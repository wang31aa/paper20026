#!/usr/bin/env python3
"""Preregistered matched-information baselines for the rebuilt Chua example.

This file intentionally does not read any historical output or claimed bound.
All variants share plant equations, initial conditions, graph, controller gain,
integrator and horizon.  Only target information and diagnostic availability
change.  Rows of L are receivers: L[i,j] < 0 denotes j -> i.
"""
from __future__ import annotations

import argparse, csv, json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np

N, D = 7, 3
ALPHA, GAMMA_MODEL, GAMMA_STATE = 16.1, 6.0, 60.0
H = np.diag([12.0, 10.0, 11.0])
DT, T_END, SAVE_DT = .004, 2.0, .02
SEEDS, SCALES = tuple(range(20)), (.5, 1.0, 1.5)
TRACKING_THRESHOLD = .0035  # fixed before the factorial run
VARIANTS = ("oracle_static", "distributed_full", "state_only_known_model",
            "no_model_local", "distributed_uncertified")


def matrices(i):
    a, ell = 9.0 - .1*(i-1), 14.286 - .1*(i-1)
    A = np.array([[-a*2/7, a, 0], [1, -1, 1],
                  [0, -ell, .0005+.0001*i]], float)
    return A, np.diag([a*3/7, 0, 0])


def phi(x):
    z = np.zeros_like(x)
    z[..., 0] = .5*(np.abs(x[..., 0]+1)-np.abs(x[..., 0]-1))
    return z


@lru_cache(maxsize=1)
def laplacian():
    return np.array([[1,0,0,0,0,0,0,-1],[-2,2,0,0,0,0,0,0],
      [0,-2,2,0,0,0,0,0],[-2,0,-1,3,0,0,0,0],
      [0,0,-3,0,3,0,0,0],[0,0,0,0,0,1,0,-1],
      [0,0,0,0,0,0,1,-1],[0,0,0,0,0,0,0,0]], float)


@lru_cache(maxsize=None)
def models(scale):
    A0, B0 = matrices(8); aa=[]; bb=[]
    for i in range(1, 8):
        A, B = matrices(i); aa.append(A0+scale*(A-A0)); bb.append(B0+scale*(B-B0))
    return np.stack(aa), np.stack(bb), A0, B0


def pack(X, Ah, Bh, s): return np.r_[X.ravel(), Ah.ravel(), Bh.ravel(), s.ravel()]


def unpack(y):
    p=0; X=y[p:p+24].reshape(8,3); p+=24
    Ah=y[p:p+63].reshape(7,3,3); p+=63
    Bh=y[p:p+63].reshape(7,3,3); p+=63
    return X, Ah, Bh, y[p:p+21].reshape(7,3)


def rhs(y, variant, scale):
    X, Ah, Bh, s = unpack(y); L=laplacian(); As,Bs,A0,B0=models(scale)
    # Target used on leader-pinning links. Direct target access is oracle-only.
    pin_target = X[7] if variant == "oracle_static" else s
    mixed = L[:N,:N] @ X[:N] + L[:N,7,None] * pin_target
    intrinsic=np.einsum('nij,nj->ni',As,X[:N])+np.einsum('nij,nj->ni',Bs,phi(X[:N]))
    dX=np.empty_like(X); dX[:N]=intrinsic-ALPHA*(mixed@H.T)
    dX[N]=A0@X[N]+B0@phi(X[N])

    Aaug=np.concatenate([Ah,A0[None]]); Baug=np.concatenate([Bh,B0[None]])
    dAh=-GAMMA_MODEL*np.einsum('ij,jkl->ikl',L[:N],Aaug)
    dBh=-GAMMA_MODEL*np.einsum('ij,jkl->ikl',L[:N],Baug)
    if variant in ("state_only_known_model",):
        internal=s@A0.T+phi(s)@B0.T
    elif variant == "no_model_local":
        internal=np.einsum('nij,nj->ni',As,s)+np.einsum('nij,nj->ni',Bs,phi(s))
    else:
        internal=np.einsum('nij,nj->ni',Ah,s)+np.einsum('nij,nj->ni',Bh,phi(s))
    saug=np.vstack([s,X[N]])
    ds=internal-GAMMA_STATE*(L[:N]@saug)
    # Inactive states are kept constant; this avoids hidden information use.
    if variant in ("oracle_static", "state_only_known_model", "no_model_local"):
        dAh[:]=0; dBh[:]=0
    if variant == "oracle_static": ds[:]=0
    return pack(dX,dAh,dBh,ds)


def rk4(y, variant, scale):
    f=lambda q: rhs(q,variant,scale)
    k1=f(y); k2=f(y+.5*DT*k1); k3=f(y+.5*DT*k2); k4=f(y+DT*k3)
    return y+DT*(k1+2*k2+2*k3+k4)/6


def instantaneous(y, variant, scale):
    X,Ah,Bh,s=unpack(y); L=laplacian(); As,Bs,A0,B0=models(scale)
    target_used=X[7] if variant=="oracle_static" else s
    mixed=L[:N,:N]@X[:N]+L[:N,7,None]*target_used
    u=-ALPHA*(mixed@H.T)
    truth=np.linalg.norm((X[:N]-X[7]).ravel())
    obs=0. if variant=="oracle_static" else np.linalg.norm((s-X[7]).ravel())
    model=(0. if variant in ("oracle_static","state_only_known_model") else
           np.nan if variant=="no_model_local" else np.sqrt(np.sum((Ah-A0)**2)+np.sum((Bh-B0)**2)))
    proxy=truth if variant=="oracle_static" else np.linalg.norm((X[:N]-s).ravel())
    # Model-mismatch diagnostic: oracle knows truth; full observer reconstructs it.
    true_w=np.concatenate([(As[i]-A0)@X[7]+(Bs[i]-B0)@phi(X[7]) for i in range(N)])
    if variant in ("distributed_full","distributed_uncertified"):
        est_w=np.concatenate([(As[i]-Ah[i])@s[i]+(Bs[i]-Bh[i])@phi(s[i]) for i in range(N)])
        diagerr=np.linalg.norm(est_w-true_w)
    elif variant in ("oracle_static","state_only_known_model"):
        diagerr=0.
    else: diagerr=np.nan
    return np.array([truth,obs,model,np.linalg.norm(u),proxy,diagerr])


@dataclass
class Row:
    variant:str; seed:int; heterogeneity:float; finite:bool; samples:int
    final_tracking_error:float; tail20_max_tracking_error:float; q95_tracking_error:float
    final_observer_error:float; tail20_max_observer_error:float
    final_model_error:float; control_l2:float; peak_control:float
    supervisor_available:bool; mismatch_diagnostic_available:bool; final_mismatch_diagnostic_error:float
    false_alarm_rate:float; missed_violation_rate:float; decision_accuracy:float


def simulate(variant,seed,scale):
    rng=np.random.default_rng(seed); X=rng.uniform(-.2,.2,(8,3)); A0,B0=matrices(8)
    Ah=A0+rng.normal(0,.8,(N,D,D)); Bh=B0+rng.normal(0,.3,(N,D,D)); s=rng.uniform(-1,1,(N,D))
    y=pack(X,Ah,Bh,s); n=round(T_END/DT); stride=round(SAVE_DT/DT); mm=[]; finite=True
    for k in range(n+1):
        if k%stride==0: mm.append(instantaneous(y,variant,scale))
        if k<n:
            y=rk4(y,variant,scale)
            if not np.all(np.isfinite(y)): finite=False; break
    m=np.stack(mm); tail=int(.8*len(m)); truth=m[:,0]>TRACKING_THRESHOLD; pred=m[:,4]>TRACKING_THRESHOLD
    supervisor_avail=variant!="distributed_uncertified"
    mismatch_avail=variant not in ("distributed_uncertified","no_model_local")
    if supervisor_avail:
        fa=np.mean(pred & ~truth); miss=np.mean(~pred & truth); acc=np.mean(pred==truth)
    else: fa=miss=acc=np.nan
    control_l2=float(np.sqrt(np.trapezoid(m[:,3]**2,dx=SAVE_DT)))
    return Row(variant,seed,scale,finite,len(m),float(m[-1,0]),float(m[tail:,0].max()),float(np.quantile(m[:,0],.95)),
      float(m[-1,1]),float(m[tail:,1].max()),float(m[-1,2]),control_l2,float(m[:,3].max()),supervisor_avail,mismatch_avail,
      float(m[-1,5]) if mismatch_avail else np.nan,float(fa),float(miss),float(acc)),m

def simulate_task(task):
    vi,si,v,scale,seed=task
    row,m=simulate(v,seed,scale)
    return vi,si,seed,row,m


def main(out):
    out.mkdir(parents=True,exist_ok=True); rows=[]; arrays={}
    tasks=[(vi,si,v,scale,seed) for vi,v in enumerate(VARIANTS)
           for si,scale in enumerate(SCALES) for seed in SEEDS]
    with ThreadPoolExecutor(max_workers=8) as pool:
      for vi,si,seed,row,m in pool.map(simulate_task,tasks,chunksize=2):
        rows.append(row); arrays[f'v{vi}_s{si}_seed{seed}']=m
    with (out/'raw_runs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rows[0]))); w.writeheader(); w.writerows(asdict(r) for r in rows)
    np.savez_compressed(out/'raw_timeseries_metrics.npz',**arrays)
    summary=[]
    for v in VARIANTS:
      for scale in SCALES:
        z=[r for r in rows if r.variant==v and r.heterogeneity==scale]
        summary.append(dict(variant=v,heterogeneity=scale,n=len(z),finite_rate=np.mean([r.finite for r in z]),
          median_tail20_max_tracking_error=np.median([r.tail20_max_tracking_error for r in z]),
          median_control_l2=np.median([r.control_l2 for r in z]),median_final_observer_error=np.median([r.final_observer_error for r in z]),
          supervisor_available_rate=np.mean([r.supervisor_available for r in z]),
          mismatch_diagnostic_available_rate=np.mean([r.mismatch_diagnostic_available for r in z]),
          median_decision_accuracy=np.nanmedian([r.decision_accuracy for r in z]) if any(r.supervisor_available for r in z) else np.nan))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    meta=dict(preregistered_before_factorial_run=True,variants=VARIANTS,seeds=SEEDS,heterogeneity_scales=SCALES,
      dt=DT,t_end=T_END,save_dt=SAVE_DT,tracking_threshold=TRACKING_THRESHOLD,alpha=ALPHA,
      gamma_model=GAMMA_MODEL,gamma_state=GAMMA_STATE,common_H=H.tolist(),gain_search=False,clipping=False,
      historical_outputs_read=False,certificate_claimed=False,
      protocol_revision='Before any factorial output was produced, dt was changed from .002 to .004 and horizon from 4 to 2 s for computational feasibility; the change was not conditioned on results.',
      supervisor_truth='stacked true tracking error > fixed threshold',
      supervisor_prediction='available stacked local residual proxy > same fixed threshold',
      rates='unconditional sample fractions over full horizon; unavailable diagnostics are NaN/abstain')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'runs':len(rows),'finite':sum(r.finite for r in rows)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path(__file__).parent/'results'); main(p.parse_args().out)
