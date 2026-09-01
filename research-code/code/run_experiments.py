#!/usr/bin/env python3
"""Clean, equation-first Chua-network robustness experiments.

This module deliberately does not read historical NPZ/CSV files or published
Delta values.  It implements an oracle-target static pinning baseline.  It is
not a reproduction of the manuscript observer closed loop because the complete
observer/LMI gains are unavailable.
"""
from __future__ import annotations

import argparse, csv, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

N_FOLLOWERS, DIM = 7, 3
ALPHA = 16.1
H_LIST = np.stack([np.diag([12., 10., 11.]) + .1*i*np.eye(3) for i in range(N_FOLLOWERS)])
BASE_DT, T_END = .004, 4.0
HET_SCALES = (.25, .50, .75, 1.00, 1.25)
SEEDS = tuple(range(20))

def chua_matrices(i: int) -> tuple[np.ndarray, np.ndarray]:
    """Equation-108-style schedule, one-indexed, as stated in project source."""
    a = 9 - .1*(i-1); ell = 14.286 - .1*(i-1)
    A = np.array([[-a*2/7, a, 0], [1, -1, 1], [0, -ell, .0005+.0001*i]], float)
    B = np.diag([a*3/7, 0, 0])
    return A, B

def nonlinearity(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[..., 0] = .5*(np.abs(x[..., 0]+1)-np.abs(x[..., 0]-1))
    return out

def published_laplacian() -> np.ndarray:
    return np.array([[1,0,0,0,0,0,0,-1],[-2,2,0,0,0,0,0,0],
      [0,-2,2,0,0,0,0,0],[-2,0,-1,3,0,0,0,0],
      [0,0,-3,0,3,0,0,0],[0,0,0,0,0,1,0,-1],
      [0,0,0,0,0,0,1,-1],[0,0,0,0,0,0,0,0]], float)

def random_rooted_tree(seed: int) -> np.ndarray:
    """Random weighted arborescence with root 7 and edges parent -> child."""
    rng = np.random.default_rng(10000+seed); order = rng.permutation(7)
    L = np.zeros((8,8)); connected = [7]
    for child in order:
        parent = int(rng.choice(connected)); w = float(rng.uniform(.75, 2.25))
        L[child, child] += w; L[child, parent] -= w; connected.append(int(child))
    return L

def tree_valid(L: np.ndarray) -> bool:
    if not np.allclose(L.sum(1), 0): return False
    reached={7}
    while True:
        new={i for i in range(7) if any(L[i,j]<0 for j in reached)}
        old=len(reached); reached |= new
        if len(reached)==old: break
    return len(reached)==8

def scaled_models(scale: float):
    At,Bt=chua_matrices(8); As=[]; Bs=[]
    for i in range(1,8):
        Ai,Bi=chua_matrices(i); As.append(At+scale*(Ai-At)); Bs.append(Bt+scale*(Bi-Bt))
    return np.stack(As),np.stack(Bs),At,Bt

def rhs(y: np.ndarray, L: np.ndarray, models) -> np.ndarray:
    X=y.reshape(8,3); As,Bs,At,Bt=models
    intrinsic=np.einsum('nij,nj->ni',As,X[:7])+np.einsum('nij,nj->ni',Bs,nonlinearity(X[:7]))
    mixed=L[:7]@X
    coupling=-ALPHA*np.einsum('nij,nj->ni',H_LIST,mixed)
    d=np.empty_like(X); d[:7]=intrinsic+coupling
    d[7]=At@X[7]+Bt@nonlinearity(X[7]); return d.ravel()

def rk4(y, dt, L, models):
    k1=rhs(y,L,models); k2=rhs(y+.5*dt*k1,L,models)
    k3=rhs(y+.5*dt*k2,L,models); k4=rhs(y+dt*k3,L,models)
    return y+dt*(k1+2*k2+2*k3+k4)/6

def contraction_certificate(L: np.ndarray, scale: float, X0: np.ndarray, target_samples: np.ndarray):
    """Transparent Euclidean contraction bound; unavailable if margin <= 0.

    Uses the global Lipschitz slope |df1/dx1|<=1 and the *observed over the
    declared full horizon* oracle mismatch maximum. Hence this is a diagnostic
    oracle envelope, not an online observer certificate.
    """
    As,Bs,At,Bt=scaled_models(scale); M=np.zeros((21,21)); E=np.diag([1.,0,0])
    for i in range(7):
        si=slice(3*i,3*i+3); M[si,si]=As[i]+Bs[i]@E-ALPHA*L[i,i]*H_LIST[i]
        for j in range(7):
            if i!=j and L[i,j]!=0: M[si,3*j:3*j+3]=-ALPHA*L[i,j]*H_LIST[i]
    mu=-np.linalg.eigvalsh((M+M.T)/2).max()
    mismatch=[]
    for s in target_samples:
        fs=nonlinearity(s)
        mismatch.append(np.concatenate([(As[i]-At)@s+(Bs[i]-Bt)@fs for i in range(7)]))
    wmax=float(np.max(np.linalg.norm(mismatch,axis=1)))
    e0=float(np.linalg.norm((X0[:7]-X0[7]).ravel()))
    return mu,wmax,e0,(wmax/mu if mu>0 else np.nan)

@dataclass
class Row:
    topology:str; seed:int; heterogeneity:float; dt:float; steps:int; finite:bool
    full_max_error:float; final_error:float; q95_error:float; tail20_max_error:float
    contraction_margin:float; mismatch_full_horizon_max:float; initial_error:float
    asymptotic_oracle_envelope:float; coverage_asymptotic:float; conservatism_q95:float

def simulate(L, topology, seed, scale, dt=BASE_DT, save_series=False):
    rng=np.random.default_rng(seed); X0=rng.uniform(-.2,.2,(8,3)); y=X0.ravel().copy()
    models=scaled_models(scale)
    n=int(round(T_END/dt)); times=np.arange(n+1)*dt; errs=np.empty(n+1); targets=np.empty((n+1,3)); states=[]
    for k in range(n+1):
        X=y.reshape(8,3); errs[k]=np.linalg.norm((X[:7]-X[7]).ravel()); targets[k]=X[7]
        if save_series: states.append(X.copy())
        if k<n:
            y=rk4(y,dt,L,models)
            if not np.all(np.isfinite(y)): break
    finite=(k==n and np.all(np.isfinite(y))); used=k+1; errs=errs[:used]; targets=targets[:used]
    mu,wmax,e0,env=contraction_certificate(L,scale,X0,targets)
    cov=float(np.mean(errs<=env)) if np.isfinite(env) else np.nan
    q95=float(np.quantile(errs,.95)); cons=float(env/q95) if np.isfinite(env) and q95>0 else np.nan
    row=Row(topology,seed,scale,dt,used-1,finite,float(errs.max()),float(errs[-1]),q95,
      float(errs[max(0,int(.8*used)):].max()),mu,wmax,e0,env,cov,cons)
    series=None
    if save_series: series=(times[:used],np.stack(states),errs)
    return row,series

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rows[0]))); w.writeheader(); w.writerows(asdict(x) for x in rows)

def main(out: Path):
    out.mkdir(parents=True,exist_ok=True); trees={f'random_tree_{i}':random_rooted_tree(i) for i in range(5)}
    assert all(tree_valid(x) for x in trees.values())
    rows=[]; reps={}
    for name,L in trees.items():
      for scale in HET_SCALES:
       for seed in SEEDS:
        row,series=simulate(L,name,seed,scale,save_series=(seed==0))
        rows.append(row)
        if series is not None:
            t,X,e=series; reps[f'{name}_h{scale:g}_t']=t; reps[f'{name}_h{scale:g}_X']=X; reps[f'{name}_h{scale:g}_error']=e
    write_csv(out/'raw_runs.csv',rows); np.savez_compressed(out/'raw_representative_trajectories.npz',**reps)
    # Step-size convergence is separate and uses identical ICs/topology/horizon.
    conv=[]; L=published_laplacian()
    for dt in (.008,.004,.002,.001):
      for seed in range(5): conv.append(simulate(L,'published_laplacian',seed,1.,dt)[0])
    write_csv(out/'dt_convergence.csv',conv)
    np.savez_compressed(out/'topologies.npz',published=published_laplacian(),**trees)
    # Group summaries preserve failures and unavailable certificates.
    summary=[]
    for name in trees:
      for scale in HET_SCALES:
        rr=[r for r in rows if r.topology==name and r.heterogeneity==scale]
        cert=[r for r in rr if np.isfinite(r.asymptotic_oracle_envelope)]
        summary.append(dict(topology=name,heterogeneity=scale,n=len(rr),finite_rate=np.mean([r.finite for r in rr]),
          certificate_available_rate=len(cert)/len(rr),median_full_max=np.median([r.full_max_error for r in rr]),
          median_final=np.median([r.final_error for r in rr]),median_coverage=np.median([r.coverage_asymptotic for r in cert]) if cert else np.nan,
          median_conservatism=np.median([r.conservatism_q95 for r in cert]) if cert else np.nan))
    with (out/'summary.csv').open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
    metadata={'baseline':'oracle-target static pinning','alpha_fixed':ALPHA,'gain_search':False,'state_clipping':False,
      'burn_in_removed':False,'tail_metric_declared':'last 20% additionally reported; full-horizon metrics are primary',
      'old_artifacts_read':False,'published_delta_used':False,'observer_comparable':False,
      'reason':'complete observer and LMI decision variables are not archived','n_random_trees':5,'n_seeds':20,
      'heterogeneity_scales':HET_SCALES,'t_end':T_END,'base_dt':BASE_DT}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).parent/'results');a=p.parse_args();main(a.out)
