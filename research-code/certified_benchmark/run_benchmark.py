#!/usr/bin/env python3
"""Pre-registered, globally certified 3-D heterogeneous-network benchmark."""
from __future__ import annotations

import argparse, csv, hashlib, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

N, D = 5, 3
A0, B0 = 2.0, 0.6
ALPHA, GAMMA_MODEL, GAMMA_STATE = 4.0, 5.0, 35.0
DT, T_END, SAVE_DT = 0.002, 4.0, 0.02
SEEDS = tuple(range(10))
LEVELS = (0.05, 0.15, 0.30)
R = B0 / (A0 - B0)  # U=1


def forcing(t):
    return np.array([np.sin(t), np.cos(np.sqrt(2)*t), np.sin(np.sqrt(3)*t)]) / np.sqrt(3)


def topologies():
    # W[i,j]>0 means j -> i; leader is column N.
    out = {}
    W=np.zeros((N,N+1)); W[0,N]=1; W[1,0]=1; W[2,1]=1; W[3,2]=1; W[4,3]=1; out['chain']=W
    W=np.zeros((N,N+1)); W[:,N]=1; out['star']=W
    W=np.zeros((N,N+1)); W[0,N]=1; W[1,0]=1; W[2,0]=1; W[3,1]=1; W[3,2]=.5; W[4,2]=1; out['branch']=W
    W=np.zeros((N,N+1)); W[0,N]=1; W[1,0]=1; W[2,1]=1; W[3,2]=1; W[4,3]=1; W[0,4]=.25; W[2,0]=.4; out['cyclic']=W
    return out


def laplacian(W):
    L=np.zeros((N+1,N+1)); L[:N,:]=-W
    L[np.arange(N),np.arange(N)]+=W.sum(1)
    return L


def graph_constants(W):
    L=laplacian(W); L1=L[:N,:N]
    g=np.linalg.solve(L1.T,np.ones(N)); G=np.diag(g)
    S=G@L1+L1.T@G
    mu=np.linalg.eigvalsh(np.diag(g**-.5)@S@np.diag(g**-.5)).min()
    return L,g,float(mu),float(np.linalg.eigvalsh(S).min())


def node_parameters(level):
    pattern=np.array([-1.,-.5,0,.5,1.])
    # Always a_i>b_i>0, including the largest registered level.
    return A0*(1+.45*level*pattern), B0*(1+.70*level*pattern[::-1])


def certificate(W,level):
    _,g,mu,smin=graph_constants(W); a,b=node_parameters(level)
    Wi=np.abs(a-A0)*R+np.abs(b-B0)*(R+1.)
    omega=float(np.sqrt(np.sum(g*Wi**2)))
    ri=-2*(a-b); r=float(np.max(ri)); d=ALPHA*mu-r; rate=d # Q=I
    delta=2*omega/rate/np.sqrt(np.min(g))
    node_delta=2*omega/rate/np.sqrt(g)
    return dict(g=g,mu=mu,S_min=smin,a=a,b=b,W=Wi,omega=omega,r=r,d=d,rate=rate,
                delta=float(delta),node_delta=node_delta)


def pack(X, ah, bh, sh): return np.concatenate([X.ravel(),ah.ravel(),bh.ravel(),sh.ravel()])
def unpack(y):
    p=0; X=y[p:p+(N+1)*D].reshape(N+1,D); p+=(N+1)*D
    ah=y[p:p+N]; p+=N; bh=y[p:p+N]; p+=N; sh=y[p:].reshape(N,D)
    return X,ah,bh,sh


def rhs(t,y,L,a,b):
    X,ah,bh,sh=unpack(y); q=forcing(t)
    f=np.empty_like(X)
    f[:N]=-a[:,None]*X[:N]+b[:,None]*(np.tanh(X[:N])+q)
    f[N]=-A0*X[N]+B0*(np.tanh(X[N])+q)
    f[:N]-=ALPHA*(L[:N]@X) # common H=I
    aa=np.r_[ah,A0]; bb=np.r_[bh,B0]; s_aug=np.vstack([sh,X[N]])
    dah=-GAMMA_MODEL*(L[:N]@aa); dbh=-GAMMA_MODEL*(L[:N]@bb)
    dsh=-ah[:,None]*sh+bh[:,None]*(np.tanh(sh)+q)-GAMMA_STATE*(L[:N]@s_aug)
    return pack(f,dah,dbh,dsh)


def rk4(t,y,dt,L,a,b):
    k1=rhs(t,y,L,a,b); k2=rhs(t+dt/2,y+dt*k1/2,L,a,b)
    k3=rhs(t+dt/2,y+dt*k2/2,L,a,b); k4=rhs(t+dt,y+dt*k3,L,a,b)
    return y+dt*(k1+2*k2+2*k3+k4)/6


@dataclass
class Row:
    topology:str; heterogeneity:float; seed:int; dt:float; finite:bool
    delta:float; max_error:float; tail20_max_error:float; final_error:float
    max_certificate_ratio:float; tail20_certificate_ratio:float
    max_node_ratio:float; target_radius_max:float; observer_final:float


def simulate(name,W,level,seed,dt=DT,t_end=T_END,store=False):
    rng=np.random.default_rng(seed); c=certificate(W,level); L=laplacian(W); a,b=node_parameters(level)
    # Pre-registered support ensures target starts inside the invariant ball R.
    s0=rng.normal(size=D); s0*=rng.uniform(0,.9*R)/np.linalg.norm(s0)
    X=np.vstack([rng.uniform(-1,1,(N,D)),s0]); ah=rng.uniform(1.2,2.8,N)
    bh=rng.uniform(.2,1.,N); sh=rng.uniform(-.5,.5,(N,D)); y=pack(X,ah,bh,sh)
    V0=np.sum(c['g'][:,None]*(X[:N]-s0)**2); y0=np.sqrt(V0)
    steps=round(t_end/dt); stride=max(1,round(SAVE_DT/dt)); rec=[]; finite=True
    for k in range(steps+1):
        t=k*dt
        if k%stride==0 or k==steps:
            xx,aa,bb,ss=unpack(y); e=xx[:N]-xx[N]; V=np.sum(c['g'][:,None]*e**2)
            envelope=np.exp(-c['rate']*t/2)*y0+2*c['omega']/c['rate']*(1-np.exp(-c['rate']*t/2))
            global_ratio=np.sqrt(V)/max(envelope,1e-15)
            node_env=envelope/np.sqrt(c['g']); node_ratio=np.max(np.linalg.norm(e,axis=1)/node_env)
            obs=np.linalg.norm(ss-xx[N]); rec.append((t,np.linalg.norm(e),global_ratio,node_ratio,np.linalg.norm(xx[N]),obs))
        if k<steps:
            y=rk4(t,y,dt,L,a,b)
            if not np.all(np.isfinite(y)): finite=False; break
    z=np.asarray(rec); tail=z[:,0]>=.8*t_end
    row=Row(name,level,seed,dt,finite,c['delta'],float(z[:,1].max()),float(z[tail,1].max()),
            float(z[-1,1]),float(z[:,2].max()),float(z[tail,2].max()),float(z[:,3].max()),
            float(z[:,4].max()),float(z[-1,5]))
    return row,z if store else None


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(rows[0]))); w.writeheader(); w.writerows(asdict(x) for x in rows)


def main(out):
    out.mkdir(parents=True,exist_ok=True); tops=topologies(); rows=[]; series={}
    for name,W in tops.items():
        for level in LEVELS:
            for seed in SEEDS:
                row,z=simulate(name,W,level,seed,store=True); rows.append(row)
                series[f'{name}_{level:.2f}_seed{seed}']=z
    write_csv(out/'raw_runs.csv',rows); np.savez_compressed(out/'representative_timeseries.npz',**series)
    audits=[]
    for dt in (.004,.002,.001):
        for name,W in tops.items():
            for seed in range(2): audits.append(simulate(name,W,.30,seed,dt=dt)[0])
    write_csv(out/'dt_audit.csv',audits)
    certs={}
    for name,W in tops.items():
        certs[name]={}
        for level in LEVELS:
            c=certificate(W,level); certs[name][str(level)]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items()}
    meta=dict(preregistered=dict(N=N,D=D,levels=LEVELS,seeds=SEEDS,dt=DT,t_end=T_END,
        alpha=ALPHA,gamma_model=GAMMA_MODEL,gamma_state=GAMMA_STATE,A0=A0,B0=B0,U=1,R=R,
        gain_search=False,clipping=False,discarded_runs=0), certificates=certs,
        equations='Plant/controller and observers are specified in PROOF.md and implemented directly in rhs().')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(runs=len(rows),finite=sum(x.finite for x in rows),
      max_full_time_certificate_ratio=max(x.max_certificate_ratio for x in rows),
      max_target_radius=max(x.target_radius_max for x in rows),
      max_tail_over_delta=max(x.tail20_max_error/x.delta if x.delta else 0 for x in rows)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=Path(__file__).parent/'results'); main(p.parse_args().out)
