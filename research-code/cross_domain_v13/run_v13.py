#!/usr/bin/env python3
"""Frozen five-domain heterogeneous intervention computation.

Each domain has a distinct plant and physical/task endpoint.  The shared
interface is limited to a target observer, graph participation parameter and
two-layer gate.  These are author-defined computations, not HIL experiments.
"""
from __future__ import annotations
import csv, json, math, sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov

R=Path(__file__).resolve().parent; C=json.loads((R/'V13_FROZEN_CONTRACT.json').read_text())
O=R/'results'; O.mkdir(exist_ok=True); DT=.06; K=160

def adjacency(n,kind,seed):
    g=np.random.default_rng(seed); A=np.zeros((n,n))
    for i in range(1,n): A[i,i-1]=1
    if kind=='ring': A[0,-1]=1
    elif kind=='clustered':
        b=max(2,n//5)
        for i in range(n):
            for j in range((i//b)*b,min(n,(i//b+1)*b)):
                if i!=j:A[i,j]=1
        for i in range(b,n,b):A[i,i-1]=1
    elif kind in ('random_directed','switching'):
        A=np.maximum(A,g.random((n,n))<min(.18,3/n));np.fill_diagonal(A,0)
    s=A.sum(1);A[s>0]/=s[s>0,None];return A

@lru_cache(None)
def graph(n,kind,vf):
    A=adjacency(n,kind,13001+31*n+C['topologies'].index(kind));v=np.zeros(n);v[:max(1,math.ceil(vf*n))]=1
    H=np.eye(n)-A+np.diag(v);P=solve_continuous_lyapunov(H.T,np.eye(n));P=(P+P.T)/2
    return A,H,float(1/(2*np.max(np.linalg.eigvalsh(P))))

def condition(seed):
    if seed in C['development_seeds']:j=C['development_seeds'].index(seed)
    elif seed in C['confirmation_seeds']:j=C['confirmation_seeds'].index(seed)
    else:j=int(seed)%5
    return C['visible_fractions'][j%3],C['heterogeneous_fractions'][(2*j+1)%3],C['fault_rotation'][j]

def setup(domain,n,top,rho,seed):
    vf,hf,fault=condition(seed); rng=np.random.default_rng(seed+97*n+1009*C['domains'].index(domain)+13*C['topologies'].index(top))
    A,H,gamma=graph(n,top,vf); het=np.zeros(n,bool);het[rng.choice(n,max(1,math.ceil(hf*n)),False)]=1
    noise=rng.normal(size=(K,n,3)); bias=np.where(het,rng.choice([-1.,1.],n)*rng.uniform(.10,.32,n),0.)
    q=rho*(.72 if fault=='delay' else 1)*(.78 if fault=='packet_loss' else 1)
    T=expm(-q*H*DT);return vf,hf,fault,rng,A,gamma,het,noise,bias,T

def edges(A,residual,policy):
    if policy in ('all_coupled','physical_filter','robust_local_feedback'):return A
    if policy=='global_gain_reduction':return .55*A
    if policy=='independent_tracking':return np.zeros_like(A)
    W=A*(residual<.30)[None,:]
    for i in range(1,len(A)):W[i,i-1]=max(W[i,i-1],.18)
    return W

def finish(domain,n,top,rho,seed,policy,gamma,vf,hf,fault,err,margin,energy,msg,spreads):
    q=margin[25:]; return dict(domain=domain,n=n,topology=top,rho=rho,gamma=gamma,seed=seed,policy=policy,
      visible_fraction=vf,heterogeneous_fraction=hf,fault=fault,task_success=int(min(q)>=0),minimum_physical_margin=min(q),
      tail_error=float(np.quantile(err[-40:],.95)),first_failure_time=next((i*DT for i,x in enumerate(margin[25:],25) if x<0),K*DT),
      control_energy=energy,communication_messages=msg,parameter_spread=json.dumps(spreads,sort_keys=True))

def simulate(t):
    domain,n,top,rho,seed,policy=t;vf,hf,fault,rng,A,gamma,het,noise,bias,T=setup(domain,n,top,rho,seed)
    z=np.zeros(n); energy=0.;msg=0;err=[];margin=[]
    if domain=='robot':
        mass=np.where(het,rng.uniform(.7,1.5,n),1);drag=np.where(het,rng.uniform(.08,.28,n),.16);auth=np.where(het,rng.uniform(.65,1.0,n),1)
        p=np.c_[-.7*np.arange(n),np.zeros(n)];v=np.zeros((n,2));sp={'mass':float(np.ptp(mass)),'drag':float(np.ptp(drag)),'authority':float(np.ptp(auth))}
    elif domain=='microgrid':
        mass=np.where(het,rng.uniform(2.5,7,n),4.5);drag=np.where(het,rng.uniform(.5,1.7,n),1);auth=np.where(het,rng.uniform(.7,1.15,n),1)
        p=np.zeros(n);v=np.zeros(n);sp={'inertia':float(np.ptp(mass)),'damping':float(np.ptp(drag)),'droop_authority':float(np.ptp(auth))}
    elif domain=='circuit':
        alpha=np.where(het,rng.uniform(.7,1.4,n),1);beta=np.where(het,rng.uniform(.7,1.3,n),1);auth=np.where(het,rng.uniform(.65,1.1,n),1)
        p=np.c_[.1*rng.normal(size=n),np.zeros(n),np.zeros(n)];sp={'alpha':float(np.ptp(alpha)),'beta':float(np.ptp(beta)),'authority':float(np.ptp(auth))}
    elif domain=='water':
        area=np.where(het,rng.uniform(.7,1.5,n),1);out=np.where(het,rng.uniform(.14,.34,n),.22);auth=np.where(het,rng.uniform(.65,1.1,n),1)
        p=np.ones(n);sp={'tank_area':float(np.ptp(area)),'outflow':float(np.ptp(out)),'pump_gain':float(np.ptp(auth))}
    else:
        mass=np.where(het,rng.uniform(.7,1.5,n),1);drag=np.where(het,rng.uniform(.06,.22,n),.12);stiff=np.where(het,rng.uniform(.7,1.5,n),1);auth=np.where(het,rng.uniform(.7,1.05,n),1)
        p=np.zeros(n);v=np.zeros(n);sp={'mass':float(np.ptp(mass)),'damping':float(np.ptp(drag)),'stiffness':float(np.ptp(stiff))}
    for k in range(K):
        target=.55*math.sin(.18*k*DT);z=target+T@(z-target)
        state=p[:,0] if getattr(p,'ndim',1)>1 else p;res=np.abs(bias)+.06*np.abs(state-z);W=edges(A,res,policy);msg+=np.count_nonzero(W)
        coup=W@(state+bias)-W.sum(1)*state
        if policy in ('all_coupled','physical_filter','robust_local_feedback'):exposure=bias
        elif policy=='global_gain_reduction':exposure=.55*bias
        elif policy=='independent_tracking':exposure=np.zeros_like(bias)
        else:exposure=bias*(res<.30)
        if fault in ('sensor_bias','mixed'): exposure*=1.35
        loss=.65 if fault in ('actuator_loss','mixed') else 1
        if domain=='robot':
            ref=np.c_[z-.7*np.arange(n),.35*np.sin(.11*k*DT+np.arange(n)/n)];u=1.4*(ref-p)-1.0*v+.30*rho*coup[:,None]+1.6*rho*rho*exposure[:,None]+.02*noise[k,:,:2]
            if policy=='robust_local_feedback':u+=.45*(ref-p)-.35*v
            if policy=='physical_filter':
                for i in range(n):
                    q=p[i]-p;ds=np.linalg.norm(q,axis=1);mask=(ds>0)&(ds<.42)
                    if mask.any():u[i]+=.18*np.sum(q[mask]/(ds[mask,None]**2+.02),axis=0)
            u=np.clip(u,-2.6*auth[:,None]*loss,2.6*auth[:,None]*loss);v+=DT*(u/mass[:,None]-drag[:,None]*v);p+=DT*v
            e=np.max(np.linalg.norm(p-ref,axis=1));sep=np.min(np.linalg.norm(p[:,None]-p[None,:]+np.eye(n)[:,:,None]*1e5,axis=2));m=min(.75-e,sep-.28)
        elif domain=='microgrid':
            u=-1.5*v-.8*p+.35*(z-v)+.24*rho*coup+2.2*rho*rho*exposure+.02*noise[k,:,0]
            if policy=='robust_local_feedback':u-=.45*v+.2*p
            if policy=='physical_filter':u-=.7*np.sign(v)*np.maximum(np.abs(v)-.45,0)
            u=np.clip(u,-2.5*auth*loss,2.5*auth*loss)
            flow=(A@(p)-A.sum(1)*p);v+=DT*(u-drag*v-1.1*p+.38*rho*flow)/mass;p+=DT*v;e=np.max(np.abs(v));m=min(.65-e,.85-np.max(np.abs(p)))
        elif domain=='circuit':
            x,y,w=p.T;u=1.1*(z-x)+.25*rho*coup+1.8*rho*rho*exposure+.02*noise[k,:,0]
            if policy=='robust_local_feedback':u-=.35*x+.2*y
            if policy=='physical_filter':u-=.5*np.sign(x)*np.maximum(np.abs(x)-1.8,0)
            u=np.clip(u,-2.4*auth*loss,2.4*auth*loss)
            dx=alpha*(y-x**3/3+x)+u;dy=(x-y+w)/beta;dw=-.35*y-.18*w;p+=DT*np.c_[dx,dy,dw];e=np.max(np.abs(x-z));m=min(1.35-e,3.2-np.max(np.abs(p)))
        elif domain=='water':
            u=.75*(1.4+z-p)+.20*rho*coup+1.7*rho*rho*exposure+.015*noise[k,:,0]
            if policy=='robust_local_feedback':u+=.4*(1.4+z-p)
            if policy=='physical_filter':u-=.8*np.maximum(p-2.1,0)
            u=np.clip(u,0,2.4*auth*loss);p+=DT*(u-out*np.sqrt(np.maximum(p,0)))/area;p=np.maximum(p,0)
            ref=1.4+target;e=np.max(np.abs(p-ref));m=min(.55-e,np.min(p)-.25,2.4-np.max(p))
        else:
            u=-1.4*p-.8*v+.30*rho*coup+2.0*rho*rho*exposure+.02*noise[k,:,0]
            if policy=='robust_local_feedback':u-=.45*p+.35*v
            if policy=='physical_filter':u-=.6*np.sign(p)*np.maximum(np.abs(p)-.4,0)
            u=np.clip(u,-2.6*auth*loss,2.6*auth*loss)
            v+=DT*(u-drag*v-stiff*p)/mass;p+=DT*v;e=np.max(np.abs(p));m=min(.55-e,1.3-np.max(np.abs(v)))
        energy+=float(np.sum(np.asarray(u)**2)*DT);err.append(float(e));margin.append(float(m))
    return finish(domain,n,top,rho,seed,policy,gamma,vf,hf,fault,err,margin,energy,int(msg),sp)

def main():
    phase=sys.argv[1] if len(sys.argv)>1 else 'development';seeds=C[phase+'_seeds']
    tasks=[(d,n,k,r,s,p) for d in C['domains'] for n in C['sizes'] for k in C['topologies'] for r in C['rho_grid'] for s in seeds for p in C['policies']]
    # Threads preserve deterministic rows in restricted runtimes that disallow
    # POSIX process semaphores; each task owns its RNG and mutable state.
    with ThreadPoolExecutor(max_workers=6) as ex:rows=list(ex.map(simulate,tasks,chunksize=30))
    path=O/f'v13_{phase}_runs.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
    print(json.dumps({'phase':phase,'rows':len(rows),'path':str(path)}))
if __name__=='__main__':main()
