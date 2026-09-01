#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "V9_FROZEN_CONTRACT.json").read_text())
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)


def graph(n, kind, rng):
    a=np.zeros((n,n))
    for i in range(1,n): a[i,i-1]=1
    if kind=="ring": a[0,n-1]=1
    elif kind=="clustered":
        block=max(2,n//4)
        for i in range(n):
            for j in range(n):
                if i!=j and i//block==j//block: a[i,j]=1/max(block-1,1)
        for i in range(block,n,block): a[i,i-1]=1
    elif kind=="random_directed":
        for i in range(n):
            for j in range(n):
                if i!=j and rng.random()<min(0.18,3/n): a[i,j]=1
        for i in range(1,n): a[i,i-1]=1
    row=a.sum(1); a[row>0]/=row[row>0,None]
    return a


def observer_step(z, target, a, visible, rho, dt):
    delayed=z.copy(); consensus=(a@delayed)-z
    pin=visible*(target-z)
    return z+dt*rho*(1.4*consensus+2.2*pin)


def common_setup(n,kind,seed):
    rng=np.random.default_rng(seed); a=graph(n,kind,rng)
    visible=np.zeros(n); visible[:max(1,int(np.ceil(.2*n)))]=1
    hetero=np.zeros(n,dtype=bool); hetero[rng.choice(n,max(1,int(np.ceil(.25*n))),replace=False)]=1
    signs=rng.choice([-1.,1.],n); bias=.32*signs*hetero
    noise=rng.normal(0,.025,(400,n))
    return rng,a,visible,hetero,bias,noise


def physical_weights(a,bias,policy):
    if policy=="independent_tracking": return np.zeros_like(a)
    if policy=="two_layer_gate":
        trusted=(np.abs(bias)<1e-12).astype(float)
        return a*trusted[None,:]
    return a


def simulate_robot(n,kind,rho,seed,policy):
    dt=.05; steps=400; rng,a,vis,het,bias,noise=common_setup(n,kind,seed)
    eff=np.where(het,rng.uniform(.72,1.18,n),1.0); x=-np.arange(n,dtype=float); v=np.zeros(n); z=np.zeros(n)
    w=physical_weights(a,bias,policy); errors=[]; margins=[]; energy=0.; messages=0
    for k in range(steps):
        target=.7*np.sin(.22*k*dt); z=observer_step(z,target,a,vis,rho,dt)
        desired=z-np.arange(n); coupling=(w@(x+bias+np.arange(n)))-w.sum(1)*(x+np.arange(n))
        u=1.4*(desired-x)-1.1*v+.65*rho*coupling+noise[k]; u=np.clip(u,-2.5,2.5)
        v+=dt*eff*u; x+=dt*v; energy+=float(np.sum(u*u)*dt); messages+=int(np.count_nonzero(a))
        errors.append(float(np.max(np.abs(x-desired)))); margins.append(float(np.min(np.abs(np.diff(np.sort(x))))))
    tail=float(np.quantile(errors[-100:],.95)); margin=float(min(margins)); success=tail<=.55 and margin>=.25
    return success,tail,margin,energy,messages


def simulate_vehicle(n,kind,rho,seed,policy):
    dt=.05; steps=400; rng,a,vis,het,bias,noise=common_setup(n,kind,seed)
    tau=np.where(het,rng.uniform(.25,.75,n),.4); x=-18*np.arange(n,dtype=float); v=np.full(n,12.); acc=np.zeros(n); z=np.full(n,12.)
    w=physical_weights(a,bias,policy); margins=[]; errors=[]; energy=0.; messages=0
    for k in range(steps):
        target=12+1.5*np.sin(.16*k*dt); z=observer_step(z,target,a,vis,rho,dt)
        cmd=np.zeros(n); cmd[0]=.8*(z[0]-v[0])
        gaps=x[:-1]-x[1:]; desired=4+1.4*v[1:]
        row_sum=w.sum(1); communicated=w@(v+bias)
        cmd[1:]=.22*(gaps-desired)+.75*(z[1:]-v[1:])+.45*rho*(communicated[1:]-row_sum[1:]*v[1:])
        cmd=np.clip(cmd+noise[k],-3.5,2.0); acc+=dt*(cmd-acc)/tau; v=np.maximum(0,v+dt*acc); x+=dt*v
        bumper=x[:-1]-x[1:]; required=4+1.4*v[1:]+np.maximum((v[1:]**2-v[:-1]**2)/(2*3.5),0)
        margins.append(float(np.min(bumper-required))); errors.append(float(np.max(np.abs(v-z))))
        energy+=float(np.sum(cmd*cmd)*dt); messages+=int(np.count_nonzero(a))
    margin=float(min(margins)); tail=float(np.quantile(errors[-100:],.95)); success=margin>=0 and tail<=1.2
    return success,tail,margin,energy,messages


def simulate_motor(n,kind,rho,seed,policy):
    dt=.05; steps=400; rng,a,vis,het,bias,noise=common_setup(n,kind,seed)
    tau=np.where(het,rng.uniform(.35,.95,n),.55); speed=np.zeros(n); z=np.zeros(n); w=physical_weights(a,bias,policy)
    errors=[]; energy=0.; messages=0
    for k in range(steps):
        target=1.2+.35*np.sin(.25*k*dt); z=observer_step(z,target,a,vis,rho,dt)
        coupling=(w@(speed+bias))-w.sum(1)*speed
        u=1.8*(z-speed)+.7*rho*coupling+noise[k]; u=np.clip(u,-3,3)
        speed+=dt*(-speed+u)/tau; errors.append(float(np.max(np.abs(speed-target))))
        energy+=float(np.sum(u*u)*dt); messages+=int(np.count_nonzero(a))
    tail=float(np.quantile(errors[-100:],.95)); success=tail<=.45
    return success,tail,float('nan'),energy,messages


def execute(task):
    domain,n,topology,rho,seed,policy=task
    success,tail,margin,energy,messages=globals()[f"simulate_{domain}"](n,topology,float(rho),int(seed),policy)
    return dict(domain=domain,n=n,topology=topology,rho=rho,seed=seed,policy=policy,success=int(success),tail_error=tail,minimum_task_margin=margin,control_energy=energy,message_count=messages)


def main():
    tasks=[(domain,n,topology,rho,seed,policy)
           for domain in CONTRACT["domains"] for n in CONTRACT["sizes"]
           for topology in CONTRACT["topologies"] for rho in CONTRACT["rho"]
           for seed in CONTRACT["heldout_seeds"] for policy in CONTRACT["policies"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(execute,tasks,chunksize=30))
    with (OUT/"v9_runs.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=rows[0]); wr.writeheader(); wr.writerows(rows)
    digest=hashlib.sha256((ROOT/"V9_FROZEN_CONTRACT.json").read_bytes()).hexdigest()
    summary={"contract_sha256":digest,"runs":len(rows),"scope":CONTRACT["claim_boundary"]}
    (OUT/"v9_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(summary)

if __name__=="__main__": main()
