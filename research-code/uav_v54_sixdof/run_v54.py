#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/"results";C=json.loads((H/"UAV_V54_FROZEN_CONTRACT.json").read_text())
N=C["agents"];DT=C["sample_time_s"];G=9.81
M=np.array(C["masses_kg"]);DR=np.array(C["drag_per_s"]);TAU=np.array(C["thrust_lags_s"]);I=np.array(C["inertia_kgm2"])
OFF=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],float)

def b3(e):
    ph,th,ps=e; cph,sph=np.cos(ph),np.sin(ph);cth,sth=np.cos(th),np.sin(th);cps,sps=np.cos(ps),np.sin(ps)
    return np.array([cps*sth*cph+sps*sph,sps*sth*cph-cps*sph,cth*cph])

def simulate(seed,rho,policy):
    rng=np.random.default_rng(seed);steps=int(C["horizon_s"]/DT)
    p=OFF+rng.normal(0,.08,(N,3));v=rng.normal(0,.03,(N,3));ang=rng.normal(0,.015,(N,3));om=np.zeros((N,3));thr=M*G
    z=np.zeros((N,3));hist=[];delay=int(round(C["communication_delay_s"]/DT));energy=comm=0.;minclr=1e9;obs=0.;sat=0
    for k in range(steps):
        t=k*DT;target=np.array([.55*t,1.2*np.sin(.22*t),1.5+.25*np.sin(.31*t)])
        z[0]=target;hist.append(z.copy());old=hist[max(0,len(hist)-1-delay)]
        adj=np.zeros((N,N));
        for i in range(1,N):
            adj[i,i-1]=1
            if policy=="all_coupled" and i>1: adj[i,0]=1
            if policy=="two_layer_supervisor" and i>1 and np.linalg.norm(p[i]-p[0])<2.4: adj[i,0]=1
        for i in range(1,N):
            js=np.flatnonzero(adj[i]);
            if len(js): z[i]+=DT*C["observer_gain"]*np.mean(old[js]-z[i],axis=0);comm+=len(js)
        req=np.zeros((N,3));tor=np.zeros((N,3))
        for i in range(N):
            kp=1.25;kd=.9;couple=0 if policy=="independent_tracking" else rho*(.55 if policy=="global_gain_reduction" else 1.)
            a=kp*(z[i]+OFF[i]-p[i])-kd*v[i]
            for j in np.flatnonzero(adj[i]): a+=couple*((p[j]+OFF[i]-OFF[j]-p[i])+.45*(v[j]-v[i]))
            a=np.clip(a,-3,3);desired=np.array([-a[1]/G,a[0]/G,0.]);tor[i]=np.clip(5*(desired-ang[i])-1.4*om[i],-.8,.8)
            req[i]=np.array([0,0,np.clip(M[i]*(G+a[2])/(max(.7,np.cos(ang[i,0])*np.cos(ang[i,1]))),.25*M[i]*G,1.8*M[i]*G)])
        w=rng.normal(0,.035,(N,3));thr+=DT*(req[:,2]-thr)/TAU
        for i in range(N):
            om[i]+=DT*(tor[i]-.08*om[i])/I[i];ang[i]+=DT*om[i]
            acc=thr[i]/M[i]*b3(ang[i])-np.array([0,0,G])-DR[i]*v[i]+w[i]
            v[i]+=DT*acc;p[i]+=DT*v[i]
        for i in range(N):
            for j in range(i): minclr=min(minclr,float(np.linalg.norm(p[i]-p[j])))
        energy+=float(np.sum((thr-M*G)**2)*DT);obs+=float(np.sqrt(np.mean((z-target)**2)));sat+=int(np.any((req[:,2]<=.251*M*G)|(req[:,2]>=1.799*M*G)))
    target=np.array([.55*C["horizon_s"],1.2*np.sin(.22*C["horizon_s"]),1.5+.25*np.sin(.31*C["horizon_s"])])
    rmse=float(np.sqrt(np.mean((p-(target+OFF))**2)));success=int(minclr>=C["minimum_clearance_m"] and rmse<=C["completion_radius_m"])
    return {"seed":seed,"rho":rho,"policy":policy,"success":success,"task_rmse_m":rmse,"minimum_clearance_m":minclr,"energy":energy,"messages":int(comm),"observer_rmse_m":obs/steps,"saturation_steps":sat}

def main():
    O.mkdir(exist_ok=True);rows=[simulate(s,r,p) for s in C["heldout_seeds"] for r in C["rho_grid"] for p in C["policies"]]
    summary={p:{"success_rate":float(np.mean([x["success"] for x in rows if x["policy"]==p])),"median_rmse_m":float(np.median([x["task_rmse_m"] for x in rows if x["policy"]==p])),"median_energy":float(np.median([x["energy"] for x in rows if x["policy"]==p])),"median_messages":float(np.median([x["messages"] for x in rows if x["policy"]==p]))} for p in C["policies"]}
    import csv
    with (O/"V54_HELDOUT.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out={"contract_sha256":hashlib.sha256((H/"UAV_V54_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"runs":len(rows),"summary":summary,"heterogeneity":{"mass_cv":float(np.std(M)/np.mean(M)),"drag_cv":float(np.std(DR)/np.mean(DR)),"lag_cv":float(np.std(TAU)/np.mean(TAU))},"claim_boundary":C["claim_boundary"]}
    (O/"V54_REPORT.json").write_text(json.dumps(out,indent=2)+"\n");print(json.dumps(out,indent=2))
if __name__=="__main__":main()
