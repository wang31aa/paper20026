#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=HERE/"results"
C=json.loads((HERE/"UAV_V57_FROZEN_CONTRACT.json").read_text())
BASE=json.loads((ROOT/"uav_v54_sixdof/UAV_V54_FROZEN_CONTRACT.json").read_text())
N=BASE["agents"]; DT=BASE["sample_time_s"]; G=9.81
M=np.array(BASE["masses_kg"]); DR=np.array(BASE["drag_per_s"]); TAU=np.array(BASE["thrust_lags_s"]); I=np.array(BASE["inertia_kgm2"])
OFF=np.array([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]],float)

def b3(e):
    ph,th,ps=e; cph,sph=np.cos(ph),np.sin(ph);cth,sth=np.cos(th),np.sin(th);cps,sps=np.cos(ps),np.sin(ps)
    return np.array([cps*sth*cph+sps*sph,sps*sth*cph-cps*sph,cth*cph])

def simulate(seed,rho,policy):
    rng=np.random.default_rng(seed);steps=int(BASE["horizon_s"]/DT)
    p=OFF+rng.normal(0,.08,(N,3));v=rng.normal(0,.03,(N,3));ang=rng.normal(0,.015,(N,3));om=np.zeros((N,3));thr=M*G
    z=np.zeros((N,3));hist=[];delay=int(round(BASE["communication_delay_s"]/DT));energy=comm=0.;minclr=1e9;obs=0.;sat=0;filter_cost=0.
    for k in range(steps):
        t=k*DT;target=np.array([.55*t,1.2*np.sin(.22*t),1.5+.25*np.sin(.31*t)])
        z[0]=target;hist.append(z.copy());old=hist[max(0,len(hist)-1-delay)]
        adj=np.zeros((N,N))
        for i in range(1,N):
            adj[i,i-1]=1
            if i>1: adj[i,0]=1
        for i in range(1,N):
            js=np.flatnonzero(adj[i])
            if len(js): z[i]+=DT*BASE["observer_gain"]*np.mean(old[js]-z[i],axis=0);comm+=len(js)
        req=np.zeros((N,3));tor=np.zeros((N,3))
        for i in range(N):
            a=1.25*(z[i]+OFF[i]-p[i])-.9*v[i]
            for j in np.flatnonzero(adj[i]): a+=rho*((p[j]+OFF[i]-OFF[j]-p[i])+.45*(v[j]-v[i]))
            if policy in ("physical_filter","two_layer_physical_filter"):
                rep=np.zeros(3)
                for j in range(N):
                    if i==j: continue
                    dvec=p[i]-p[j];dist=float(np.linalg.norm(dvec))
                    if 1e-9<dist<C["filter"]["activation_distance_m"]:
                        rep+=C["filter"]["gain_m2ps2"]*(1/dist-1/C["filter"]["activation_distance_m"])*dvec/(dist**3)
                norm=float(np.linalg.norm(rep))
                if norm>C["filter"]["acceleration_cap_mps2"]:rep*=C["filter"]["acceleration_cap_mps2"]/norm
                a+=rep;filter_cost+=float(np.dot(rep,rep))*DT
            if policy=="two_layer_physical_filter" and i>1 and np.linalg.norm(p[i]-p[0])>=2.4:
                # Remove only the direct physical root edge; retain the chain as
                # the target-information backbone.
                a-=rho*((p[0]+OFF[i]-OFF[0]-p[i])+.45*(v[0]-v[i]))
            a=np.clip(a,-3,3);desired=np.array([-a[1]/G,a[0]/G,0.]);tor[i]=np.clip(5*(desired-ang[i])-1.4*om[i],-.8,.8)
            req[i]=np.array([0,0,np.clip(M[i]*(G+a[2])/(max(.7,np.cos(ang[i,0])*np.cos(ang[i,1]))),.25*M[i]*G,1.8*M[i]*G)])
        w=rng.normal(0,.035,(N,3));thr+=DT*(req[:,2]-thr)/TAU
        for i in range(N):
            om[i]+=DT*(tor[i]-.08*om[i])/I[i];ang[i]+=DT*om[i]
            acc=thr[i]/M[i]*b3(ang[i])-np.array([0,0,G])-DR[i]*v[i]+w[i]
            v[i]+=DT*acc;p[i]+=DT*v[i]
        for i in range(N):
            for j in range(i):minclr=min(minclr,float(np.linalg.norm(p[i]-p[j])))
        energy+=float(np.sum((thr-M*G)**2)*DT);obs+=float(np.sqrt(np.mean((z-target)**2)));sat+=int(np.any((req[:,2]<=.251*M*G)|(req[:,2]>=1.799*M*G)))
    target=np.array([.55*BASE["horizon_s"],1.2*np.sin(.22*BASE["horizon_s"]),1.5+.25*np.sin(.31*BASE["horizon_s"])])
    rmse=float(np.sqrt(np.mean((p-(target+OFF))**2)));success=int(minclr>=C["task"]["minimum_clearance_m"] and rmse<=C["task"]["completion_radius_m"])
    return {"seed":seed,"rho":rho,"policy":policy,"success":success,"task_rmse_m":rmse,"minimum_clearance_m":minclr,"energy":energy,"filter_cost":filter_cost,"messages":int(comm),"observer_rmse_m":obs/steps,"saturation_steps":sat}

def main():
    OUT.mkdir(exist_ok=True);rows=[simulate(s,r,p) for s in C["heldout_seeds"] for r in C["rho_grid"] for p in C["policies"]]
    with (OUT/"V57_HELDOUT.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={p:{"runs":len([x for x in rows if x["policy"]==p]),"success_rate":float(np.mean([x["success"] for x in rows if x["policy"]==p])),"median_clearance_m":float(np.median([x["minimum_clearance_m"] for x in rows if x["policy"]==p])),"median_rmse_m":float(np.median([x["task_rmse_m"] for x in rows if x["policy"]==p])),"median_energy":float(np.median([x["energy"] for x in rows if x["policy"]==p]))} for p in C["policies"]}
    report={"contract_sha256":hashlib.sha256((HERE/"UAV_V57_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),"runs":len(rows),"summary":summary,"claim_boundary":C["claim_boundary"]}
    (OUT/"V57_REPORT.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
