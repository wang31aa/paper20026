#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/'results'; CP=HERE/'MARINE_V58_FROZEN_CONTRACT.json'
C=json.loads(CP.read_text()); N=C['agents']; DT=C['sample_time_s']; STEPS=int(C['horizon_s']/DT)
P=C['heterogeneous_parameters']; MASS=np.array(P['mass_kg']); IZ=np.array(P['yaw_inertia_kgm2'])
D=np.column_stack((P['surge_damping_Nspm'],P['sway_damping_Nspm'],P['yaw_damping_NmspRad']))
LAG=np.array(P['actuator_lag_s']); BIAS=np.array(P['current_bias_mps']); S=C['task']['formation_spacing_m']
OFF=np.array([[0,0],[-S,-S],[-S,S],[-2*S,-S],[-2*S,S]],float)

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def simulate(seed,rho,policy):
    rng=np.random.default_rng(seed); eta=np.zeros((N,3)); eta[:,:2]=OFF+rng.normal(0,.28,(N,2)); nu=rng.normal(0,.03,(N,3))
    tau=np.zeros((N,3)); z=np.zeros((N,2)); history=[]; delay=round(C['information']['message_delay_s']/DT)
    min_clear=1e9; energy=messages=observer_sum=saturation=0
    for k in range(STEPS):
        t=k*DT; target=np.array([0.42*t,2.2*np.sin(.10*t)])
        z[0]=target; history.append(z.copy()); old=history[max(0,len(history)-1-delay)]
        for i in range(1,N):
            z[i]+=DT*C['information']['observer_gain_per_s']*rho*(old[i-1]-z[i]); messages+=1
        desired=np.zeros((N,2)); desired[0]=target
        if policy=='independent_tracking': desired[1:]=target
        else: desired[1:]=z[1:]
        force=np.zeros((N,3))
        for i in range(N):
            ep=desired[i]+OFF[i]-eta[i,:2]; ev=-nu[i,:2]
            acc=0.75*ep+0.9*ev
            if policy!='independent_tracking':
                for j in range(N):
                    if i==j: continue
                    # Physical influence is distance limited; the information backbone is retained separately.
                    dist=np.linalg.norm(eta[i,:2]-eta[j,:2])
                    if dist<7.5:
                        w=rho
                        if policy=='two_layer_physical_filter' and dist<2.6: w=0.25*rho
                        acc+=w*(0.22*((eta[j,:2]-OFF[j]+OFF[i])-eta[i,:2])+0.16*(nu[j,:2]-nu[i,:2]))
            speed=np.linalg.norm(acc)
            if speed>1.35: acc*=1.35/speed; saturation+=1
            heading=np.arctan2(acc[1],acc[0]) if np.linalg.norm(acc)>.03 else eta[i,2]
            force[i,0]=MASS[i]*np.cos(wrap(heading-eta[i,2]))*np.linalg.norm(acc)
            force[i,2]=IZ[i]*(1.6*wrap(heading-eta[i,2])-0.8*nu[i,2])
        tau+=DT*(force-tau)/LAG[:,None]
        disturbance=rng.normal(0,[.45,.55,.08],(N,3)); disturbance[:,:2]+=BIAS*MASS[:,None]*.20
        for i in range(N):
            nudot=(tau[i]-D[i]*nu[i]+disturbance[i])/np.array([MASS[i],MASS[i],IZ[i]])
            nu[i]+=DT*nudot; c,s=np.cos(eta[i,2]),np.sin(eta[i,2]); R=np.array([[c,-s],[s,c]])
            eta[i,:2]+=DT*(R@nu[i,:2]); eta[i,2]=wrap(eta[i,2]+DT*nu[i,2])
        for i in range(N):
            for j in range(i): min_clear=min(min_clear,float(np.linalg.norm(eta[i,:2]-eta[j,:2])))
        energy+=float(np.sum(tau*tau)*DT); observer_sum+=float(np.sqrt(np.mean((z-target)**2)))
    target=np.array([0.42*C['horizon_s'],2.2*np.sin(.10*C['horizon_s'])])
    rmse=float(np.sqrt(np.mean((eta[:,:2]-(target+OFF))**2)))
    success=int(min_clear>=C['task']['minimum_clearance_m'] and rmse<=C['task']['terminal_formation_rmse_m'])
    return dict(seed=seed,rho=rho,policy=policy,success=success,terminal_rmse_m=rmse,minimum_clearance_m=min_clear,
                energy_N2s=energy,messages=messages,observer_rmse_m=observer_sum/STEPS,saturation_events=saturation)

def main():
    OUT.mkdir(exist_ok=True); rows=[simulate(s,r,p) for s in C['heldout_seeds'] for r in C['rho_grid'] for p in C['policies']]
    with (OUT/'V58_HELDOUT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    safe=set(C['frozen_theory_prediction']['certified_safe_rho']); q=[x for x in rows if x['policy']=='all_coupled']
    fs=sum(x['rho'] in safe and not x['success'] for x in q); fu=sum(x['rho'] not in safe and x['success'] for x in q)
    report={'contract_sha256':hashlib.sha256(CP.read_bytes()).hexdigest(),'runs':len(rows),'all_coupled_false_safe':fs,
            'all_coupled_false_unsafe':fu,'promotion_pass':fs==0,
            'summary':{p:{'runs':len([x for x in rows if x['policy']==p]),'success_rate':float(np.mean([x['success'] for x in rows if x['policy']==p])),
                          'median_rmse_m':float(np.median([x['terminal_rmse_m'] for x in rows if x['policy']==p])),
                          'median_clearance_m':float(np.median([x['minimum_clearance_m'] for x in rows if x['policy']==p]))} for p in C['policies']},
            'claim_boundary':C['claim_boundary']}
    (OUT/'V58_REPORT.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
