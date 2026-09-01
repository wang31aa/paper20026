#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=HERE/'results'; CP=HERE/'MARINE_V59_FROZEN_CONTRACT.json'; C=json.loads(CP.read_text())
sys.path.insert(0,str(ROOT/'marine_v58_unseen_domain')); import run_v58 as B

def target_state(t): return np.array([.42*t,2.2*np.sin(.10*t)]),np.array([.42,.22*np.cos(.10*t)])
def simulate(seed,rho,policy):
 rng=np.random.default_rng(seed); eta=np.zeros((B.N,3));eta[:,:2]=B.OFF+rng.normal(0,.28,(B.N,2));nu=rng.normal(0,.03,(B.N,3));tau=np.zeros((B.N,3))
 zp=np.zeros((B.N,2));zv=np.zeros((B.N,2));hp=[];hv=[];delay=round(B.C['information']['message_delay_s']/B.DT);minc=1e9;energy=messages=obs=sat=0
 for k in range(B.STEPS):
  t=k*B.DT;tp,tv=target_state(t);zp[0]=tp;zv[0]=tv;hp.append(zp.copy());hv.append(zv.copy());op=hp[max(0,len(hp)-1-delay)];ov=hv[max(0,len(hv)-1-delay)]
  for i in range(1,B.N):
   zp[i]+=B.DT*(zv[i]+C['observer_gain_per_s']*rho*(op[i-1]-zp[i]));zv[i]+=B.DT*C['observer_gain_per_s']*rho*(ov[i-1]-zv[i]);messages+=1
  dp=np.tile(tp,(B.N,1)) if policy=='independent_tracking' else zp.copy();dv=np.tile(tv,(B.N,1)) if policy=='independent_tracking' else zv.copy();force=np.zeros((B.N,3))
  for i in range(B.N):
   ep=dp[i]+B.OFF[i]-eta[i,:2];acc=.75*ep+.9*(dv[i]-nu[i,:2])
   if policy!='independent_tracking':
    for j in range(B.N):
     if i==j:continue
     dist=np.linalg.norm(eta[i,:2]-eta[j,:2])
     if dist<7.5:
      w=rho
      if policy=='two_layer_physical_filter' and dist<2.6:w=.25*rho
      acc+=w*(.22*((eta[j,:2]-B.OFF[j]+B.OFF[i])-eta[i,:2])+.16*(nu[j,:2]-nu[i,:2]))
   an=np.linalg.norm(acc)
   if an>1.35:acc*=1.35/an;sat+=1
   hd=np.arctan2(acc[1],acc[0]) if np.linalg.norm(acc)>.03 else eta[i,2];force[i,0]=B.MASS[i]*np.cos(B.wrap(hd-eta[i,2]))*np.linalg.norm(acc);force[i,2]=B.IZ[i]*(1.6*B.wrap(hd-eta[i,2])-.8*nu[i,2])
  tau+=B.DT*(force-tau)/B.LAG[:,None];disturb=rng.normal(0,[.45,.55,.08],(B.N,3));disturb[:,:2]+=B.BIAS*B.MASS[:,None]*.20
  for i in range(B.N):
   nu[i]+=B.DT*(tau[i]-B.D[i]*nu[i]+disturb[i])/np.array([B.MASS[i],B.MASS[i],B.IZ[i]]);co,si=np.cos(eta[i,2]),np.sin(eta[i,2]);eta[i,:2]+=B.DT*(np.array([[co,-si],[si,co]])@nu[i,:2]);eta[i,2]=B.wrap(eta[i,2]+B.DT*nu[i,2])
  for i in range(B.N):
   for j in range(i):minc=min(minc,float(np.linalg.norm(eta[i,:2]-eta[j,:2])))
  energy+=float(np.sum(tau*tau)*B.DT);obs+=float(np.sqrt(np.mean((zp-tp)**2)))
 tp,_=target_state(B.C['horizon_s']);rmse=float(np.sqrt(np.mean((eta[:,:2]-(tp+B.OFF))**2)));success=int(minc>=B.C['task']['minimum_clearance_m'] and rmse<=B.C['task']['terminal_formation_rmse_m'])
 return dict(seed=seed,rho=rho,policy=policy,success=success,terminal_rmse_m=rmse,minimum_clearance_m=minc,energy_N2s=energy,messages=messages,observer_rmse_m=obs/B.STEPS,saturation_events=sat)
def main():
 OUT.mkdir(exist_ok=True);rows=[simulate(s,r,p) for s in C['heldout_seeds'] for r in C['rho_grid'] for p in C['policies']]
 with (OUT/'V59_HELDOUT.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 safe=set(C['certified_safe_rho']);q=[x for x in rows if x['policy']=='all_coupled'];fs=sum(x['rho'] in safe and not x['success'] for x in q);fu=sum(x['rho'] not in safe and x['success'] for x in q)
 report={'contract_sha256':hashlib.sha256(CP.read_bytes()).hexdigest(),'runs':len(rows),'false_safe':fs,'false_unsafe':fu,'promotion_pass':fs==0,'summary':{p:{'success_rate':float(np.mean([x['success'] for x in rows if x['policy']==p])),'median_rmse_m':float(np.median([x['terminal_rmse_m'] for x in rows if x['policy']==p])),'median_clearance_m':float(np.median([x['minimum_clearance_m'] for x in rows if x['policy']==p]))} for p in C['policies']},'claim_boundary':C['claim_boundary']}
 (OUT/'V59_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
