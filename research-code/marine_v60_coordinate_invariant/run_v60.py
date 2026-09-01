#!/usr/bin/env python3
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent;OUT=H/'results';CP=H/'MARINE_V60_FROZEN_CONTRACT.json';C=json.loads(CP.read_text());sys.path.insert(0,str(ROOT/'marine_v58_unseen_domain'));import run_v58 as B
def tgt(t):return np.array([.42*t,2.2*np.sin(.10*t)]),np.array([.42,.22*np.cos(.10*t)])
def Ri(p):c,s=np.cos(p),np.sin(p);return np.array([[c,-s],[s,c]])
def sim(seed,rho,policy):
 rng=np.random.default_rng(seed);eta=np.zeros((B.N,3));eta[:,:2]=B.OFF+rng.normal(0,.28,(B.N,2));nu=rng.normal(0,.03,(B.N,3));tau=np.zeros((B.N,3));zp=np.zeros((B.N,2));zv=np.zeros((B.N,2));hp=[];hv=[];delay=round(B.C['information']['message_delay_s']/B.DT);mc=1e9;en=msg=ob=0
 for k in range(B.STEPS):
  tp,tv=tgt(k*B.DT);zp[0]=tp;zv[0]=tv;hp.append(zp.copy());hv.append(zv.copy());op=hp[max(0,len(hp)-1-delay)];ov=hv[max(0,len(hv)-1-delay)]
  for i in range(1,B.N):zp[i]+=B.DT*(zv[i]+1.3*rho*(op[i-1]-zp[i]));zv[i]+=B.DT*1.3*rho*(ov[i-1]-zv[i]);msg+=1
  vi=np.array([Ri(eta[i,2])@nu[i,:2] for i in range(B.N)]);dp=np.tile(tp,(B.N,1)) if policy=='independent_tracking' else zp;dv=np.tile(tv,(B.N,1)) if policy=='independent_tracking' else zv;f=np.zeros((B.N,3))
  for i in range(B.N):
   acc=.75*(dp[i]+B.OFF[i]-eta[i,:2])+.9*(dv[i]-vi[i])
   if policy!='independent_tracking':
    for j in range(B.N):
     if i==j:continue
     ds=np.linalg.norm(eta[i,:2]-eta[j,:2])
     if ds<7.5:
      w=rho*(.25 if policy=='two_layer_physical_filter' and ds<2.6 else 1);acc+=w*(.22*((eta[j,:2]-B.OFF[j]+B.OFF[i])-eta[i,:2])+.16*(vi[j]-vi[i]))
   an=np.linalg.norm(acc);acc=acc*min(1,1.35/max(an,1e-12));hd=np.arctan2(acc[1],acc[0]) if an>.03 else eta[i,2];f[i,0]=B.MASS[i]*np.cos(B.wrap(hd-eta[i,2]))*np.linalg.norm(acc);f[i,2]=B.IZ[i]*(1.6*B.wrap(hd-eta[i,2])-.8*nu[i,2])
  tau+=B.DT*(f-tau)/B.LAG[:,None];d=rng.normal(0,[.45,.55,.08],(B.N,3));d[:,:2]+=B.BIAS*B.MASS[:,None]*.20
  for i in range(B.N):nu[i]+=B.DT*(tau[i]-B.D[i]*nu[i]+d[i])/np.array([B.MASS[i],B.MASS[i],B.IZ[i]]);eta[i,:2]+=B.DT*(Ri(eta[i,2])@nu[i,:2]);eta[i,2]=B.wrap(eta[i,2]+B.DT*nu[i,2])
  for i in range(B.N):
   for j in range(i):mc=min(mc,float(np.linalg.norm(eta[i,:2]-eta[j,:2])))
  en+=float(np.sum(tau*tau)*B.DT);ob+=float(np.sqrt(np.mean((zp-tp)**2)))
 tp,_=tgt(B.C['horizon_s']);er=float(np.sqrt(np.mean((eta[:,:2]-(tp+B.OFF))**2)));ok=int(mc>=1.5 and er<=2.0);return dict(seed=seed,rho=rho,policy=policy,success=ok,terminal_rmse_m=er,minimum_clearance_m=mc,energy_N2s=en,messages=msg,observer_rmse_m=ob/B.STEPS)
def main():
 OUT.mkdir(exist_ok=True);rows=[sim(s,r,p) for s in C['heldout_seeds'] for r in C['rho_grid'] for p in C['policies']]
 with (OUT/'V60_HELDOUT.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 safe=set(C['certified_safe_rho']);q=[x for x in rows if x['policy']=='all_coupled'];fs=sum(x['rho'] in safe and not x['success'] for x in q);fu=sum(x['rho'] not in safe and x['success'] for x in q);rep={'contract_sha256':hashlib.sha256(CP.read_bytes()).hexdigest(),'runs':len(rows),'false_safe':fs,'false_unsafe':fu,'promotion_pass':fs==0,'by_rho':{str(r):sum(x['success'] for x in q if x['rho']==r) for r in C['rho_grid']},'summary':{p:{'success_rate':float(np.mean([x['success'] for x in rows if x['policy']==p])),'median_rmse_m':float(np.median([x['terminal_rmse_m'] for x in rows if x['policy']==p])),'median_clearance_m':float(np.median([x['minimum_clearance_m'] for x in rows if x['policy']==p]))}for p in C['policies']},'claim_boundary':C['claim_boundary']};(OUT/'V60_REPORT.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
