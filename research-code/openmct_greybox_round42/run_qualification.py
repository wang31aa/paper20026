#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / "PROTOCOL.json").read_text())

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def read_log(path: Path) -> pd.DataFrame:
    d=pd.read_csv(path,skiprows=2)
    needed={"MEAS","PWM","REF","DT_ms"}
    if not needed.issubset(d): raise ValueError(f"missing columns in {path}: {sorted(needed-set(d))}")
    if not np.isfinite(d[list(needed)].to_numpy(float)).all(): raise ValueError(f"nonfinite analysed channel in {path}")
    if (d.DT_ms.to_numpy(float)<=0).any(): raise ValueError(f"nonpositive DT in {path}")
    return d

def paths(raw: Path):
    train=sorted((raw/"03_system_identification").glob("* ms sampling time/raw_data.txt"))
    test=sorted((raw/"04_continuous_PID_validation").glob("*_ms/raw_data_*.txt"))
    test+=sorted((raw/"05_discrete_controller_validation").glob("Case_*/raw_data.txt"))
    if len(train)!=3 or len(test)!=7: raise ValueError(f"expected 3/7 files, found {len(train)}/{len(test)}")
    return train,test

def rows(d: pd.DataFrame, timing: str):
    y=d.MEAS.to_numpy(float); p=d.PWM.to_numpy(float); dt=d.DT_ms.to_numpy(float)/1000
    # k=1,...,n-2 permits both current and one-lag PWM contracts.
    state=y[1:-1]; target=y[2:]; interval=dt[1:-1]
    pwm=p[1:-1] if timing=="pwm_k" else p[:-2]
    return state,pwm,interval,target

def zoh(theta,state,pwm,dt,intercept=True):
    a,b=theta[:2]; c=theta[2] if intercept else 0.0
    A=np.exp(-a*dt); gain=-np.expm1(-a*dt)/a
    return A*state+gain*(b*pwm+c)

def fit(frames,timing,intercept=True):
    rr=[rows(d,timing) for d in frames]
    state=np.concatenate([x[0] for x in rr]); pwm=np.concatenate([x[1] for x in rr])
    dt=np.concatenate([x[2] for x in rr]); target=np.concatenate([x[3] for x in rr])
    scale=max(float(np.std(target)),1e-12)
    x0=np.array([5.0,5.0,0.0]) if intercept else np.array([5.0,5.0])
    lo=np.array(PROTOCOL["model"]["bounds"]["a"][:1]+PROTOCOL["model"]["bounds"]["b"][:1]+([] if not intercept else PROTOCOL["model"]["bounds"]["c"][:1]),float)
    hi=np.array([PROTOCOL["model"]["bounds"]["a"][1],PROTOCOL["model"]["bounds"]["b"][1]]+([] if not intercept else [PROTOCOL["model"]["bounds"]["c"][1]]),float)
    res=least_squares(lambda th:(zoh(th,state,pwm,dt,intercept)-target)/scale,x0,bounds=(lo,hi),method="trf",max_nfev=3000)
    if not res.success: raise RuntimeError(res.message)
    pred=zoh(res.x,state,pwm,dt,intercept); resid=target-pred
    return res,state,pwm,dt,target,resid,scale

def nrmse(y,pred,scale): return float(np.sqrt(np.mean((y-pred)**2))/scale)

def timing_cv(frames,timing):
    vals=[]
    scale=max(float(np.std(np.concatenate([d.MEAS.to_numpy(float) for d in frames]))),1e-12)
    for i,d in enumerate(frames):
        fitset=[x for j,x in enumerate(frames) if j!=i]; res,*_=fit(fitset,timing)
        s,p,dt,y=rows(d,timing); vals.append(nrmse(y,zoh(res.x,s,p,dt),scale))
    return vals

def recursive(d,theta,timing,intercept=True):
    y=d.MEAS.to_numpy(float); p=d.PWM.to_numpy(float); dt=d.DT_ms.to_numpy(float)/1000
    out=np.empty(len(y)-2); current=float(y[1])
    for q,k in enumerate(range(1,len(y)-1)):
        pwm=float(p[k] if timing=="pwm_k" else p[k-1])
        current=float(zoh(theta,np.array([current]),np.array([pwm]),np.array([dt[k]]),intercept)[0]); out[q]=current
    return y[2:],out

def arx_fit(frames):
    def design(d):
        y=d.MEAS.to_numpy(float); r=d.REF.to_numpy(float); p=d.PWM.to_numpy(float); dt=d.DT_ms.to_numpy(float)
        return np.column_stack([y[1:-1],y[:-2],p[1:-1],r[2:],r[1:-1],dt[1:-1]]),y[2:]
    xy=[design(d) for d in frames]; X=np.vstack([x for x,_ in xy]); Y=np.concatenate([y for _,y in xy])
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1; Z=np.column_stack([np.ones(len(X)),(X-mu)/sd])
    P=np.eye(Z.shape[1])*1e-5; P[0,0]=0; beta=np.linalg.solve(Z.T@Z+P,Z.T@Y)
    return mu,sd,beta

def arx_recursive(d,fitv):
    mu,sd,beta=fitv; y=d.MEAS.to_numpy(float); r=d.REF.to_numpy(float); p=d.PWM.to_numpy(float); dt=d.DT_ms.to_numpy(float)
    out=[]; ykm1=float(y[0]); yk=float(y[1])
    for k in range(1,len(y)-1):
        x=np.array([yk,ykm1,p[k],r[k+1],r[k],dt[k]]); nxt=float(np.r_[1,(x-mu)/sd]@beta)
        out.append(nxt); ykm1,yk=yk,nxt
    return y[2:],np.asarray(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--archive",type=Path,required=True); ap.add_argument("--raw",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    archive_hash=sha256(a.archive); train_p,test_p=paths(a.raw); train=[read_log(p) for p in train_p]
    input_hashes={str(p.relative_to(a.raw)):sha256(p) for p in train_p+test_p}
    cv={t:timing_cv(train,t) for t in PROTOCOL["timing_contracts"]}; chosen=min(cv,key=lambda t:(float(np.mean(cv[t])),t))
    res,state,pwm,dt,target,resid,scale=fit(train,chosen); noint,*_=fit(train,chosen,False)
    theta=res.x; n=len(target); k=len(theta); rank=int(np.linalg.matrix_rank(res.jac))
    colnorm=np.linalg.norm(res.jac,axis=0); Jscaled=res.jac/np.where(colnorm>0,colnorm,1); cond=float(np.linalg.cond(Jscaled))
    sigma2=float(np.sum(res.fun**2)/max(n-k,1)); cov=np.linalg.pinv(res.jac.T@res.jac)*sigma2; se=np.sqrt(np.maximum(np.diag(cov),0)); ci=np.column_stack([theta-1.96*se,theta+1.96*se])
    loo=[]
    for i,p in enumerate(train_p):
        rr,*_=fit([x for j,x in enumerate(train) if j!=i],chosen); loo.append({"left_out":str(p.relative_to(a.raw)),"theta":rr.x.tolist()})
    rel=max(float(np.max(np.abs(np.asarray(x["theta"])-theta)/np.maximum(np.abs(theta),1e-9))) for x in loo)
    dom={c:[float(min(d[c].min() for d in train)),float(max(d[c].max() for d in train))] for c in ["MEAS","PWM","DT_ms"]}
    arx=arx_fit(train); ymean=float(np.mean(np.concatenate([d.MEAS for d in train]))); yscale=max(float(np.std(np.concatenate([d.MEAS for d in train]))),1e-12)
    table=[]
    for pth in test_p:
        d=read_log(pth); truth,grey=recursive(d,theta,chosen); _,g0=recursive(d,noint.x,chosen,False); _,arp=arx_recursive(d,arx)
        pers=np.full_like(truth,float(d.MEAS.iloc[1])); mean=np.full_like(truth,ymean)
        exits={c:bool((d[c]<dom[c][0]).any() or (d[c]>dom[c][1]).any()) for c in dom}
        table.append({"file":str(pth.relative_to(a.raw)),"n":len(truth),"grey_recursive_nrmse":nrmse(truth,grey,yscale),"persistence_recursive_nrmse":nrmse(truth,pers,yscale),"mean_nrmse":nrmse(truth,mean,yscale),"arx_recursive_nrmse":nrmse(truth,arp,yscale) if np.isfinite(arp).all() else None,"no_intercept_recursive_nrmse":nrmse(truth,g0,yscale),"grey_finite":bool(np.isfinite(grey).all()),"arx_finite":bool(np.isfinite(arp).all()),**{f"exit_{c}":v for c,v in exits.items()}})
    df=pd.DataFrame(table); df.to_csv(a.out/"holdout_metrics.csv",index=False)
    ident={"rank":rank,"scaled_sensitivity_condition":cond,"theta":{"a":float(theta[0]),"b":float(theta[1]),"c":float(theta[2])},"ci95":{"a":ci[0].tolist(),"b":ci[1].tolist(),"c":ci[2].tolist()},"pwm_std":float(np.std(pwm)),"leave_one_out":loo,"max_relative_parameter_change":rel}
    (a.out/"identifiability.json").write_text(json.dumps(ident,indent=2)+"\n")
    checks={"archive_hash":bool(archive_hash==PROTOCOL["source"]["archive_sha256"]),"three_train_seven_holdout":bool(len(train_p)==3 and len(test_p)==7),"rank_3":bool(rank==3),"condition":bool(cond<=PROTOCOL["identifiability_gate"]["scaled_sensitivity_condition_max"]),"positive_ci_a_b":bool(ci[0,0]>0 and ci[1,0]>0),"loo_stability":bool(rel<=PROTOCOL["identifiability_gate"]["leave_one_aprbs_out_max_relative_parameter_change"]),"pwm_excitation":bool(float(np.std(pwm))>=PROTOCOL["identifiability_gate"]["minimum_pwm_standard_deviation"]),"all_grey_finite":bool(df.grey_finite.all()),"median_beats_persistence":bool(float(df.grey_recursive_nrmse.median())<float(df.persistence_recursive_nrmse.median())),"every_record_beats_persistence":bool((df.grey_recursive_nrmse<df.persistence_recursive_nrmse).all()),"zero_domain_exits":bool(not df[[c for c in df if c.startswith("exit_")]].to_numpy(bool).any())}
    authorized=bool(all(checks.values()))
    result={"protocol_sha256":sha256(HERE/"PROTOCOL.json"),"archive":{"observed_sha256":archive_hash,"expected_sha256":PROTOCOL["source"]["archive_sha256"]},"analysed_input_sha256":input_hashes,"timing":{"cv_nrmse":cv,"selected":chosen},"training":{"rows":n,"domain":dom,"speed_scale":yscale},"identifiability":ident,"checks":checks,"qualification_passed":authorized,"network_simulation_authorized":authorized,"stop_reason":None if authorized else [x for x,v in checks.items() if not v],"claim_boundary":PROTOCOL["claim_boundary"]}
    (a.out/"qualification.json").write_text(json.dumps(result,indent=2)+"\n")
    frozen_names=("holdout_metrics.csv","identifiability.json","qualification.json")
    hashes={name:sha256(a.out/name) for name in frozen_names}; (a.out/"SHA256SUMS.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps({"passed":authorized,"failed":[k for k,v in checks.items() if not v]},indent=2))

if __name__=="__main__": main()
