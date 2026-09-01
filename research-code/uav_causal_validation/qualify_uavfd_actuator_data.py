#!/usr/bin/env python3
"""Field qualification and exploratory motor-response fits for UAV-FD."""
from __future__ import annotations
import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np
from scipy.io import loadmat

EXPECTED={"no_fault":"065e36e917d0e6188a3b766e17a720c7",
          "fault_m4_10":"76cde4e92881f3ec0f02d2af40a516c9"}

def md5(path): return hashlib.md5(path.read_bytes()).hexdigest()

def fit_record(path: Path, record: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); d=loadmat(path,squeeze_me=True,struct_as_record=False)
    required=['RCOU','RCOU_label','ESC_label',*[f'ESC_{i}' for i in range(6)]]
    missing=[k for k in required if k not in d]
    if missing: raise ValueError(f'{record} missing {missing}')
    rc=np.asarray(d['RCOU'],float); command_time=rc[:,1]*1e-6
    rows=[]
    for motor in range(6):
        esc=np.asarray(d[f'ESC_{motor}'],float); time=esc[:,1]*1e-6
        rpm=esc[:,3]; pwm=np.interp(time,command_time,rc[:,10+motor])
        active=(rpm>500)&(pwm>1050)&np.isfinite(rpm)&np.isfinite(pwm)
        valid=active[:-1]&active[1:]
        X=np.c_[rpm[:-1],pwm[:-1]-1000,np.ones(len(rpm)-1)][valid]
        y=rpm[1:][valid]
        split=int(.6*len(y))
        if split<30 or len(y)-split<20: raise ValueError(f'{record} motor {motor} insufficient excitation')
        coefficient=np.linalg.lstsq(X[:split],y[:split],rcond=None)[0]
        prediction=X[split:]@coefficient; rmse=float(np.sqrt(np.mean((prediction-y[split:])**2)))
        phi=float(coefficient[0]); dt=float(np.median(np.diff(time[active])))
        tau=float(-dt/np.log(phi)) if 0<phi<1 else None
        gain=float(coefficient[1]/(1-phi)) if phi<1 else None
        rows.append({"record":record,"motor":motor+1,"samples":len(y),"phi":phi,
                     "time_constant_s":tau,"steady_rpm_per_pwm":gain,
                     "heldout_rmse_rpm":rmse})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('no_fault',type=Path);p.add_argument('fault',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    hashes={"no_fault":md5(a.no_fault),"fault_m4_10":md5(a.fault)}
    if hashes!=EXPECTED: raise SystemExit(f'hash mismatch: {hashes}')
    rows=fit_record(a.no_fault,'NO_FAULT1')+fit_record(a.fault,'FAULT_M4_10')
    tau=[r['time_constant_s'] for r in rows if r['time_constant_s'] is not None]
    gain=[r['steady_rpm_per_pwm'] for r in rows if r['steady_rpm_per_pwm'] is not None]
    result={"dataset":"UAV-FD","doi":"10.5281/zenodo.7648996","license":"CC-BY-4.0",
            "downloaded_file_hashes":hashes,"field_qualification":"PASS",
            "fits":rows,"observed_exploratory_envelope":{"time_constant_s":[min(tau),max(tau)],
                    "steady_rpm_per_pwm":[min(gain),max(gain)]},
            "eligible_to_freeze_uav_plant":False,
            "blocking_reasons":["only one no-fault and one fault flight were downloaded",
                "within-hexarotor motor response is not identical to between-UAV translational plant response",
                "flight condition and feedback compensation confound a direct fault causal contrast"],
            "permitted_use":"physical actuator range qualification and sensitivity-design input only"}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fits'},indent=2))

if __name__=='__main__':main()
