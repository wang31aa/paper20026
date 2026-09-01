#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "vehicle_v46"))
from run_v46 import (DT, N, LENGTH, HEADWAY, STANDSTILL, CLEARANCE, RESPONSE,
                     load_source, margin, parameters, simulate)  # noqa: E402

SPEC = json.loads((HERE / "V48_CONTRACT.json").read_text())
OUT = HERE / "results"


def calibration(source):
    split = int(0.6 * len(source["time"]))
    cal = np.column_stack((source["acc_av"][:split], source["acc_sv1"][:split], source["acc_sv2"][:split]))
    disagreement = np.abs(cal - np.median(cal, axis=1, keepdims=True))
    return float(np.quantile(disagreement, .9)), float(np.quantile(disagreement, .6)), max(25.0, .5 * float(np.quantile(source["distance_av_headway"][:split], .1)))


def run_supervisor(source, stress, delay, rho, seed, high, low, initial_gap,
                   warning_mode="joint", warning_threshold=.2):
    split = int(.6 * len(source["time"]))
    target = source["speed_av"][split:split + 200]
    recorded = np.column_stack((source["acc_av"], source["acc_sv1"], source["acc_sv2"]))
    mismatch = recorded - np.median(recorded, axis=1, keepdims=True)
    mismatch = mismatch[split:split + len(target)]
    rng = np.random.default_rng(seed + int(100 * stress) + int(10 * delay))
    lag, drag, accel_max, brake_max = parameters(seed)
    v = np.full(N, target[0]) + rng.normal(0, .25, N); a = np.zeros(N)
    x = -np.arange(N, dtype=float) * initial_gap; z = np.full(N, target[0])
    active = np.ones(N - 1, dtype=bool); delay_steps = int(round(delay / DT))
    zhist = [z.copy() for _ in range(delay_steps + 1)]
    margins=[]; errors=[]; controls=[]; joint=[]; choices={k:0 for k in ("all", "independent", "gate", "barrier")}
    bad=0; failed=False; first_failure=len(target)*DT; first_warning=None; saturated=0; comm=0
    for k, ref in enumerate(target):
        z[0]=ref; delayed=zhist[max(0,len(zhist)-1-delay_steps)]
        for i in range(1,N):
            if rng.random()<rho:
                z[i]+=DT*2.2*rho*(delayed[i-1]-z[i]); comm+=1
        forcing=np.r_[mismatch[k,0],mismatch[k]][:N]
        req=np.zeros(N); req[0]=np.clip((ref-v[0])*1.8,-brake_max[0],accel_max[0])
        for i in range(1,N):
            gap=x[i-1]-x[i]; desired=STANDSTILL+HEADWAY*v[i]
            residual=abs(v[i-1]-v[i])+.35*abs(z[i-1]-z[i])+stress*abs(forcing[i])
            local=margin(gap,v[i],v[i-1],LENGTH[i],LENGTH[i-1])
            observer_error=abs(z[i]-ref)
            physical=.22*(gap-desired)+.72*(v[i-1]-v[i])
            if local<3.0:
                req[i]=min(physical+.65*(z[i]-v[i])+stress*forcing[i],-brake_max[i]*min(1.0,(3.0-local)/3.0)); choices["barrier"]+=1
            elif observer_error>1.2:
                req[i]=.65*(z[i]-v[i])+stress*forcing[i]; choices["independent"]+=1
            elif residual>high:
                active[i-1]=False; req[i]=.65*(z[i]-v[i])+stress*forcing[i]; choices["gate"]+=1
            else:
                active[i-1]=True; req[i]=physical+.65*(z[i]-v[i])+stress*forcing[i]; choices["all"]+=1
        applied=np.clip(req,-brake_max,accel_max); saturated+=int(np.count_nonzero(abs(applied-req)>1e-10))
        a+=DT*((applied-a)/lag-drag*v); v=np.maximum(0,v+DT*a); x+=DT*v; zhist.append(z.copy())
        current=min(margin(x[i-1]-x[i],v[i],v[i-1],LENGTH[i],LENGTH[i-1]) for i in range(1,N))
        obs=float(np.sqrt(np.mean((z[1:]-ref)**2))); headroom=float(np.min((accel_max-np.maximum(applied,0))/np.maximum(accel_max,1e-9)))
        M=min(current/3.0,1.0-obs/1.2,headroom); joint.append(M)
        warning_value = current if warning_mode == "physical" else M
        if first_warning is None and 0 <= warning_value < warning_threshold:
            first_warning=k*DT
        margins.append(current); errors.append(float(np.sqrt(np.mean((v[1:]-ref)**2)))); controls.append(float(np.sum(applied*applied)*DT))
        if current<0: bad+=1
        else:
            if bad>=5 and not failed: failed=True; first_failure=(k-bad+1)*DT
            bad=0
    if bad>=5 and not failed: failed=True; first_failure=(len(target)-bad)*DT
    tail=max(1,len(errors)//5)
    return {"seed":seed,"stress":stress,"delay_s":delay,"rho":rho,"policy":"viability_supervisor","task_success":int(not failed),
            "first_failure_s":first_failure,"first_warning_s":first_warning if first_warning is not None else len(target)*DT,
            "warning_lead_s":max(first_failure-first_warning,0) if failed and first_warning is not None else 0.0,
            "minimum_margin_m":float(min(margins)),"minimum_joint_margin":float(min(joint)),
            "tail_tracking_rmse_mps":float(np.mean(errors[-tail:])),"control_energy":float(sum(controls)),
            "communication_messages":comm,"saturation_fraction":saturated/(N*len(target)),
            **{f"choice_{k}":v for k,v in choices.items()}}


def main():
    source=load_source(); high,low,gap=calibration(source); rows=[]
    for seed in SPEC["seeds"]:
      for stress in SPEC["stress"]:
       for delay in SPEC["delay_s"]:
        for rho in SPEC["participation"]:
         rows.append(run_supervisor(source,stress,delay,rho,seed,high,low,gap))
         for policy in SPEC["comparators"]:
          r=simulate(source,stress,delay,rho,seed,policy,high,low,gap)
          r.update({"first_warning_s":"","warning_lead_s":"","minimum_joint_margin":"","choice_all":"","choice_independent":"","choice_gate":"","choice_barrier":""}); rows.append(r)
    OUT.mkdir(parents=True,exist_ok=True)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/"V48_RESULTS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={}
    for policy in ["viability_supervisor"]+SPEC["comparators"]:
        rr=[r for r in rows if r["policy"]==policy]
        summary[policy]={"runs":len(rr),"success_rate":float(np.mean([r["task_success"] for r in rr])),
                         "median_energy":float(np.median([r["control_energy"] for r in rr])),
                         "median_margin_m":float(np.median([r["minimum_margin_m"] for r in rr]))}
    sup=[r for r in rows if r["policy"]=="viability_supervisor"]
    failed=[r for r in sup if not r["task_success"]]
    report={"contract_sha256":hashlib.sha256((HERE/"V48_CONTRACT.json").read_bytes()).hexdigest(),"rows":len(rows),"paired_conditions":len(sup),"policies":summary,
            "supervisor_failed_runs":len(failed),"warning_detected_before_failure":sum(r["warning_lead_s"]>0 for r in failed),
            "median_warning_lead_s":float(np.median([r["warning_lead_s"] for r in failed])) if failed else None,"claim_boundary":SPEC["claim_boundary"]}
    (OUT/"V48_SUMMARY.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
