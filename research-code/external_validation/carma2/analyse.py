#!/usr/bin/env python3
import csv, hashlib, math, statistics
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "cache" / "CARMA2.csv"
OUT = ROOT / "results" / "vehicle_run_metrics.csv"
COLORS = ("BLACK", "GREEN", "GREY", "SILVER", "WHITE")
ACCEL_OVERRIDE = dict(zip(COLORS,("accelerator_pedal_override_active_BLACK",
    "accelerator_pedal_override_active_GREEN","accelerator_pedal_override_active_GREY",
    "accelerator_pedal_override_active_SILVER","accelerator_pedal_override_active_WHITE")))

def number(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (TypeError, ValueError):
        return None

def quantile(values, q):
    a = sorted(values)
    if not a: return None
    z = (len(a)-1)*q; lo = int(z); hi = min(lo+1,len(a)-1); w=z-lo
    return a[lo]*(1-w)+a[hi]*w

def corr(a,b):
    if len(a)<3: return None
    ma=statistics.fmean(a); mb=statistics.fmean(b)
    xa=[x-ma for x in a]; xb=[x-mb for x in b]
    den=math.sqrt(sum(x*x for x in xa)*sum(x*x for x in xb))
    return sum(x*y for x,y in zip(xa,xb))/den if den else None

def split(run):
    n=int(run)
    return "calibration" if n<=6 else ("validation" if n<=9 else "test")

def main():
    grouped=defaultdict(lambda: {c:{"seq":[],"radar":[]} for c in COLORS})
    with SOURCE.open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            run=row["RunNo"]
            for color in COLORS:
                mode=number(row.get(f"command_mode_cacc_{color}"))
                cmd=number(row.get(f"speed_command_cacc_{color}"))
                vel=number(row.get(f"vehspdavgdrvn_srx_{color}"))
                ov1=number(row.get(f"throttle_ovr_flag_{color}"))
                ov2=number(row.get(ACCEL_OVERRIDE[color]))
                t=number(row.get(f"bin_utc_time_s_{color}"))
                rg=number(row.get(f"flrrtrk1range_srx_{color}"))
                if rg is not None and 0 < rg <= 200: grouped[run][color]["radar"].append(rg)
                if mode==2 and cmd is not None and vel is not None and ov1==0 and ov2==0 and t is not None:
                    grouped[run][color]["seq"].append((t,cmd,vel))
    records=[]
    for run in sorted(grouped,key=int):
        for color in COLORS:
            seq=grouped[run][color]["seq"]; radar=grouped[run][color]["radar"]
            if len(seq)<100: continue
            seq.sort(); times=[x[0] for x in seq]
            diffs=[b-a for a,b in zip(times,times[1:]) if b>a]
            dt=statistics.median(diffs) if diffs else 0.1
            maxstep=max(0,int(round(5.0/dt)))
            cmd=np.asarray([x[1] for x in seq]); vel=np.asarray([x[2] for x in seq])
            cmd_std=float(cmd.std()); excitation=float(np.mean(np.abs(np.diff(cmd))>1e-6))
            best=(None,None)
            for lag in range(maxstep+1) if cmd_std>=1e-3 else ():
                if lag>=len(seq)-2: break
                a=cmd if lag==0 else cmd[:-lag]; b=vel[lag:]
                sa=a.std(); sb=b.std()
                c=float(np.mean((a-a.mean())*(b-b.mean()))/(sa*sb)) if sa and sb else None
                if c is not None and (best[1] is None or c>best[1]): best=(lag,c)
            err=[abs(x[2]-x[1]) for x in seq]
            records.append({
                "run":run,"split":split(run),
                "vehicle":color,"eligible_n":len(seq),"dt_s":dt,
                "median_abs_error":statistics.median(err),"p95_abs_error":quantile(err,.95),
                "rmse":math.sqrt(statistics.fmean([x*x for x in err])),
                "best_lag_s":best[0]*dt if best[0] is not None else "",
                "best_correlation":best[1] if best[1] is not None else "",
                "lag_boundary_flag":int(best[0]==maxstep) if best[0] is not None else "",
                "command_std":cmd_std,"command_change_fraction":excitation,
                "radar_n":len(radar),"radar_median_m":statistics.median(radar) if radar else "",
                "radar_p05_m":quantile(radar,.05) if radar else ""})
    OUT.parent.mkdir(exist_ok=True)
    fields=list(records[0])
    with OUT.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    print("source_sha256",hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    print("vehicle_run_rows",len(records));print("output",OUT)

if __name__ == "__main__": main()
