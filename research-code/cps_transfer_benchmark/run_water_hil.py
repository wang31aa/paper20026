#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

RIDGE = 1e-5
SEED = 20260729
TRAIN = "monitor_data_randomized_setpoints.csv"
CAL = "monitor_data_normal_march21st.csv"
TESTS = {
    "monitor_data_levelmeter.csv": "level_sensor_spoof",
    "monitor_data_flowmeter.csv": "flow_sensor_spoof",
    "monitor_data_fillvalve_march21st.csv": "fill_valve_spoof",
    "monitor_data_Display.csv": "display_spoof",
}


def read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1, 2], low_memory=False)
    df["Timestamp"] = pd.to_datetime(df.Timestamp, errors="coerce")
    for c in df.columns:
        if c != "Timestamp":
            if df[c].dtype == object:
                df[c] = df[c].map({"TRUE": 1.0, "FALSE": 0.0}).fillna(pd.to_numeric(df[c], errors="coerce"))
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Timestamp", "I_Level_Meter", "I_Flow_Meter"])


def design(df: pd.DataFrame):
    y = df[["I_Level_Meter", "I_Flow_Meter"]].to_numpy(float)
    cols = ["Q_Fill_Valve", "Q_Discharge_Valve", "LowSetpoint", "HighSetpoint",
            "I_ModeSelector", "Q_Fill_Light", "Q_Discharge_Light"]
    u = df[cols].fillna(0).to_numpy(float)
    dt = df.Timestamp.diff().dt.total_seconds().to_numpy(dtype=float, copy=True)
    dt[~np.isfinite(dt)] = np.nanmedian(dt[np.isfinite(dt)])
    x = np.column_stack([np.ones(len(df)-1), y[:-1], u[:-1], dt[1:]])
    return x, y[1:]


def fit(x, y):
    mu, sd = x[:, 1:].mean(0), x[:, 1:].std(0)
    sd[sd == 0] = 1
    z = np.column_stack([x[:, 0], (x[:, 1:] - mu) / sd])
    p = np.eye(z.shape[1]) * RIDGE; p[0, 0] = 0
    beta = np.linalg.solve(z.T @ z + p, z.T @ y)
    return beta, mu, sd


def predict(x, beta, mu, sd):
    z = np.column_stack([x[:, 0], (x[:, 1:] - mu) / sd])
    return z @ beta


def scale_resid(y, pred, scale):
    return np.sqrt(np.sum(((y-pred)/scale)**2, axis=1))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--raw",type=Path,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    train=read(args.raw/TRAIN); xt,yt=design(train); beta,mu,sd=fit(xt,yt)
    train_pred=predict(xt,beta,mu,sd); q25,q75=np.quantile(yt-train_pred,[.25,.75],axis=0)
    scale=np.maximum((q75-q25)/1.349,1e-12)
    cal=read(args.raw/CAL); xc,yc=design(cal); cal_score=scale_resid(yc,predict(xc,beta,mu,sd),scale)
    threshold=float(np.quantile(cal_score,.99,method="higher")); cal_median=float(np.median(cal_score))
    rng=np.random.default_rng(SEED); rows=[]; missing=[]
    for filename,label in TESTS.items():
        path=args.raw/filename
        if not path.exists(): missing.append(filename); continue
        df=read(path); x,y=design(df); score=scale_resid(y,predict(x,beta,mu,sd),scale)
        n=min(50000,len(score),len(cal_score)); a=rng.choice(score,n,replace=False); b=rng.choice(cal_score,n,replace=False)
        # Common-language probability P(test residual > normal residual), paired Monte Carlo.
        cl=float(np.mean(a>b))
        rows.append({"file":filename,"condition":label,"n":len(score),"median_residual":float(np.median(score)),
                     "median_ratio_to_normal":float(np.median(score)/cal_median),
                     "threshold_exceedance_rate":float(np.mean(score>threshold)),
                     "common_language_probability":cl})
    out=pd.DataFrame(rows); out.to_csv(args.out/"water_hil_session_metrics.csv",index=False)
    summary={"source":{"doi":"10.6084/m9.figshare.28735547.v2","license":"CC-BY-4.0"},
             "split":{"train":TRAIN,"calibration":CAL,"test":[x for x in TESTS if x not in missing],"missing":missing},
             "model":{"type":"two-output causal ridge","ridge":RIDGE,"training_rows":len(train),
                      "calibration_rows":len(cal),"output_scales":scale.tolist(),"threshold_99":threshold},
             "normal":{"median_residual":cal_median,"threshold_exceedance_rate":float(np.mean(cal_score>threshold))},
             "sessions":rows,"claim_boundary":"condition-level offline HIL replay; no pointwise attack timing or O1-O3/C1 deployment"}
    (args.out/"water_hil_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.out.glob("water_hil*")) if p.name!="water_hil_SHA256SUMS.json"}
    (args.out/"water_hil_SHA256SUMS.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(json.dumps(summary["sessions"],indent=2))

if __name__=="__main__": main()
