#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from python_translation.model import exact_step
from python_translation.workspace import load_official_workspace

def minimum_pair(p):
    return float(min(np.linalg.norm(frame[i]-frame[j]) for frame in p
                     for i in range(frame.shape[0]-1) for j in range(i+1,frame.shape[0])))

ap=argparse.ArgumentParser(); ap.add_argument("workspace",type=Path)
ap.add_argument("log",type=Path); ap.add_argument("--output",type=Path,default=HERE/"results/nmpc_source_replay_metrics.json")
a=ap.parse_args(); w=load_official_workspace(a.workspace)
rows=[json.loads(line) for line in a.log.read_text().splitlines() if line.strip()]
n=w.parameters.n; dt=w.parameters.dt
translated_p=[np.asarray(r["state"],float)[:3*n].reshape(n,3) for r in rows]
translated_v=[np.asarray(r["state"],float)[3*n:].reshape(n,3) for r in rows]
translated_u=[np.asarray(r["applied_control"],float).reshape(n,3) for r in rows]
if rows:
    final=exact_step(np.asarray(rows[-1]["state"],float),translated_u[-1],dt,n)
    translated_p.append(final[:3*n].reshape(n,3)); translated_v.append(final[3*n:].reshape(n,3))
translated_p=np.asarray(translated_p); translated_v=np.asarray(translated_v); translated_u=np.asarray(translated_u)
length=min(len(translated_p),len(w.positions)); rp=w.positions[:length].reshape(length,n,3); rv=w.velocities[:length].reshape(length,n,3)
position_rmse=float(np.sqrt(np.mean((translated_p[:length]-rp)**2)))
position_range=float(np.ptp(rp)); normalized=position_rmse/max(position_range,1e-12)
speed_r=np.linalg.norm(rv,axis=2); speed_t=np.linalg.norm(translated_v[:length],axis=2)
pair_r=minimum_pair(rp); pair_t=minimum_pair(translated_p[:length])
energy_t=float(np.sum(translated_u**2)*dt)
if w.controls is not None:
    cu=np.asarray(w.controls,float).reshape(-1,n,3); energy_r=float(np.sum(cu**2)*dt)
else: energy_r=None
direction=np.asarray(w.parameters.reference_direction,float); direction/=np.linalg.norm(direction)
endpoint=float(np.min(w.positions[-1].reshape(n,3)@direction))
completed=bool(len(translated_p) and np.all(translated_p[-1]@direction>=endpoint))
duration_t=float((len(translated_p)-1)*dt); duration_r=float(w.times[-1]-w.times[0])
source_collision=pair_r < 2*w.parameters.collision_radius
translated_collision=pair_t < 2*w.parameters.collision_radius
criteria={
 "position_rmse":position_rmse<=.10,
 "normalized_position_rmse":normalized<=.05,
 "completion_time":abs(duration_t-duration_r)<=max(.5,.05*duration_r),
 "minimum_pair_distance":abs(pair_t-pair_r)<=max(.05,.05*pair_r),
 "mean_speed":abs(float(speed_t.mean()-speed_r.mean()))<=max(.05,.05*float(speed_r.mean())),
 "p95_speed":abs(float(np.percentile(speed_t,95)-np.percentile(speed_r,95)))<=max(.05,.05*float(np.percentile(speed_r,95))),
 "control_integral":energy_r is not None and abs(energy_t-energy_r)<=.1*max(energy_r,1e-12),
 "collision_event":translated_collision==source_collision,
 "completion_classification":completed,
 "all_solver_steps_passed":bool(rows) and all(r["solver_success"] for r in rows),
 "official_solver_tolerances":bool(rows) and all(r["solver_official_tolerance_pass"] for r in rows),
}
report={"workspace":str(a.workspace),"free_running":True,"recorded_intermediate_state_used":False,
 "metrics":{"position_rmse":position_rmse,"normalized_position_rmse":normalized,
 "completion_time_recorded":duration_r,"completion_time_translated":duration_t,
 "minimum_pair_recorded":pair_r,"minimum_pair_translated":pair_t,
 "mean_speed_recorded":float(speed_r.mean()),"mean_speed_translated":float(speed_t.mean()),
 "p95_speed_recorded":float(np.percentile(speed_r,95)),"p95_speed_translated":float(np.percentile(speed_t,95)),
 "control_integral_recorded":energy_r,"control_integral_translated":energy_t},
 "criteria":criteria,"qualified":all(criteria.values())}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2)); raise SystemExit(0 if report["qualified"] else 2)
