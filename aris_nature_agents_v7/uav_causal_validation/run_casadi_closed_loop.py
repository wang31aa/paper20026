#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from python_translation.casadi_nmpc import CasadiNMPC
from python_translation.model import constraint_margins, exact_step, split_state
from python_translation.workspace import load_official_workspace

p=argparse.ArgumentParser()
p.add_argument("workspace",type=Path); p.add_argument("--max-steps",type=int,default=1000)
p.add_argument("--backend",choices=("sqpmethod","ipopt"),default="sqpmethod")
p.add_argument("--log",type=Path,default=HERE/"results/casadi_free_run.jsonl")
a=p.parse_args(); w=load_official_workspace(a.workspace); c=CasadiNMPC(w.parameters,a.backend)
x=w.initial_state.copy(); rows=[]; completed=False
for k in range(a.max_steps):
    r=c.solve(x); u=r.controls[0]; m=constraint_margins(x,u,w.parameters)
    pos,vel=split_state(x,w.parameters.n)
    rows.append({"step":k,"time":k*w.parameters.dt,"state":x.tolist(),
      "requested_control":u.tolist(),"applied_control":u.tolist(),
      "solver_success":r.success,"solver_iterations":r.iterations,
      "solver_seconds":r.solve_seconds,"objective":r.objective,
      "minimum_predicted_constraint_margin":float(r.minimum_constraint_margin),
      "minimum_pair_margin":float(np.min(m["pair"])),
      "minimum_obstacle_margin":float(np.min(m["obstacle"])) if len(m["obstacle"]) else None,
      "mean_speed":float(np.mean(np.linalg.norm(vel,axis=1)))})
    if not r.success: break
    x=exact_step(x,u,w.parameters.dt,w.parameters.n)
    if np.all(split_state(x,w.parameters.n)[0][:,0]>w.end_line): completed=True; break
a.log.parent.mkdir(parents=True,exist_ok=True)
a.log.write_text("\n".join(json.dumps(x,separators=(",",":")) for x in rows)+"\n")
summary={"backend":a.backend,"completed":completed,"steps":len(rows),
         "all_solver_steps_passed":bool(rows) and all(x["solver_success"] for x in rows)}
(a.log.parent/"casadi_free_run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
if not summary["completed"] or not summary["all_solver_steps_passed"]: raise SystemExit(2)
