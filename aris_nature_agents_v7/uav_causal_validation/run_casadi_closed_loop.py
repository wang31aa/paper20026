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
maximum_steps=min(a.max_steps,max(1,len(w.times)-1))
direction=np.asarray(w.parameters.reference_direction,dtype=float)
direction=direction/np.linalg.norm(direction)
source_end_projection=float(np.min(w.positions[-1].reshape(w.parameters.n,3)@direction))
x=w.initial_state.copy(); rows=[]; completed=False
for k in range(maximum_steps):
    r=c.solve(x); u=r.controls[0]; m=constraint_margins(x,u,w.parameters)
    pos,vel=split_state(x,w.parameters.n)
    rows.append({"step":k,"time":k*w.parameters.dt,"state":x.tolist(),
      "requested_control":u.tolist(),"applied_control":u.tolist(),
      "solver_success":r.success,"solver_iterations":r.iterations,
      "solver_return_status":r.return_status,
      "solver_official_tolerance_pass":r.official_tolerance_pass,
      "solver_primal_infeasibility":r.primal_infeasibility,
      "solver_dual_infeasibility":r.dual_infeasibility,
      "solver_complementarity":r.complementarity,
      "solver_seconds":r.solve_seconds,"objective":r.objective,
      "minimum_predicted_constraint_margin":float(r.minimum_constraint_margin),
      "minimum_pair_margin":float(np.min(m["pair"])),
      "minimum_obstacle_margin":float(np.min(m["obstacle"])) if len(m["obstacle"]) else None,
      "mean_speed":float(np.mean(np.linalg.norm(vel,axis=1)))})
    if not r.success: break
    x=exact_step(x,u,w.parameters.dt,w.parameters.n)
    if np.all(split_state(x,w.parameters.n)[0]@direction>=source_end_projection):
        completed=True; break
a.log.parent.mkdir(parents=True,exist_ok=True)
a.log.write_text("\n".join(json.dumps(x,separators=(",",":")) for x in rows)+"\n")
summary={"backend":a.backend,"completed":completed,"steps":len(rows),
         "maximum_source_steps":maximum_steps,
         "source_end_projection":source_end_projection,
         "all_solver_steps_passed":bool(rows) and all(x["solver_success"] for x in rows),
         "all_steps_passed_official_solver_tolerances":bool(rows) and all(x["solver_official_tolerance_pass"] for x in rows)}
(a.log.parent/"casadi_free_run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
# This program is a trajectory producer.  It must preserve an incomplete or
# failed run for the downstream fail-closed replay scorer instead of aborting
# before the frozen metrics can be computed.
