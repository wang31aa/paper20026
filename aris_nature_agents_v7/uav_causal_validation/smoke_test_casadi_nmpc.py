#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from python_translation.workspace import load_official_workspace
from python_translation.casadi_nmpc import CasadiNMPC
p=argparse.ArgumentParser(); p.add_argument("workspace",type=Path)
p.add_argument("--backend",choices=("sqpmethod","ipopt"),default="sqpmethod")
a=p.parse_args()
w=load_official_workspace(a.workspace); build=time.perf_counter(); c=CasadiNMPC(w.parameters, backend=a.backend); build=time.perf_counter()-build
r=c.solve(w.initial_state)
report={"backend":f"CasADi 3.7.2 {a.backend} sparse symbolic translation","build_seconds":build,
        "solve_seconds":r.solve_seconds,"success":r.success,"iterations":r.iterations,
        "objective":r.objective,"horizon_seconds":w.parameters.horizon_seconds,
        "horizon_steps":w.parameters.horizon_steps,"official_acados_replay":False,
        "scope":"full-horizon optimizer smoke test; not complete trajectory qualification"}
(HERE/'results/casadi_nmpc_smoke_test.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
