#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
from python_translation.workspace import load_official_workspace
from python_translation.casadi_nmpc import CasadiNMPC
w=load_official_workspace(Path(sys.argv[1])); build=time.perf_counter(); c=CasadiNMPC(w.parameters, backend="sqpmethod"); build=time.perf_counter()-build
r=c.solve(w.initial_state)
report={"backend":"CasADi 3.7.2 SQP/QRQP sparse symbolic translation","build_seconds":build,
        "solve_seconds":r.solve_seconds,"success":r.success,"iterations":r.iterations,
        "objective":r.objective,"horizon_seconds":w.parameters.horizon_seconds,
        "horizon_steps":w.parameters.horizon_steps,"official_acados_replay":False,
        "scope":"full-horizon optimizer smoke test; not complete trajectory qualification"}
(HERE/'results/casadi_nmpc_smoke_test.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
