#!/usr/bin/env python3
import csv,hashlib,json,math
from pathlib import Path
H=Path(__file__).resolve().parent;O=H/'results'
d=list(csv.reader((O/'MSS_OTTER_DERIVATIVE_WITNESSES.csv').open()));t=list(csv.reader((O/'MSS_OTTER_PID_REPLAY.csv').open()))
vals=[[float(v) for v in r] for r in d];trace=[[float(v) for v in r] for r in t]
checks={'five_derivative_witnesses':len(vals)==5 and all(len(r)==14 for r in vals),'finite_derivatives':all(math.isfinite(v) for r in vals for v in r),'closed_loop_rows':len(trace)==12001 and all(len(r)==17 for r in trace),'finite_closed_loop':all(math.isfinite(v) for r in trace for v in r),'time_endpoint':abs(trace[-1][0]-600)<1e-9,'official_runtime_recorded':(O/'MSS_RUNTIME.txt').exists()}
report={'status':'PLANT_AND_PID_RUNTIME_QUALIFIED' if all(checks.values()) else 'NOT_QUALIFIED','checks':checks,'source_commit':'99bf0b30e9ae0dca3515d02e0bbd06c54cc0c2f7','derivative_sha256':hashlib.sha256((O/'MSS_OTTER_DERIVATIVE_WITNESSES.csv').read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256((O/'MSS_OTTER_PID_REPLAY.csv').read_bytes()).hexdigest(),'scope':'official OTTER plant and deterministic SIMotter PID-branch runtime; not heterogeneous multi-vessel intervention, HIL or physical validation'}
(O/'MSS_REPLAY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
