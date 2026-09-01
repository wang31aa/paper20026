#!/usr/bin/env python3
"""Repeated wall-clock SIL timing gate; this never promotes HIL status."""
from pathlib import Path
import json, subprocess, sys

HERE=Path(__file__).resolve().parent; runs=[]
for repeat in range(10):
    subprocess.run([sys.executable,str(HERE/'run_sil.py')],check=True,stdout=subprocess.DEVNULL)
    status=json.loads((HERE/'results/status.json').read_text())
    runs.append({'repeat':repeat,'deadline_misses':status['deadline_misses'],
                 'maximum_compute_time_s':status['maximum_compute_time_s'],
                 'wall_time_s':status['wall_time_s']})
payload={'schema':'REPEATED-SIL-TIMING-1','runs':runs,
         'total_cycles':sum(500 for _ in runs),
         'total_deadline_misses':sum(x['deadline_misses'] for x in runs),
         'maximum_compute_time_s':max(x['maximum_compute_time_s'] for x in runs),
         'repeated_realtime_sil_qualified':all(x['deadline_misses']==0 for x in runs),
         'hil_executed':False,
         'interpretation':'Repeated host SIL timing only; no external clock, plant, I/O or HIL endpoint.'}
(HERE/'results/repeated_timing_status.json').write_text(json.dumps(payload,indent=2)+'\n')
status=json.loads((HERE/'results/status.json').read_text())
status['status']='sil_interface_qualified' if payload['repeated_realtime_sil_qualified'] else 'sil_interface_qualified_with_repeat_deadline_misses'
status['realtime_deadline_qualified']=payload['repeated_realtime_sil_qualified']
status['deadline_misses']=payload['total_deadline_misses']
status['maximum_compute_time_s']=payload['maximum_compute_time_s']
status['timing_repeats']=len(runs)
status['timing_cycles']=payload['total_cycles']
(HERE/'results/status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(payload,indent=2))
