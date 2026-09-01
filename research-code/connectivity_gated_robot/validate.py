#!/usr/bin/env python3
import csv,itertools,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import laplacian
R=Path(__file__).resolve().parent/'results'
def rows(name): return list(csv.DictReader((R/name).open()))
robot=rows('summary.csv');vehicle=rows('vehicle_confirmation_summary.csv');uav=rows('uav_confirmation_summary.csv')
assert all(len(x)==2 for x in (robot,vehicle,uav))
def ratio(x,key):
 d={r['policy']:float(r[key]) for r in x};return d['gated']/d['all_coupled']
assert math.isclose(ratio(robot,'survival_time'),1.144/.808,rel_tol=1e-12)
assert ratio(vehicle,'survival_time')==1
assert math.isclose(ratio(uav,'survival_time'),1.2/1.068,rel_tol=1e-12)
mus=[]
for bits in itertools.product([0.,1.],repeat=4):
 a=np.array((1.,)+bits)
 if a.sum()>=3: mus.append(float(np.linalg.eigvalsh(laplacian(a)+laplacian(a).T).min()))
assert len(mus)==11 and min(mus)>0
summary={'robot_survival_ratio':ratio(robot,'survival_time'),'vehicle_survival_ratio':1.0,
 'uav_survival_ratio':ratio(uav,'survival_time'),'robot_p95_ratio':ratio(robot,'heldout_p95_core_error'),
 'vehicle_p95_ratio':ratio(vehicle,'heldout_p95_core_error'),'uav_p95_ratio':ratio(uav,'heldout_p95_core_error'),
 'robot_admissible_modes':len(mus),'robot_common_metric_mu_min':min(mus)}
(R/'validated_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (R/'validated_summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['domain','survival_ratio','p95_ratio'])
 w.writerow(['robot',summary['robot_survival_ratio'],summary['robot_p95_ratio']])
 w.writerow(['vehicle',summary['vehicle_survival_ratio'],summary['vehicle_p95_ratio']])
 w.writerow(['uav',summary['uav_survival_ratio'],summary['uav_p95_ratio']])
print(json.dumps(summary,indent=2));print('PASS: switching metric and three calibration/holdout domain comparisons')
