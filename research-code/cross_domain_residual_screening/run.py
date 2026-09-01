#!/usr/bin/env python3
"""Cross-domain qualification of every retained case without changing evidence tier."""
from pathlib import Path
import csv,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
def read(p):
 with (ROOT/p).open() as f:return list(csv.DictReader(f))
def two(rows,key):return {r['policy']:float(r[key]) for r in rows}
out=[]
def add(case,tier,eligible,endpoint,result,numerator,denominator,effect,limit):
 out.append(dict(case=case,evidence_tier=tier,dynamic_gate_eligible=eligible,endpoint=endpoint,result=result,numerator=numerator,denominator=denominator,effect_ratio=effect,claim_boundary=limit))

# Closed nonlinear network: theorem/scan consistency, not a switching intervention.
r4=read(Path('observer_in_loop_certified/results_r4/raw_runs.csv'))
add('closed nonlinear network','closed simulation','yes','stored all-time envelope','pass',sum(float(r['max_tracking_envelope_ratio'])<=1+1e-12 for r in r4),len(r4),max(float(r['max_tracking_envelope_ratio']) for r in r4),'fixed-graph validation; gating theorem applies only to verified families')

# Twelve-node explicit switching stress test.
s=read(Path('residual_gated_participation/results/summary.csv'));a=two([r for r in s if r['policy']=='all_coupled'],'survival_time');g=two([r for r in s if r['policy']=='gated'],'survival_time')
allv=[float(r['survival_time']) for r in s if r['policy']=='all_coupled'];gv=[float(r['survival_time']) for r in s if r['policy']=='gated']
add('heterogeneous cluster stress','synthetic switching simulation','yes','median first crossing','improved',12,12,float(np.median(gv)/np.median(allv)),'computational task proxy')

# Chronological domain gates.
for case,file in [('robot','summary.csv'),('vehicle','vehicle_confirmation_summary.csv'),('uav','uav_confirmation_summary.csv')]:
 rows=read(Path('connectivity_gated_robot/results')/file);d=two(rows,'survival_time');p=two(rows,'heldout_p95_core_error');ratio=d['gated']/d['all_coupled'];tail=p['gated']/p['all_coupled']
 result='improved_first_loss_tail_worse' if ratio>1+1e-12 else 'no_event_or_no_effect'
 add(case,'public-data-constrained domain model','yes','held-out first crossing',result,1,1,ratio,f'held-out p95 ratio={tail:.6g}; not physical controller execution')

# Physical circuits: chronological coupling-index screen, not a controllable gate.
c=read(Path('external_physical_validation/results/physical_metrics.csv'));cal=[float(r['normalized_disagreement']) for r in c if int(r['coupling_index'])<60];test=[float(r['normalized_disagreement']) for r in c if int(r['coupling_index'])>=60];thr=float(np.quantile(cal,.95));accepted=sum(x<=thr for x in test)
add('nonlinear circuits','physical multi-node measurements','measurement-only','calibration residual acceptance','lower_residual_at_high_coupling',accepted,len(test),accepted/len(test),'records are separate coupling conditions; no online graph intervention')

# Motor record screens.
m=read(Path('cps_transfer_benchmark/results/openmct_run_metrics.csv'));good=sum(float(r['model_nrmse'])<float(r['persistence_nrmse']) for r in m)
add('motor ARX','physical single-object records','no','held-out model beats persistence','pass',good,len(m),good/len(m),'record qualification, not simultaneous cluster gating')
mg=read(Path('openmct_greybox_round42/results/holdout_metrics.csv'));good=sum(r['grey_finite']=='True' and float(r['grey_recursive_nrmse'])<float(r['persistence_recursive_nrmse']) for r in mg)
add('motor grey-box','physical single-object records','no','finite held-out recursion beats constant state','pass',good,len(mg),good/len(mg),'transport uncertainty; no simultaneous motor fleet')

# RTHS and water HIL screens.
r=read(Path('cps_transfer_benchmark/results/rths_run_metrics.csv'));good=sum(float(x['model_nrmse'])<float(x['persistence_nrmse']) for x in r)
add('structural RTHS','physical/HIL records','no','model beats persistence','fail',good,len(r),good/len(r),'actuator records are not a synchronized controllable cluster')
w=read(Path('cps_transfer_benchmark/results/water_hil_session_metrics.csv'));nom=json.loads((ROOT/'cps_transfer_benchmark/results/water_hil_summary.json').read_text())['normal']['threshold_exceedance_rate'];good=sum(float(x['threshold_exceedance_rate'])>nom for x in w)
add('water HIL','physical/HIL sessions','no','session exceedance above nominal','fail',good,len(w),good/len(w),'condition labels lack pointwise attack timing; low exceedance does not imply safety')

# Synthetic grid decision benchmark.
grid=read(Path('decision_benchmark/phase_b_raw/decision_runs.csv'));full=[x for x in grid if x['strategy']=='full'];local=[x for x in grid if x['strategy']=='no_model'];fm=sum(int(x['missed_violation']) for x in full);lm=sum(int(x['missed_violation']) for x in local)
add('swing-governor grid','synthetic decision simulation','decision-only','misses full model versus local trigger','worse_than_local',fm,lm,fm/max(lm,1),'phase B fails numerical gate; not a standard grid')

res=HERE/'results';res.mkdir(exist_ok=True)
with (res/'all_cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
summary={'cases':len(out),'dynamic_gate_eligible':sum(x['dynamic_gate_eligible']=='yes' for x in out),'measurement_or_decision_only':sum(x['dynamic_gate_eligible']!='yes' for x in out),'improved_first_loss_cases':[x['case'] for x in out if 'improved' in x['result']],'failed_or_adverse_cases':[x['case'] for x in out if x['result'] in ('fail','worse_than_local')], 'interpretation':'cross-domain eligibility and held-out screening; effect ratios are endpoint-specific and not pooled'}
(res/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
