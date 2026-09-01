#!/usr/bin/env python3
import csv,json,sys
from pathlib import Path
import numpy as np

def main(root: Path):
    required=['raw_runs.csv','raw_representative_trajectories.npz','dt_audit.csv','metadata.json']
    errors=[f'missing {x}' for x in required if not (root/x).is_file()]
    if errors: print('\n'.join(errors)); return 1
    meta=json.loads((root/'metadata.json').read_text())
    rows=list(csv.DictReader((root/'raw_runs.csv').open())); dt=list(csv.DictReader((root/'dt_audit.csv').open()))
    expected=2*3*4*10; expected_dt=2*3*4*1*3
    if len(rows)!=expected: errors.append(f'raw row count {len(rows)} != {expected}')
    if len(dt)!=expected_dt: errors.append(f'dt row count {len(dt)} != {expected_dt}')
    keys={(r['model'],int(r['n']),r['scenario'],int(r['seed'])) for r in rows}
    if len(keys)!=expected: errors.append('duplicate or missing factorial cells')
    if meta.get('gain_search') is not False or meta.get('clipping') is not False: errors.append('protocol integrity flags failed')
    if meta.get('discarded_runs')!=0: errors.append('discarded runs is nonzero')
    if not str(meta.get('clean_semantics','')).startswith('continuous equations'):
        errors.append('clean scenario is not declared stage-consistent continuous')
    for r in rows+dt:
        for field in ('tail20_max_tracking_per_sqrt_n','final_tracking_per_sqrt_n','failure_time'):
            if field not in r: errors.append(f'missing derived field {field}'); break
        n=int(r['n'])
        if abs(float(r['final_tracking_per_sqrt_n'])-float(r['final_tracking'])/np.sqrt(n))>1e-10:
            errors.append('incorrect sqrt(N) final normalization'); break
        if abs(float(r['tail20_max_tracking_per_sqrt_n'])-float(r['tail20_max_tracking'])/np.sqrt(n))>1e-10:
            errors.append('incorrect sqrt(N) tail normalization'); break
    data=np.load(root/'raw_representative_trajectories.npz')
    if len(data.files)!=2*3*4*2: errors.append(f'NPZ array count {len(data.files)} != 48')
    # Identical scenario-free ICs imply exact identical initial metrics across scenarios.
    for model in ('chua','lorenz'):
      for n in (8,16,32):
       for seed in range(10):
        z=[r for r in rows if r['model']==model and int(r['n'])==n and int(r['seed'])==seed]
        for field in ('initial_tracking','initial_observer'):
         if np.ptp([float(r[field]) for r in z])>1e-12: errors.append(f'IC mismatch {model} N{n} seed{seed} {field}')
    # dt audit must include each declared dt; do not require convergence to pass.
    if {float(r['dt']) for r in dt}!={.004,.002,.001}: errors.append('dt levels incomplete')
    # Failed endpoints must never be used in a convergence ratio.
    ldelay=[r for r in dt if r['model']=='lorenz' and r['scenario']=='delay']
    if len(ldelay)!=9 or not all(r['failure_reason'] for r in ldelay):
        errors.append('expected Lorenz-delay failure coverage at all three dt is absent')
    summary_path=root/'dt_summary.csv'; sensitivity_path=root/'dt_failure_sensitivity.csv'
    if summary_path.exists():
        ss=list(csv.DictReader(summary_path.open())); x=[r for r in ss if r['model']=='lorenz' and r['scenario']=='delay']
        if len(x)!=1 or x[0]['status']!='unavailable' or x[0]['median_relative_endpoint_difference_dt002_vs_dt001']:
            errors.append('failed Lorenz-delay endpoint comparison was not marked unavailable')
    else: errors.append('missing dt_summary.csv')
    if not sensitivity_path.exists(): errors.append('missing dt_failure_sensitivity.csv')
    else:
        fs=list(csv.DictReader(sensitivity_path.open())); ld=[r for r in fs if r['model']=='lorenz' and r['scenario']=='delay']
        if len(ld)!=3 or not all(r['failed_dt004']=='True' and r['failed_dt002']=='True' and r['failed_dt001']=='True' for r in ld):
            errors.append('Lorenz-delay failure-time sensitivity incomplete')
    report={'status':'FAIL' if errors else 'PASS','raw_runs':len(rows),'dt_runs':len(dt),
      'reported_failures':sum(bool(r['failure_reason']) for r in rows),'errors':errors}
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    return bool(errors)
if __name__=='__main__': sys.exit(main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'results'))
