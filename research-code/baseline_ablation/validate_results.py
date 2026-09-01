#!/usr/bin/env python3
"""Independent structural and arithmetic validator for baseline outputs."""
import csv, json, sys
from pathlib import Path
import numpy as np

p=Path(sys.argv[1] if len(sys.argv)>1 else Path(__file__).parent/'results')
meta=json.loads((p/'metadata.json').read_text()); rows=list(csv.DictReader((p/'raw_runs.csv').open()))
assert len(rows)==len(meta['variants'])*len(meta['heterogeneity_scales'])*len(meta['seeds'])==300
assert all(x['finite']=='True' for x in rows)
assert len({(x['variant'],x['heterogeneity'],x['seed']) for x in rows})==300
raw=np.load(p/'raw_timeseries_metrics.npz'); assert len(raw.files)==300
maxdiff=0.; decision_diff=0.
for vi,v in enumerate(meta['variants']):
 for si,s in enumerate(meta['heterogeneity_scales']):
  for seed in meta['seeds']:
   x=next(q for q in rows if q['variant']==v and float(q['heterogeneity'])==s and int(q['seed'])==seed)
   m=raw[f'v{vi}_s{si}_seed{seed}']; tail=int(.8*len(m))
   vals=[(float(x['final_tracking_error']),m[-1,0]),(float(x['tail20_max_tracking_error']),m[tail:,0].max()),
         (float(x['peak_control']),m[:,3].max())]
   maxdiff=max(maxdiff,max(abs(a-b) for a,b in vals))
   if x['supervisor_available']=='True':
    truth=m[:,0]>meta['tracking_threshold']; pred=m[:,4]>meta['tracking_threshold']
    decision_diff=max(decision_diff,abs(float(x['decision_accuracy'])-np.mean(pred==truth)))
assert maxdiff<1e-12 and decision_diff<1e-12
# Uncertified is exactly the full observer/control dynamics with reporting disabled.
pairdiff=0.
for si,_ in enumerate(meta['heterogeneity_scales']):
 for seed in meta['seeds']:
  pairdiff=max(pairdiff,float(np.max(np.abs(raw[f'v1_s{si}_seed{seed}'][:,:5]-raw[f'v4_s{si}_seed{seed}'][:,:5]))))
assert pairdiff<1e-12
result={'status':'PASS','rows':len(rows),'all_finite':True,'max_summary_recompute_difference':maxdiff,
        'max_decision_recompute_difference':decision_diff,'full_vs_uncertified_dynamics_max_difference':pairdiff}
(p/'validation.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
