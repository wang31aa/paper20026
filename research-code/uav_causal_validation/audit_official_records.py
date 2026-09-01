#!/usr/bin/env python3
import csv, hashlib, json, shutil, subprocess, tempfile, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/private/tmp/epfl_swarm_dataset.zip')
OUT = HERE / 'results'; OUT.mkdir(exist_ok=True)

def minimum_pair_distance(pos):
    n = pos.shape[1] // 3; best = float('inf')
    xyz = pos.reshape(pos.shape[0], n, 3)
    for i in range(n):
        for j in range(i): best = min(best, float(np.min(np.linalg.norm(xyz[:,i]-xyz[:,j],axis=1))))
    return best

def main():
    rows=[]
    with tempfile.TemporaryDirectory() as td:
        # Info-ZIP can recover this upstream archive's prefixed/offset entries;
        # Python's ZipExtFile rejects many otherwise intact workspace members.
        extracted=Path(td)/'archive'; extracted.mkdir()
        process=subprocess.run(['unzip','-q',str(ARCHIVE),'-d',str(extracted)],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        paths=sorted(extracted.rglob('workspace.mat'))
        if not paths:
            raise RuntimeError(f'archive extraction yielded no workspaces: {process.stderr.decode()}')
        for p in paths:
            name=str(p.relative_to(extracted))
            try: d=loadmat(p,squeeze_me=True,struct_as_record=False)
            except Exception as e:
                rows.append({'path':name,'controller':'unknown','assessable':0,'reason':type(e).__name__}); continue
            controller='mpc' if '/mpc/' in name else ('pf' if '/pf/' in name else 'unknown')
            pos=np.atleast_2d(d.get('pos_history',np.empty((0,0))))
            vel=np.atleast_2d(d.get('vel_history',np.empty((0,0))))
            t=np.ravel(d.get('time_history',np.array([])))
            S=d.get('S',None); n=int(getattr(S,'nb_agents',0)) if S is not None else 0
            ok=pos.shape[0]>1 and pos.shape[1]==3*n and vel.shape==pos.shape and t.size==pos.shape[0]
            if not ok:
                rows.append({'path':name,'controller':controller,'assessable':0,'reason':'shape_or_field'}); continue
            speed=np.linalg.norm(vel.reshape(vel.shape[0],n,3),axis=2)
            rcoll=float(getattr(S,'r_coll',float('nan')))
            mind=minimum_pair_distance(pos)
            row={'path':name,'controller':controller,'assessable':1,'reason':'',
                 'samples':len(t),'agents':n,'duration_s':float(t[-1]-t[0]),
                 'position_range':float(np.ptp(pos)),'minimum_pair_distance':mind,
                 'collision_radius':rcoll,'collision_event':int(mind<2*rcoll),
                 'mean_speed':float(np.mean(speed)),'p95_speed':float(np.quantile(speed,.95)),
                 'control_history_present':int('U_history' in d or 'accel_history' in d)}
            rows.append(row)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (OUT/'official_record_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary={'archive_sha256':hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
             'workspace_records':len(rows),'assessable_records':sum(int(r.get('assessable',0)) for r in rows),
             'mpc_records':sum(r.get('controller')=='mpc' for r in rows),
             'pf_records':sum(r.get('controller')=='pf' for r in rows),
             'collision_events_from_archive_threshold':sum(int(r.get('collision_event',0)) for r in rows),
             'claim_boundary':'L0 record qualification; no source-controller replay or gate intervention'}
    (OUT/'official_record_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
