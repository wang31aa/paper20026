#!/usr/bin/env python3
import json, tempfile, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat

ARCHIVE=Path('/private/tmp/epfl_swarm_dataset.zip')
EXTRACTED=Path('/private/tmp/epfl_uav_workspace/dataset/data/mpc/comparison/01/workspace.mat')
name='dataset/data/mpc/comparison/01/workspace.mat'
if EXTRACTED.is_file():
    d=loadmat(EXTRACTED,squeeze_me=True,struct_as_record=False)
else:
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(ARCHIVE) as z:
        p=Path(td)/'w.mat';p.write_bytes(z.read(name));d=loadmat(p,squeeze_me=True,struct_as_record=False)
pos=np.asarray(d['pos_history'],float);vel=np.asarray(d['vel_history'],float);u=np.asarray(d['U_history'],float);t=np.asarray(d['time_history'],float)
dt=np.diff(t); pred_v=vel[:-1]+dt[:,None]*u
# Position integrator can be explicit, semi-implicit or acados ERK; audit both.
pred_p_exp=pos[:-1]+dt[:,None]*vel[:-1]
pred_p_semi=pos[:-1]+dt[:,None]*vel[1:]
result={'record':name,'velocity_one_step_rmse':float(np.sqrt(np.mean((pred_v-vel[1:])**2))),
        'position_explicit_rmse':float(np.sqrt(np.mean((pred_p_exp-pos[1:])**2))),
        'position_semiimplicit_rmse':float(np.sqrt(np.mean((pred_p_semi-pos[1:])**2))),
        'status':'TRANSLATION_COMPONENT_ONLY_NOT_SOURCE_REPLAY'}
out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
(out/'double_integrator_test.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
