#!/usr/bin/env python3
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
C=json.loads((HERE/'VHIL_V2_FROZEN_CONTRACT.json').read_text()); S=json.loads((HERE/'results/qualification_registry.json').read_text())
V1=json.loads((ROOT/'virtual_hil_v1/results/qualification_registry.json').read_text()); R=list(csv.DictReader((HERE/'results/tcp_cycles.csv').open()))
assert len(R)==C['acceptance']['complete_rows']; assert not S['hardware_hil'] and not S['entity_platform']
diff={}
for d in C['domains']:
    assert S['domains'][d]['minimum_parameter_spread']>0
    diff[d]={}
    for p in C['policies']:
        a=S['domains'][d]['final_margins'][p]; b=V1['domains'][d]['policy_final_margins'][p]; diff[d][p]=abs(a-b)
        assert diff[d][p]<=C['acceptance']['cross_backend_final_margin_tolerance'],(d,p,a,b)
out={'checks':{'complete':True,'heterogeneous':True,'cross_backend_agreement':True,'hardware_hil':False,'entity_platform':False},'absolute_final_margin_differences':diff,'interpretation':'Two software transports agree; this is implementation replication, not independent entity-HIL evidence.'}
(HERE/'results/cross_backend_validation.json').write_text(json.dumps(out,indent=2)+'\n'); print('PASS: eight domains agree across process-pipe and TCP virtual-platform backends; entity/HIL promotion remains false')
