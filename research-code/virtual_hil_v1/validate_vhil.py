#!/usr/bin/env python3
from pathlib import Path
import csv,json
H=Path(__file__).resolve().parent; C=json.loads((H/'VHIL_V1_FROZEN_CONTRACT.json').read_text())
S=json.loads((H/'results/qualification_registry.json').read_text()); R=list(csv.DictReader((H/'results/vhil_cycles.csv').open()))
assert S['qualification']=='process_isolated_virtual_hil' and S['hardware_hil'] is False and S['physical_experiment'] is False
assert len(R)==len(C['domains'])*len(C['policies'])*C['cycles']
assert set(S['domains'])==set(C['domains'])
failures=[]
for d in C['domains']:
    assert S['domains'][d]['minimum_parameter_spread']>0
    if S['domains'][d]['deadline_miss_fraction']>C['acceptance']['deadline_miss_fraction_max']:
        failures.append({'domain':d,'deadline_miss_fraction':S['domains'][d]['deadline_miss_fraction']})
    for p in C['policies']:
        rr=[r for r in R if r['domain']==d and r['policy']==p]
        assert len(rr)==C['cycles']; assert [int(x['cycle']) for x in rr]==list(range(C['cycles']))
        assert all(len(json.loads(x['state_json']))==C['nodes'] for x in rr)
        assert all(len(json.loads(x['adjacency_json']))==C['nodes'] for x in rr)
    finals=[S['domains'][d]['policy_final_margins'][p] for p in C['policies']]
    assert max(finals)-min(finals)>1e-10, (d,finals)
if S['deadline_miss_fraction']>C['acceptance']['deadline_miss_fraction_max']:
    failures.append({'domain':'all','deadline_miss_fraction':S['deadline_miss_fraction']})
if failures:
    raise SystemExit('FAIL_CLOSED: functional virtual-HIL logs complete, but hard-real-time qualification failed: '+json.dumps(failures,sort_keys=True))
print(f"PASS: {len(C['domains'])} process-isolated heterogeneous virtual-HIL domains, {len(R)} cycle rows; hardware/physical promotion remains false")
