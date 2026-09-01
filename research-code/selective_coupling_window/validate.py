#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'results/selective_window.csv').open()))
assert len(rows)==60
assert len({(r['topology'],r['rho']) for r in rows})==60
for r in rows:
    vals=[float(r[k]) for k in ('rho','exact_policy_upper','lyapunov_certificate_upper','lower_ultimate','lower_peak_information')]
    assert all(math.isfinite(x) and x>=0 for x in vals)
    assert float(r['exact_policy_upper'])+1e-12>=float(r['lower_ultimate'])
    assert float(r['lyapunov_certificate_upper'])+1e-12>=float(r['exact_policy_upper'])
    assert int(r['exact_policy_certified'])==int(float(r['exact_policy_upper'])<=.15)
    assert int(r['lyapunov_certified'])==int(float(r['lyapunov_certificate_upper'])<=.15)
    assert int(r['ultimate_impossible'])==int(float(r['lower_ultimate'])>.15)
    assert int(r['peak_impossible'])==int(float(r['lower_peak_information'])>.20)
summary=json.loads((HERE/'results/summary.json').read_text())
assert summary['analysis_status']=='exploratory_protocol_deviation'
assert set(summary['topologies'])=={'direct_pinning','directed_chain','directed_cyclic'}
print('PASS: 60 exploratory opposing-exclusion evaluations validated')
