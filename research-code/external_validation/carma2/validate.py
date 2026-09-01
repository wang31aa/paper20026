#!/usr/bin/env python3
import csv, hashlib, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'cache'/'CARMA2.csv'; RESULT=ROOT/'results'/'vehicle_run_metrics.csv'
EXPECTED='de7c967feee09ed054716dec2fd4e383bcfcacfdb263a890809a8680febc9520'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
rows=list(csv.DictReader(RESULT.open()))
assert len(rows)>0 and len({(r['run'],r['vehicle']) for r in rows})==len(rows)
assert all(r['split'] in {'calibration','validation','test'} for r in rows)
assert all(int(r['eligible_n'])>=100 for r in rows)
for r in rows:
    for k in ('dt_s','median_abs_error','p95_abs_error','rmse','command_std','command_change_fraction'):
        assert math.isfinite(float(r[k])) and float(r[k])>=0
test=[r for r in rows if r['split']=='test' and r['run']=='10']
assert len(test)==5
p95=[float(r['p95_abs_error']) for r in test]
assert max(p95)/min(p95)>2
print('PASS',len(rows),'vehicle-run rows;',len(test),'complete held-out run-10 channels')
