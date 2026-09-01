#!/usr/bin/env python3
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
rows=list(csv.DictReader((OUT/'pressure_summary.csv').open()))
# Freeze prediction on all-coupled outcomes only; threshold chosen on two domains
# from a fixed grid, then applied unchanged to the third.
base=[r for r in rows if r['policy']=='all_coupled']; domains=sorted({r['domain'] for r in base}); out=[]
for test in domains:
 train=[r for r in base if r['domain']!=test];te=[r for r in base if r['domain']==test]
 best=None
 for th in [1+i*.1 for i in range(61)]:
  tp=sum(float(r['risk_score'])>=th and int(r['failure']) for r in train);fn=sum(float(r['risk_score'])<th and int(r['failure']) for r in train)
  tn=sum(float(r['risk_score'])<th and not int(r['failure']) for r in train);fp=sum(float(r['risk_score'])>=th and not int(r['failure']) for r in train)
  ba=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1));cand=(ba,-th,th)
  if best is None or cand>best:best=cand
 th=best[2];tp=fn=tn=fp=0
 for r in te:
  pred=float(r['risk_score'])>=th;y=bool(int(r['failure']));tp+=pred and y;fn+=(not pred) and y;tn+=(not pred) and not y;fp+=pred and not y
 pos=tp+fn; neg=tn+fp
 out.append({'held_out_domain':test,'threshold':th,'n_runs':len(te),'tp':tp,'fn':fn,'tn':tn,'fp':fp,
  'sensitivity':tp/pos if pos else 'NA','specificity':tn/neg if neg else 'NA',
  'false_negative_rate':fn/pos if pos else 'NA'})
with (OUT/'leave_domain_out.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
print(json.dumps(out,indent=2))
