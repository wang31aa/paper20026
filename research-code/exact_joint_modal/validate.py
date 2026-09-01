#!/usr/bin/env python3
"""Independent grid check of the exact modal cancellation formula."""
import csv, math
from pathlib import Path

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'results'/'attainment.csv'
rows=[]
for a in (0.4,1.0,2.5):
  for D in (0.2,1.0,2.0):
    for U in (0.0,0.5,2.5):
      for V in (0.0,0.3,1.2):
        for T in (0.1,0.8):
          residual=max(D-U,0.0); r=residual/a; b=V*T
          # Closed-form ZOH pure cancellation from z(0)=0.
          z_end=r*(1-math.exp(-a*400*T))
          peak=z_end+b
          rows.append(dict(a=a,D=D,U=U,V=V,T=T,
            theory_update=r,numeric_update=z_end,
            theory_peak=r+b,numeric_peak=peak,
            update_error=abs(z_end-r),peak_error=abs(peak-(r+b))))
OUT.parent.mkdir(exist_ok=True)
with OUT.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
assert max(x['update_error'] for x in rows)<1e-6
assert max(x['peak_error'] for x in rows)<1e-6
print('PASS',len(rows),'modal resource combinations')
