#!/usr/bin/env python3
"""Numerically audit the derivative-based classification theorem."""
import json, math, random
from pathlib import Path

rng=random.Random(20260820)
cases=[]
for _ in range(500):
 A=10**rng.uniform(-2,1); gamma=10**rng.uniform(-1,1)
 f=rng.random(); B=10**rng.uniform(-2,1); C=10**rng.uniform(-3,0)
 lo=10**rng.uniform(-1,-.05); hi=lo+10**rng.uniform(-.2,.7)
 def d(r): return -A/(gamma*r*r)+B*f+2*C*r*f
 def dd(r): return 2*A/(gamma*r**3)+2*C*f
 # Increasing derivative implies one of the three exhaustive cases.
 assert dd(lo)>0 and dd(hi)>0
 if d(lo)>=0: kind='increasing'
 elif d(hi)<=0: kind='decreasing'
 else:
  a,b=lo,hi
  for __ in range(90):
   m=(a+b)/2
   if d(m)<0:a=m
   else:b=m
  root=(a+b)/2
  assert lo<root<hi and abs(d(root))<1e-10
  kind='interior_minimum'
 cases.append(kind)
counts={k:cases.count(k) for k in sorted(set(cases))}
out={'seed':20260820,'cases':len(cases),'classification_counts':counts,'strict_convexity_checks_passed':True,'root_residual_tolerance':1e-10}
path=Path(__file__).with_name('task_lifted_classifier_validation.json')
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
