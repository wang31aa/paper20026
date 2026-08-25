#!/usr/bin/env python3
"""Fail-closed finite checks for the paper's decisive two-layer theory claims."""
from __future__ import annotations
import json, math, random
from itertools import product
from pathlib import Path

def kernel(states, task, actions, disturbances, transition, safe):
    c=set(task)
    while True:
        pre={x for x in states if any(all(safe(x,a,w) and transition(x,a,w) in c
              for w in disturbances) for a in actions[x])}
        new=set(task)&pre
        if new==c:return set(c)
        c=new

# Inner/exact/outer kernel sandwich on an exhaustive finite game.
X=tuple(range(12)); A={x:(0,1) for x in X}; W=(0,)
task=set(range(10))
def F(x,a,w):
    return min(11,x+1) if a==0 else max(0,x-2+w)
def H(x,a,w): return True
K=kernel(X,task,A,W,F,H)
cells={i:set(range(3*i,min(12,3*i+3))) for i in range(4)}
def abstract(inner):
    Xa=tuple(cells); Aa={i:(0,1) for i in Xa}
    Ta={i for i,c in cells.items() if (c<=task if inner else bool(c&task))}
    def Fa(i,a,w):
        images={F(x,a,w) for x in cells[i]}
        hit=[j for j,q in cells.items() if q&images]
        return hit[0] if len(hit)==1 else max(hit)
    return kernel(Xa,Ta,Aa,W,Fa,lambda *_:True)
Ki,Ko=abstract(True),abstract(False)
lift_i=set().union(*(cells[i] for i in Ki)) if Ki else set()
lift_o=set().union(*(cells[i] for i in Ko)) if Ko else set()
assert lift_i<=K<=lift_o

# Coordinate conjugacy and system-specific outputs.
states=tuple(range(5)); actions={x:(0,1) for x in states}; D=(-1,1)
Fc=lambda x,a,w:max(0,min(4,x-a))
Ka=kernel(states,{0,1,2,3},actions,D,Fc,lambda x,a,w:x<=3)
Kb=kernel(states,{0,1,2,3},actions,D,
          lambda x,a,w:max(0,min(4,x+a+(w>0))),
          lambda x,a,w:not(x==3 and a==1))
assert Ka!=Kb
perm={0:3,1:1,2:4,3:0,4:2}; inv={v:k for k,v in perm.items()}
Kr=kernel(tuple(perm[x] for x in states),{perm[x] for x in {0,1,2,3}},
          {perm[x]:actions[x] for x in states},D,
          lambda y,a,w:perm[Fc(inv[y],a,w)],
          lambda y,a,w:inv[y]<=3)
assert Kr=={perm[x] for x in Ka}

# Two-layer composition counterexample and selective-only region.
def gfp(states,task,actions,transition,admissible):
    c=set(task)
    while True:
        n={x for x in c if any(admissible(x,a) and transition(x,a) in c for a in actions[x])}
        if n==c:return c
        c=n
S={"x","i","p","f"}; Acts={s:("I","P") for s in S}
T={("x","I"):"i",("x","P"):"p",("i","I"):"i",("i","P"):"f",
   ("p","I"):"f",("p","P"):"p",("f","I"):"f",("f","P"):"f"}
ki=gfp(S,{"x","i"},Acts,lambda x,a:T[x,a],lambda x,a:a=="I")
kp=gfp(S,{"x","p"},Acts,lambda x,a:T[x,a],lambda x,a:a=="P")
k2=gfp(S,{"x"},Acts,lambda x,a:T[x,a],lambda x,a:False)
assert "x" in ki&kp and "x" not in k2
S2={"x","s","f"}; A2={s:("all","off","selective") for s in S2}
T2={(x,a):("s" if x in {"x","s"} and a=="selective" else "f")
    for x,a in product(S2,A2["x"])}
assert "x" in gfp(S2,{"x","s"},A2,lambda x,a:T2[x,a],lambda x,a:a=="selective")
assert "x" not in gfp(S2,{"x","s"},A2,lambda x,a:T2[x,a],lambda x,a:a=="all")

# Conditional, not universal, participation regimes.
rng=random.Random(20260823); grid=[.2+2.8*i/1200 for i in range(1201)]
counts={"decreasing":0,"increasing":0,"interior":0}; tol=2e-9
for case in range(1200):
    mode=case%3; curves=[]
    for _ in range(4):
        a=10**rng.uniform(-2,.6); c=10**rng.uniform(-2,.4); b=rng.uniform(-.5,.8)
        if mode==0:c=0.
        elif mode==1:a=0.
        curves.append((a,c,b))
    y=[max(b+a/r+c*r*r for a,c,b in curves) for r in grid]
    d=[v-u for u,v in zip(y,y[1:])]
    assert all(y[i]<=(y[i-1]+y[i+1])/2+tol for i in range(1,len(y)-1))
    feasible=[i for i,v in enumerate(y) if v<=1]
    if feasible: assert feasible==list(range(min(feasible),max(feasible)+1))
    if mode==0: assert all(z<=tol for z in d); counts["decreasing"]+=1
    elif mode==1: assert all(z>=-tol for z in d); counts["increasing"]+=1
    elif d[0]<0<d[-1]:
        assert 0<y.index(min(y))<len(y)-1; counts["interior"]+=1

out={"schema":"DECISIVE-THEORY-CLOUD-CHECK-1","status":"PASS",
     "checks":{"kernel_sandwich":True,"coordinate_conjugacy":True,
       "system_specific_outputs":True,"separate_layers_do_not_compose":True,
       "selective_only_region":True,"conditional_participation_regimes":True},
     "classification_counts":counts,
     "scope":"Independent cloud execution of finite/property checks; not proof text, HIL, physical experiment, or cross-domain physical validation."}
Path("decisive_theory_validation.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
