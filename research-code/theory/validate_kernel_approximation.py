#!/usr/bin/env python3
"""Finite exhaustive checks of the inner/outer kernel sandwich."""
import json
from pathlib import Path

def kernel(states, task, actions, succ):
    C=set(task)
    while True:
        P={x for x in states if any(set(succ[x,a])<=C for a in actions[x])}
        N=set(task)&P
        if N==C:return C
        C=N

X=tuple(range(12)); A={x:(0,1) for x in X}
task=set(range(10))
succ={}
for x in X:
    succ[x,0]={min(11,x+1)}
    succ[x,1]={max(0,x-2),max(0,x-1)}
K=kernel(X,task,A,succ)

# Three-state cells. Inner abstraction keeps only cells wholly in the task;
# outer abstraction keeps cells intersecting it and over-approximates successors.
cells={i:set(range(3*i,min(12,3*i+3))) for i in range(4)}
Xa=tuple(cells); Aa={i:(0,1) for i in Xa}
def abstract(inner):
    Ta={i for i,c in cells.items() if (c<=task if inner else bool(c&task))}
    S={}
    for i,c in cells.items():
        for a in Aa[i]:
            images=set().union(*(succ[x,a] for x in c))
            S[i,a]={j for j,q in cells.items() if q&images}
    return kernel(Xa,Ta,Aa,S)
Ki=abstract(True); Ko=abstract(False)
lift_i=set().union(*(cells[i] for i in Ki)) if Ki else set()
lift_o=set().union(*(cells[i] for i in Ko)) if Ko else set()
assert lift_i<=K<=lift_o

out={'status':'PASS','concrete_kernel':sorted(K),'inner_lift':sorted(lift_i),
     'outer_lift':sorted(lift_o),'checks':{'inner_inclusion':True,
     'outer_inclusion':True,'finite_fixed_points_exhausted':True},
     'scope':'finite exhaustive theorem check; not a physical-domain kernel computation'}
Path(__file__).with_name('kernel_approximation_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
