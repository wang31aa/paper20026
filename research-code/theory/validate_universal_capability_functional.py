#!/usr/bin/env python3
"""Exhaustively check the greatest-fixed-point operator on distinct games."""
import json
from pathlib import Path

def kernel(states, task, actions, disturbances, transition, safe):
    c=set(task)
    while True:
        pre={x for x in states if any(
            all(safe(x,a,w) and transition(x,a,w) in c for w in disturbances)
            for a in actions[x])}
        new=set(task)&pre
        if new==c:return sorted(c)
        c=new

# Same state count/action count/disturbance count, but different transition and
# task geometry.  A reduced raw feature vector cannot distinguish the games.
states=tuple(range(5)); actions={x:(0,1) for x in states}; W=(-1,1)
games={
 "contracting": dict(task={0,1,2,3}, F=lambda x,a,w:max(0,min(4,x-a)), H=lambda x,a,w:x<=3),
 "mismatch_exposing": dict(task={0,1,2,3}, F=lambda x,a,w:max(0,min(4,x+a+(w>0))), H=lambda x,a,w:not(x==3 and a==1)),
 "disconnected_task": dict(task={0,2,4}, F=lambda x,a,w:x if a==0 else (x+2)%5, H=lambda x,a,w:x in {0,2,4})
}
out={name:kernel(states,g['task'],actions,W,g['F'],g['H']) for name,g in games.items()}
assert out['contracting'] != out['mismatch_exposing']
assert out['disconnected_task'] == [0,2,4]

# Relabel every state and verify conjugacy of the capability kernel.
perm={0:3,1:1,2:4,3:0,4:2}; inv={v:k for k,v in perm.items()}
g=games['contracting']
relabeled=kernel(
    tuple(perm[x] for x in states), {perm[x] for x in g['task']},
    {perm[x]:actions[x] for x in states}, W,
    lambda y,a,w:perm[g['F'](inv[y],a,w)],
    lambda y,a,w:g['H'](inv[y],a,w))
assert relabeled == sorted(perm[x] for x in out['contracting'])

# A task-set enclosure with a common predecessor produces nested kernels.
F=games['contracting']['F']; H=lambda x,a,w:True
enclosure={
    'inner':kernel(states,{0,1,2},actions,W,F,H),
    'nominal':kernel(states,{0,1,2,3},actions,W,F,H),
    'outer':kernel(states,set(states),actions,W,F,H)}
assert set(enclosure['inner']) <= set(enclosure['nominal']) <= set(enclosure['outer'])

# A dimension-changing feedback refinement: two concrete states represent one
# abstract state. The abstract winning kernel must lift into the concrete one.
Xa=tuple(range(4)); Aa={x:(0,1) for x in Xa}; Wa=(-1,1)
Fa=lambda z,a,w:max(0,min(3,z-a)); Ha=lambda z,a,w:z<=2; Ta={0,1,2}
Ka=kernel(Xa,Ta,Aa,Wa,Fa,Ha)
Xc=tuple(range(8)); Ac={x:(0,1) for x in Xc}; Wc=Wa
alpha=lambda x:x//2
Fc=lambda x,a,w:2*Fa(alpha(x),a,w)+(x%2)
Hc=lambda x,a,w:Ha(alpha(x),a,w); Tc={x for x in Xc if alpha(x) in Ta}
Kc=kernel(Xc,Tc,Ac,Wc,Fc,Hc)
lifted=sorted(x for x in Xc if alpha(x) in Ka)
assert set(lifted) <= set(Kc)
refinement_holds=all(
    alpha(Fc(x,a,w)) == Fa(alpha(x),a,w) and Hc(x,a,w) <= Ha(alpha(x),a,w)
    for x in Xc for a in Ac[x] for w in Wc)
assert refinement_holds

# Deliberately corrupt one successor; the finite checker must reject the
# refinement contract rather than silently promote the abstraction.
def Fc_bad(x,a,w):
    return 7 if (x,a,w)==(0,1,1) else Fc(x,a,w)
bad_refinement_holds=all(
    alpha(Fc_bad(x,a,w)) == Fa(alpha(x),a,w) and Hc(x,a,w) <= Ha(alpha(x),a,w)
    for x in Xc for a in Ac[x] for w in Wc)
assert not bad_refinement_holds

payload={
 "schema":"UF-NUMERICAL-CHECK-2","games":out,
 "isomorphic_relabeling":{"permutation":perm,"kernel":relabeled},
 "descriptor_enclosure":enclosure,
 "feedback_refinement":{"abstract_kernel":Ka,"concrete_kernel":Kc,
                         "lifted_abstract_kernel":lifted},
 "checks":{"same_operator_all_games":True,"domain_specific_outputs":True,
           "reduced_raw_features_insufficient":True,
           "isomorphism_conjugacy":True,"kernel_sandwich":True,
           "task_preserving_refinement":refinement_holds,
           "invalid_refinement_rejected":not bad_refinement_holds},
 "scope":"finite exhaustive unit check; not physical cross-domain validation"}
path=Path(__file__).with_name('universal_capability_functional_validation.json')
path.write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
