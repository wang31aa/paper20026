#!/usr/bin/env python3
from __future__ import annotations
import hashlib, itertools, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
C=json.loads((HERE/"V53_FROZEN_CONTRACT.json").read_text())

def states(m): return list(itertools.product(range(m+1), repeat=4))

def transition(x,u,d,m):
    g1,g2,c1,c2=x; u1,u2=u
    return (min(m,max(0,g1-c1)), min(m,max(0,g2-c2)),
            min(m,max(0,c1+d-u1)), min(m,max(0,c2+u1-u2)))

def kernel(m):
    S=states(m); safe={x for x in S if x[0]>=C["safe_gap_index"] and x[1]>=C["safe_gap_index"]}
    K=set(safe); actions=list(itertools.product(C["brake_actions"],repeat=2))
    while True:
        P={x for x in K if any(all(transition(x,u,d,m) in K for d in C["leader_disturbances"]) for u in actions)}
        if P==K: break
        K=P
    return K,len(S),len(safe)

def coarse_cell(y,factor,m):
    # Nearest-grid Voronoi cells.  Boundary cells are one-sided, so the
    # zero-closing-speed face is not spuriously mixed with a positive-speed
    # cell when constructing the certified inner lift.
    return tuple(min(m,(v+factor//2)//factor) for v in y)

def main():
    OUT.mkdir(exist_ok=True); rows=[]
    for m in C["coarse_resolutions"]:
        f=C["refinement_factor"]; fine=2*m
        Kf,nf,sf=kernel(fine); cells={x:[] for x in states(m)}
        for y in states(fine): cells[coarse_cell(y,f,m)].append(y)
        inner={x for x,ys in cells.items() if ys and all(y in Kf for y in ys)}
        outer={x for x,ys in cells.items() if any(y in Kf for y in ys)}
        exact,_,safe=kernel(m)
        assert inner<=exact<=outer
        rows.append({"coarse_resolution":m,"fine_resolution":fine,"fine_states":nf,
                     "fine_safe_states":sf,"fine_kernel_states":len(Kf),
                     "coarse_safe_states":safe,"coarse_exact_states":len(exact),
                     "inner_states":len(inner),"outer_states":len(outer),
                     "ambiguous_states":len(outer-inner),
                     "ambiguity_fraction_of_safe":len(outer-inner)/safe})
    report={"contract_sha256":hashlib.sha256((HERE/"V53_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),
            "rows":rows,"claim_boundary":C["claim_boundary"]}
    (OUT/"V53_REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()
