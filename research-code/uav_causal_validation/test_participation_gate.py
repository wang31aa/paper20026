#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from python_translation.model import SwarmParameters, join_state, stage_cost
from python_translation.participation import (
    is_strongly_connected, repair_connectivity, residual_weights,
)

def main():
    adjacency = np.array([[0,1,0,0,1],[1,0,1,0,0],[0,1,0,1,0],
                          [0,0,1,0,1],[1,0,0,1,0]])
    residual = np.array([0.2, 0.3, 4.0, 0.4, 0.5])
    raw = residual_weights(adjacency, residual, threshold=1.0, floor=0.0)
    # Continuous gate enters the controller cost through non-uniform edge weights.
    assert np.any((raw > 0) & (raw < 1))
    binary = adjacency * (residual[:, None] <= 1.0)
    repaired = repair_connectivity(binary, adjacency, residual, minimum_weight=0.2)
    assert is_strongly_connected(repaired)
    assert not np.array_equal(repaired, adjacency)
    parameters = SwarmParameters(5, .1, .2, 3, 150., .5, np.array([1.,0.,0.]),
                                 .8, 2., .1, np.empty((0,3)))
    position=np.array([[0.,0.,0.],[.8,0,0],[1.6,.1,0],[2.4,0,0],[3.2,-.1,0]])
    state=join_state(position,np.tile([.5,0,0],(5,1))); control=np.zeros(15)
    full_cost=stage_cost(state,control,parameters,adjacency)
    gated_cost=stage_cost(state,control,parameters,repaired)
    assert abs(full_cost-gated_cost)>1e-12
    report={"validator":"PASS","node_level_nonuniform":True,
            "connectivity_repaired":True,"controller_objective_changed":True,
            "full_stage_cost":full_cost,"gated_stage_cost":gated_cost,
            "postprocessing_only":False}
    out=Path(__file__).resolve().parent/'results'/'participation_gate_unit_test.json'
    out.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
