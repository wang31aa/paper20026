#!/usr/bin/env python3
"""Generate immutable exogenous innovations shared by every policy arm."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scenario-id',required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--steps',type=int,required=True)
    p.add_argument('--agents',type=int,default=5); p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--noise-sd',type=float,default=.01); p.add_argument('--wind-sd',type=float,default=.05)
    p.add_argument('--loss-probability',type=float,default=.05); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); rng=np.random.default_rng(args.seed)
    measurement=rng.normal(0,args.noise_sd,(args.steps,args.agents,6))
    wind=rng.normal(0,args.wind_sd,(args.steps,args.agents,3))
    packet=rng.random((args.steps,args.agents,args.agents))>=args.loss_probability
    for k in range(args.steps): np.fill_diagonal(packet[k],True)
    latency=rng.integers(0,3,(args.steps,args.agents,args.agents),endpoint=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,measurement_noise=measurement,wind=wind,
                        packet_received=packet,latency_steps=latency)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest={"scenario_id":args.scenario_id,"seed":args.seed,"steps":args.steps,
              "agents":args.agents,"dt":args.dt,"sha256":digest,
              "consumption_rule":"the same file is read-only input to every plant-policy arm"}
    manifest_path=args.output.with_suffix(args.output.suffix+'.json')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
