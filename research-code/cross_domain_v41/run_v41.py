#!/usr/bin/env python3
"""Corrected V40 protocol using actual V28 switching in five extension domains."""
from pathlib import Path
import json, sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'cross_domain_v40'),str(ROOT/'cross_domain_v28')]
import run_v40 as protocol
import run_v28 as switching_extension
class ExtensionAdapter:
    @staticmethod
    def simulate(domain,n,rho,seed,policy):
        return switching_extension.simulate((domain,n,rho,seed,policy))
protocol.HERE=HERE
protocol.OUT=HERE/'results'; protocol.OUT.mkdir(exist_ok=True)
protocol.CONTRACT_PATH=HERE/'V41_FROZEN_MODEL_AUDIT_CONTRACT.json'
protocol.OUTPUT_PREFIX='v41'
protocol.C=json.loads(protocol.CONTRACT_PATH.read_text())
protocol.ext5=ExtensionAdapter
if __name__=='__main__': protocol.main()
