#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--module",type=Path,required=True); ap.add_argument("--archive",type=Path); ap.add_argument("--raw",type=Path); ap.add_argument("--results",type=Path,required=True)
    a=ap.parse_args(); protocol=json.loads((a.module/"PROTOCOL.json").read_text()); result=json.loads((a.results/"qualification.json").read_text())
    frozen=json.loads((a.results/"SHA256SUMS.json").read_text()); hash_ok=all(sha(a.results/name)==digest for name,digest in frozen.items())
    old=pd.read_csv(a.results/"holdout_metrics.csv")
    if (a.archive is None) != (a.raw is None):
        raise SystemExit("--archive and --raw must be supplied together for optional raw rebuild")
    checks={
      "protocol_id":protocol["protocol_id"]=="OPENMCT-GB-R42-1",
      "protocol_declares_pre_holdout_freeze":protocol["frozen_before_holdout_execution"] is True,
      "protocol_hash_matches":result["protocol_sha256"]==sha(a.module/"PROTOCOL.json"),
      "result_hashes_match":hash_ok,
      "exact_3_7_split":len(result["analysed_input_sha256"])==10 and len(old)==7,
      "no_reference_in_greybox_equation":"reference" not in protocol["model"]["continuous_equation"].lower(),
      "authorization_equals_all_checks":result["network_simulation_authorized"]==all(result["checks"].values()),
      "failure_has_stop_reasons":result["qualification_passed"] or bool(result["stop_reason"]),
      "claim_boundary_nonphysical":"not controller deployment" in result["claim_boundary"]
    }
    mode="frozen-results"
    if a.archive is not None:
        mode="raw-rebuild"
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"results"; subprocess.run([sys.executable,str(a.module/"run_qualification.py"),"--archive",str(a.archive),"--raw",str(a.raw),"--out",str(out)],check=True,capture_output=True,text=True)
            fresh=json.loads((out/"qualification.json").read_text()); new=pd.read_csv(out/"holdout_metrics.csv")
            pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-12)
            checks["fresh_result_identity"]=fresh==result
    report={"passed":all(checks.values()),"mode":mode,"checks":checks,"scientific_gate_passed":result["qualification_passed"],"network_simulation_authorized":result["network_simulation_authorized"]}
    (a.results/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]: raise SystemExit(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
