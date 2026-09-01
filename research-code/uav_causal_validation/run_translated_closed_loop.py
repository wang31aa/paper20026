#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from python_translation.closed_loop import run_nmpc
from python_translation.workspace import load_official_workspace


def main():
    parser = argparse.ArgumentParser(description="Free-run the translated NMPC from an official initial state")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--solver-iterations", type=int, default=100)
    parser.add_argument("--log", type=Path, default=HERE / "results" / "translated_nmpc_free_run.jsonl")
    parser.add_argument("--summary", type=Path, default=HERE / "results" / "translated_nmpc_free_run_summary.json")
    args = parser.parse_args()
    workspace = load_official_workspace(args.workspace)
    summary = run_nmpc(workspace.parameters, workspace.initial_state, workspace.end_line,
                       args.max_steps, args.log, args.solver_iterations)
    payload = asdict(summary) | {"workspace": str(args.workspace),
        "backend": "SciPy SLSQP semantic translation", "official_acados_replay": False,
        "qualification_scope": "free-running translated closed loop only"}
    args.summary.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__": main()

