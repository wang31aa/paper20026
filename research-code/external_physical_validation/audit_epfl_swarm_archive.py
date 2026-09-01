#!/usr/bin/env python3
"""Read-only structural audit of Zenodo 4379168 without redistributing raw data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat


EXPECTED_MD5 = "d0dbf0bfb4891a3f34fcb2e381971087"


def digest(path: Path) -> str:
    h = hashlib.md5()  # published repository checksum, not a security claim
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).with_name("results") /
                    "epfl_swarm_archive_audit.json")
    args = ap.parse_args()
    md5 = digest(args.zip)
    if md5 != EXPECTED_MD5:
        raise SystemExit(f"checksum mismatch: {md5}")

    keysets: Counter[tuple[str, ...]] = Counter()
    categories: Counter[str] = Counter()
    hitl = []
    with zipfile.ZipFile(args.zip) as zf:
        mats = [n for n in zf.namelist() if n.endswith(".mat")]
        for name in mats:
            data = loadmat(io.BytesIO(zf.read(name)))
            keys = tuple(sorted(k for k in data if not k.startswith("__")))
            keysets[keys] += 1
            parts = name.split("/")
            controller = parts[2] if len(parts) > 2 else "unknown"
            study = parts[3] if len(parts) > 3 else "unknown"
            categories[f"{controller}/{study}"] += 1
            if "/hitl/" in name:
                hitl.append({
                    "path": name,
                    "keys": keys,
                    "shapes": {k: list(np.asarray(data[k]).shape) for k in keys},
                    "dtypes": {k: str(np.asarray(data[k]).dtype) for k in keys},
                })

    result = {
        "dataset_doi": "10.5281/zenodo.4379168",
        "code_doi": "10.5281/zenodo.4379503",
        "article_doi": "10.1038/s42256-021-00341-y",
        "dataset_version": "v1.0",
        "licence": "CC BY 4.0",
        "archive_bytes": args.zip.stat().st_size,
        "archive_md5": md5,
        "mat_file_count": len(mats),
        "category_counts": dict(sorted(categories.items())),
        "keyset_counts": [
            {"keys": list(keys), "count": count}
            for keys, count in keysets.most_common()
        ],
        "hitl_records": hitl,
        "field_decision": {
            "five_agent_position": "pos_history has 15 columns = 5 x 3",
            "five_agent_velocity": "vel_history has 15 columns = 5 x 3",
            "mpc_simulated_command": "U_history appears in simulation MPC records",
            "hitl_command": "pos_cmd_history is a commanded-position trajectory, not a measured actuator/control input",
            "hitl_low_level_u_t_present": False,
            "leader_pinning_present": False,
            "paper_controller_deployed": False,
        },
        "gate_decision": "REAL_MULTI_AGENT_CLOSED_LOOP_OUTPUT_BENCHMARK_ONLY",
        "claim_boundary": (
            "The archive is strong external physical swarm evidence with MPC/PF "
            "controller families, but it does not deploy the manuscript observer/controller "
            "and the HIL record does not expose low-level u(t)."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(mats)} MAT files; {len(hitl)} HIL record(s); raw archive checksum verified")


if __name__ == "__main__":
    main()
