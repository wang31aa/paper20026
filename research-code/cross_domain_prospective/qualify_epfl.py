#!/usr/bin/env python3
"""Run-level qualification of the EPFL five-agent MPC/PF archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXPECTED_MD5 = "d0dbf0bfb4891a3f34fcb2e381971087"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair_min(pos: np.ndarray) -> float:
    best = float("inf")
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            best = min(best, float(np.linalg.norm(pos[i] - pos[j])))
    return best


def split_for(path: str) -> str:
    # Path order is frozen independently of outcomes, within controller/study.
    run = int(path.split("/")[-2]) if path.split("/")[-2].isdigit() else 0
    if run <= 5:
        return "calibration"
    if run <= 7:
        return "validation"
    return "retrospective_test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=Path(__file__).with_name("results"))
    args = ap.parse_args()
    if md5(args.zip) != EXPECTED_MD5:
        raise SystemExit("EPFL archive checksum mismatch")

    rows, exclusions = [], []
    with zipfile.ZipFile(args.zip) as zf:
        all_mats = sorted(n for n in zf.namelist() if n.endswith(".mat"))
        names = [n for n in all_mats if n.endswith("workspace.mat")]
        exclusions.extend({"path": n, "reason": "not_a_workspace_record"}
                          for n in all_mats if not n.endswith("workspace.mat"))
        for name in names:
            d = loadmat(io.BytesIO(zf.read(name)), squeeze_me=True, struct_as_record=False)
            pos_raw = np.asarray(d["pos_history"], float)
            vel_raw = np.asarray(d["vel_history"], float)
            if pos_raw.ndim != 2 or pos_raw.shape[1] != 15:
                exclusions.append({"path": name, "reason": f"pos_history_shape={list(pos_raw.shape)}"})
                continue
            pos = pos_raw.reshape(len(pos_raw), 5, 3)
            vel = vel_raw.reshape(len(vel_raw), 5, 3)
            s = d["S"]
            radius = float(getattr(s, "r_coll", np.nan))
            max_v = float(getattr(s, "max_v", np.nan))
            min_sep = min(pair_min(frame) for frame in pos)
            max_speed = float(np.linalg.norm(vel, axis=2).max())
            control = np.asarray(d.get("U_history", d.get("accel_history", [])), float)
            control_l2 = float(np.sum(control * control)) if control.size else float("nan")
            parts = name.split("/")
            controller, study = parts[2], parts[3]
            rows.append({
                "path": name,
                "controller": controller,
                "study": study,
                "split": "transport_only" if study == "hitl" else split_for(name),
                "samples": len(pos),
                "agents": 5,
                "collision_radius_source": radius,
                "minimum_pair_separation_source": min_sep,
                "separation_margin_source": min_sep - 2.0 * radius,
                "maximum_speed_source": max_speed,
                "declared_max_speed_source": max_v,
                "speed_margin_source": max_v - max_speed,
                "control_l2_source": control_l2,
                "has_low_level_or_sim_control": int("U_history" in d or "accel_history" in d),
                "task_failed_archive_definition": int(min_sep < 2.0 * radius or max_speed > max_v + 1e-9),
            })

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "epfl_run_qualification.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    with (args.outdir / "epfl_exclusions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("path", "reason")); writer.writeheader(); writer.writerows(exclusions)
    summary = {
        "archive_md5": EXPECTED_MD5,
        "mat_files": len(all_mats),
        "records": len(rows),
        "excluded_records": len(exclusions),
        "controllers": {c: sum(r["controller"] == c for r in rows) for c in sorted({r["controller"] for r in rows})},
        "splits": {s: sum(r["split"] == s for r in rows) for s in sorted({r["split"] for r in rows})},
        "archive_defined_failures": sum(r["task_failed_archive_definition"] for r in rows),
        "claim_boundary": "run-level source-controller qualification; no proposed-gate intervention",
    }
    (args.outdir / "epfl_qualification_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
