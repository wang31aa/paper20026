#!/usr/bin/env python3
"""Generate a prospective three-arm acquisition schedule, never outcome data."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import random
import sys

from contract_common import ARMS, PROTOCOL_ID, require_safe_id, sha256_bytes

FIELDS = [
    "protocol_id", "schedule_status", "block_id", "profile_id", "load_id",
    "repeat_id", "arm", "acquisition_order", "planned_run_id",
    "randomisation_seed",
]


def render_schedule(profiles: list[str], loads: list[str], blocks: int, seed: int) -> bytes:
    rng = random.Random(seed)
    rows: list[dict[str, str | int]] = []
    order = 0
    for profile in profiles:
        for load in loads:
            for repeat in range(1, blocks + 1):
                block_id = f"B-{profile}-{load}-{repeat:03d}"
                shuffled = list(ARMS)
                rng.shuffle(shuffled)
                for arm in shuffled:
                    order += 1
                    rows.append({
                        "protocol_id": PROTOCOL_ID,
                        "schedule_status": "planned_not_executed",
                        "block_id": block_id,
                        "profile_id": profile,
                        "load_id": load,
                        "repeat_id": repeat,
                        "arm": arm,
                        "acquisition_order": order,
                        "planned_run_id": f"R-{profile}-{load}-{repeat:03d}-{arm}",
                        "randomisation_seed": seed,
                    })
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True,
                        help="Prospectively named leader profile; repeat for multiple profiles.")
    parser.add_argument("--load", action="append", required=True,
                        help="Prospectively named load condition; repeat for multiple loads.")
    parser.add_argument("--blocks", type=int, required=True,
                        help="Matched blocks per profile/load; must be at least 10.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frozen-at-utc", required=True,
                        help="Registration timestamp supplied by the experimenter; no clock is inferred.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Must not already exist; overwrite is prohibited.")
    args = parser.parse_args()

    if args.blocks < 10:
        parser.error("--blocks must be at least 10")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    profiles = list(dict.fromkeys(args.profile))
    loads = list(dict.fromkeys(args.load))
    for value in profiles:
        require_safe_id(value, "profile")
    for value in loads:
        require_safe_id(value, "load")
    if args.out_dir.exists():
        parser.error("--out-dir already exists; schedule overwrite is prohibited")

    schedule = render_schedule(profiles, loads, args.blocks, args.seed)
    schedule_hash = sha256_bytes(schedule)
    manifest = {
        "schema": "PHYS-E2E-OCT-R34-randomisation-v1",
        "protocol_id": PROTOCOL_ID,
        "status": "future_plan_only_not_executed",
        "frozen_at_utc": args.frozen_at_utc,
        "profiles": profiles,
        "loads": loads,
        "matched_blocks_per_profile_load": args.blocks,
        "arms": list(ARMS),
        "randomisation_seed": args.seed,
        "planned_block_count": len(profiles) * len(loads) * args.blocks,
        "planned_run_count": len(profiles) * len(loads) * args.blocks * len(ARMS),
        "schedule_file": "randomisation_schedule.csv",
        "schedule_sha256": schedule_hash,
        "contains_measurements": False,
        "contains_outcomes": False,
        "warning": "This is a prospective acquisition plan, not an experiment result.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "randomisation_schedule.csv").write_bytes(schedule)
    (args.out_dir / "randomisation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: wrote future plan only; {manifest['planned_block_count']} blocks, "
          f"{manifest['planned_run_count']} planned runs, sha256={schedule_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
