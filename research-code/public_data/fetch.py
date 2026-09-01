#!/usr/bin/env python3
"""Opt-in, checksum-gated fetcher for public raw inputs; never runs by default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = json.loads((ROOT / "sources.json").read_text())["sources"]
CHECKS = json.loads((ROOT / "checksums.json").read_text())["datasets"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dataset", choices=["openmct-v1"])
parser.add_argument("--accept-license", action="store_true",
                    help="confirm that the caller accepts the recorded upstream licence")
parser.add_argument("--output", type=Path,
                    help="destination ZIP (default: public_data/raw/<registered name>)")
args = parser.parse_args()
if not args.accept_license:
    raise SystemExit("REFUSED: pass --accept-license after reviewing attribution/OPENMCT.md")
source = next(item for item in SOURCES if item["id"] == args.dataset)
record = CHECKS[args.dataset]
destination = (args.output or (ROOT / "raw" / record["archive_filename"])).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    if sha256(destination) == record["archive_sha256"]:
        print(f"PASS: existing checksum-bound archive: {destination}")
        raise SystemExit(0)
    raise SystemExit(f"REFUSED: existing destination has the wrong checksum: {destination}")
request = urllib.request.Request(source["download_url"], headers={"User-Agent": "nature-tac-repro/1.0"})
fd, temporary_name = tempfile.mkstemp(prefix="openmct-v1-", suffix=".part", dir=destination.parent)
os.close(fd)
temporary = Path(temporary_name)
try:
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        if urllib.parse.urlsplit(response.geturl()).scheme != "https":
            raise SystemExit(f"REFUSED: non-HTTPS redirect target: {response.geturl()}")
        while block := response.read(1024 * 1024):
            out.write(block)
    if sha256(temporary) != record["archive_sha256"]:
        raise SystemExit("REFUSED: downloaded archive SHA-256 mismatch")
    temporary.replace(destination)
finally:
    temporary.unlink(missing_ok=True)
print(f"PASS: downloaded and checksum-bound {args.dataset} to {destination}")
