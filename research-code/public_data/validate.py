#!/usr/bin/env python3
"""Validate public-source schema, rights decisions and four-layer boundaries."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
LAYERS = ("raw", "calibrated_models", "simulation_results", "source_data")
RIGHTS = {
    "permitted-with-attribution",
    "permitted-with-attribution-file-manifest-not-yet-frozen",
    "hold-conflicting-CC-BY-4.0-and-CC-BY-SA-4.0-statements",
    "not-cleared-no-explicit-dataset-licence",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL public-data provenance: {message}")


sources_doc = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
checks_doc = json.loads((ROOT / "checksums.json").read_text(encoding="utf-8"))
artifacts_doc = json.loads((ROOT / "artifacts.json").read_text(encoding="utf-8"))
sources = sources_doc.get("sources")
if sources_doc.get("schema_version") != "1.0" or not isinstance(sources, list):
    fail("sources.json schema/version")
required = {"id", "title", "doi", "version", "landing_page", "creators",
            "licence_spdx", "redistribution_permitted", "redistribution_status",
            "raw_present_in_release", "local_acquisition_status", "analysis_scope",
            "claim_role"}
ids: set[str] = set()
for item in sources:
    missing = required - set(item)
    if missing:
        fail(f"{item.get('id', '<unknown>')} missing {sorted(missing)}")
    if item["id"] in ids:
        fail(f"duplicate source id {item['id']}")
    ids.add(item["id"])
    if item["redistribution_status"] not in RIGHTS:
        fail(f"unrecognised rights status for {item['id']}")
    if item["raw_present_in_release"] is not False:
        fail(f"raw release inclusion must remain false for {item['id']}")
    if item["redistribution_status"].startswith(("hold-", "not-cleared-")) and item["redistribution_permitted"] is True:
        fail(f"rights hold cannot be marked permitted for {item['id']}")
if ids != set(checks_doc.get("datasets", {})):
    fail("source/checksum dataset identifiers differ")
for layer in LAYERS:
    if not (ROOT / layer / "README.md").is_file():
        fail(f"missing layer contract: {layer}/README.md")
artifacts = artifacts_doc.get("artifacts")
if artifacts_doc.get("schema_version") != "1.0" or not isinstance(artifacts, list):
    fail("artifacts.json schema/version")
artifact_required = {"path", "sha256", "data_layer", "dataset_id", "dataset_version",
                     "producer", "synthetic", "claim_boundary"}
for artifact in artifacts:
    missing = artifact_required - set(artifact)
    if missing:
        fail(f"artifact missing {sorted(missing)}")
    if artifact["data_layer"] not in LAYERS[1:]:
        fail(f"derived artifact has invalid layer: {artifact['path']}")
    if artifact["dataset_id"] not in ids:
        fail(f"artifact has unknown dataset: {artifact['path']}")
    path = PROJECT / artifact["path"]
    if not path.is_file() or digest(path.read_bytes()) != artifact["sha256"]:
        fail(f"derived artifact missing or hash mismatch: {artifact['path']}")

openmct = checks_doc["datasets"]["openmct-v1"]
qualification = json.loads((PROJECT / "cps_transfer_benchmark/results/openmct_qualification/openmct_qualification.json").read_text())
if openmct["archive_sha256"] != qualification["archive"]["expected_sha256"]:
    fail("OpenMCT archive hash differs from qualification")
if openmct["member_bindings"] != qualification["archive_to_extract_bindings"]:
    fail("OpenMCT member bindings differ from qualification")

archive = None
if len(sys.argv) == 3 and sys.argv[1] == "--archive":
    archive = Path(sys.argv[2])
elif len(sys.argv) != 1:
    fail("usage: validate.py [--archive OPENMCT_V1_ZIP]")
if archive is not None:
    if digest(archive.read_bytes()) != openmct["archive_sha256"]:
        fail("OpenMCT archive SHA-256 mismatch")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for logical, binding in openmct["member_bindings"].items():
            member = binding["archive_member"]
            if member not in names or digest(zf.read(member)) != binding["sha256"]:
                fail(f"OpenMCT member mismatch: {logical}")
    print(f"PASS: OpenMCT V1 archive and {len(openmct['member_bindings'])} analysed members are bound")
print(f"PASS: {len(sources)} public sources, {len(artifacts)} derived artifacts, four layers and fail-closed rights states validated")
