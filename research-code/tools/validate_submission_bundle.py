#!/usr/bin/env python3
"""Require every submission-bundle payload to equal its authoritative source."""
from __future__ import annotations

import hashlib
import csv
from pathlib import Path

from build_submission_bundle import FILES

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission_bundle"
SOURCE_BY_NAME = {
    ("NC_Rebuilt_SI.pdf" if (ROOT / rel).name == "NC_Rebuilt_SI_Submission.pdf" else (ROOT / rel).name): ROOT / rel
    for rel in FILES
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


errors: list[str] = []

# Require every journal PDF to be newer than all local inputs recorded by
# LaTeX. This prevents a historical rendering from passing copy equality.
for stem, label in [
    ("NC_Rebuilt_Main", "main"),
    ("NC_Rebuilt_SI_Submission", "submission SI"),
    ("NC_Rebuilt_Extended_Data", "Extended Data"),
]:
    pdf = ROOT / "manuscript" / f"{stem}.pdf"
    fls = ROOT / "manuscript" / f"{stem}.fls"
    if not fls.is_file():
        errors.append(f"missing {label} recorder file; rebuild {stem}.tex with -recorder")
        continue
    if not pdf.is_file():
        errors.append(f"missing {label} PDF")
        continue
    newest_input = 0.0
    for line in fls.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        raw = Path(line[6:])
        path = raw if raw.is_absolute() else (ROOT / "manuscript" / raw)
        try:
            resolved = path.resolve()
            resolved.relative_to(ROOT)
        except (OSError, ValueError):
            continue
        if resolved.is_file() and resolved.suffix.lower() in {".tex", ".pdf", ".png", ".jpg", ".jpeg"}:
            newest_input = max(newest_input, resolved.stat().st_mtime)
    if pdf.stat().st_mtime + 1e-6 < newest_input:
        errors.append(f"stale {label} PDF relative to its recorded local inputs")

for required in {"V46_POLICY_SUMMARY.csv", "V46_PAIRED_RESULTS.csv"}:
    if required not in SOURCE_BY_NAME:
        errors.append(f"active V46 source table is not registered: {required}")

v46_runs = ROOT / "vehicle_v46/results/V46_PAIRED_RESULTS.csv"
v46_summary = ROOT / "vehicle_v46/results/V46_POLICY_SUMMARY.csv"
if v46_runs.is_file() and v46_summary.is_file():
    with v46_runs.open(newline="", encoding="utf-8") as stream:
        run_rows = list(csv.DictReader(stream))
    with v46_summary.open(newline="", encoding="utf-8") as stream:
        summary_rows = list(csv.DictReader(stream))
    if len(run_rows) != 864:
        errors.append(f"V46 paired table has {len(run_rows)} rows, expected 864")
    if len(summary_rows) != 8:
        errors.append(f"V46 policy summary has {len(summary_rows)} rows, expected 8")
    if summary_rows and any(int(row["runs"]) != 108 for row in summary_rows):
        errors.append("V46 policy summary does not report 108 conditions per policy")

for name, source in SOURCE_BY_NAME.items():
    bundled = BUNDLE / name
    if not source.is_file() or not bundled.is_file():
        errors.append(f"missing source or bundle: {name}")
    elif digest(source) != digest(bundled):
        errors.append(f"stale bundle copy: {name}")

ledger = BUNDLE / "SHA256SUMS.txt"
if ledger.is_file():
    rows = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        value, name = line.split("  ", 1)
        rows[name] = value
    for name in SOURCE_BY_NAME:
        path = BUNDLE / name
        if path.is_file() and rows.get(name) != digest(path):
            errors.append(f"bundle ledger mismatch: {name}")
else:
    errors.append("missing bundle ledger")

if errors:
    raise SystemExit("FAIL submission bundle\n" + "\n".join(errors))
print(f"PASS: {len(SOURCE_BY_NAME)} submission payloads equal authoritative sources")
