#!/usr/bin/env python3
"""Fail closed on the panel-resolved source/script/output provenance ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "figures" / "PANEL_PROVENANCE.json"

EXPECTED = {
    "Main Figure 1": set("abcd"),
    "Main Figure 2": set("abcde"),
    "Main Figure 3": set("abcd"),
    "Main Figure 4": set("abcd"),
    "Extended Data Figure 1": set("abc"),
    "Extended Data Figure 2": set("abc"),
    "Extended Data Figure 3": set("abcd"),
    "Extended Data Figure 3": set("abcd"),
    "Extended Data Figure 4": set("abcd"),
    "Extended Data Figure 5": set("abc"),
    "Extended Data Figure 6": set("abcd"),
    "Extended Data Figure 7": set("abc"),
    "Extended Data Figure 8": set("abcd"),
}
EXPECTED_BUILDERS = {
    "Fig1_dimensionless_boundary": "boundary_law/derive_boundary_law.py",
    "Fig12_observer_in_loop_certified": "figures/build_observer_in_loop_certified_figure.py",
    "Fig13_mechanism_scan": "figures/build_r4_mechanism.py",
    "Fig14_openmct_greybox_bridge": "figures/build_openmct_greybox_bridge.py",
    "Fig10_cps_transfer": "figures/build_cps_transfer_figure.py",
    "Fig11_openmct_motor": "figures/build_openmct_figure.py",
    "Fig6_ablation": "figures/build_figures.py",
    "Fig2_robustness": "figures/build_figures.py",
    "Fig3_numerics_certificates": "figures/build_figures.py",
    "Fig4_oracle_diagnostic": "figures/build_figures.py",
    "Fig9_matlab_reproduction": "figures/build_matlab_reproduction_figure.py",
    "Fig7_extension": "figures/build_figures.py",
}

# Validate the displays cited by the current manuscript, not archived figures.
EXPECTED = {
    "Main Figure 1": set("abcd"),
    "Main Figure 2": set("abcde"),
    "Main Figure 3": set("abcd"),
    "Main Figure 4": set("abcd"),
    "Extended Data Figure 1": set("abcd"),
    "Extended Data Figure 2": set("abcd"),
    "Extended Data Figure 3": set("abcd"),
    "Extended Data Figure 4": set("abc"),
}
EXPECTED_BUILDERS = {
    "Fig1_dimensionless_boundary": "boundary_law/derive_boundary_law.py",
    "Fig12_observer_in_loop_certified": "figures/build_observer_in_loop_certified_figure.py",
    "Fig13_mechanism_scan": "figures/build_r4_mechanism.py",
    "Fig10_cps_transfer": "figures/build_cps_transfer_figure.py",
    "Fig11_openmct_motor": "figures/build_openmct_figure.py",
    "Fig14_openmct_greybox_bridge": "figures/build_openmct_greybox_bridge.py",
    "Fig15_public_cluster_cases": "public_cluster_cases/run_cases.py",
    "Fig16_discovery_extension": "discovery_extension/run_extension.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_artifact(item: object, context: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"{context}: artifact is not an object")
        return
    relative = item.get("path")
    if not isinstance(relative, str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:
        errors.append(f"{context}: unsafe or missing relative path")
        return
    path = ROOT / relative
    if not path.is_file():
        errors.append(f"{context}: missing {relative}")
        return
    if item.get("bytes") != path.stat().st_size:
        errors.append(f"{context}: size mismatch {relative}")
    if item.get("sha256") != sha256(path):
        errors.append(f"{context}: sha256 mismatch {relative}")


def main() -> None:
    payload = json.loads(LEDGER.read_text(encoding="utf-8"))
    errors: list[str] = []
    records = payload.get("records")
    if payload.get("schema_version") != 1 or payload.get("hash_algorithm") != "sha256":
        errors.append("unsupported schema/hash algorithm")
    if not isinstance(records, list):
        raise SystemExit("FAIL panel provenance\nrecords is not a list")
    expected_count = sum(len(panels) for panels in EXPECTED.values())
    if payload.get("record_count") != len(records) or len(records) != expected_count:
        errors.append(f"record count is {len(records)}, expected {expected_count}")

    seen: dict[str, set[str]] = {name: set() for name in EXPECTED}
    output_hashes: dict[str, set[str]] = {}
    builder_hashes: dict[str, set[str]] = {}
    artifact_checks = 0
    for index, record in enumerate(records):
        ctx = f"record {index + 1}"
        if not isinstance(record, dict):
            errors.append(f"{ctx}: not an object")
            continue
        figure, panel, stem = record.get("figure"), record.get("panel"), record.get("file_stem")
        if figure not in EXPECTED or panel not in EXPECTED.get(figure, set()):
            errors.append(f"{ctx}: unexpected figure/panel {figure!r}/{panel!r}")
        else:
            if panel in seen[figure]:
                errors.append(f"{ctx}: duplicate {figure} panel {panel}")
            seen[figure].add(panel)
        if stem not in EXPECTED_BUILDERS:
            errors.append(f"{ctx}: unexpected file stem {stem!r}")

        sources = record.get("source_files")
        analyses = record.get("analysis_scripts")
        data_class = record.get("data_class")
        if not isinstance(sources, list) or not isinstance(analyses, list):
            errors.append(f"{ctx}: sources/analysis scripts must be lists")
            continue
        if not sources and data_class != "schematic_no_values":
            errors.append(f"{ctx}: quantitative panel has no source file")
        if data_class == "schematic_no_values" and sources:
            errors.append(f"{ctx}: schematic-only panel unexpectedly has quantitative sources")
        if not isinstance(record.get("independent_or_display_unit"), str):
            errors.append(f"{ctx}: missing unit declaration")

        for kind, items in (("source", sources), ("analysis", analyses)):
            for item_index, item in enumerate(items):
                validate_artifact(item, f"{ctx} {kind}[{item_index}]", errors)
                artifact_checks += 1
        builder = record.get("figure_builder")
        derived = record.get("derived_svg")
        validate_artifact(builder, f"{ctx} builder", errors)
        validate_artifact(derived, f"{ctx} SVG", errors)
        artifact_checks += 2
        if isinstance(builder, dict) and stem in EXPECTED_BUILDERS and builder.get("path") != EXPECTED_BUILDERS[stem]:
            errors.append(f"{ctx}: wrong builder for {stem}")
        expected_svg = f"figures/{stem}.svg"
        if isinstance(derived, dict) and derived.get("path") != expected_svg:
            errors.append(f"{ctx}: wrong SVG for {stem}")
        if isinstance(derived, dict):
            output_hashes.setdefault(str(stem), set()).add(str(derived.get("sha256")))
        if isinstance(builder, dict):
            builder_hashes.setdefault(str(stem), set()).add(str(builder.get("sha256")))

    for figure, panels in EXPECTED.items():
        if seen[figure] != panels:
            errors.append(f"coverage mismatch for {figure}: {sorted(seen[figure])} != {sorted(panels)}")
    for stem, hashes in output_hashes.items():
        if len(hashes) != 1:
            errors.append(f"inconsistent output hash within {stem}")
    for stem, hashes in builder_hashes.items():
        if len(hashes) != 1:
            errors.append(f"inconsistent builder hash within {stem}")

    if errors:
        raise SystemExit("FAIL panel provenance\n" + "\n".join(errors))
    print(f"PASS: {expected_count}/{expected_count} panels, {len(output_hashes)} figures and {artifact_checks} source/script/output hash checks")


if __name__ == "__main__":
    main()
