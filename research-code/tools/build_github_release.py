#!/usr/bin/env python3
"""Build a rights-filtered, size-safe GitHub repository candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT.parent / "Nature_TAC2023_GitHub_Release"
MAX_GIT_BYTES = 20_000_000
TRANSIENT = {".aux", ".log", ".out", ".pyc", ".fls", ".fdb_latexmk"}
ROOT_FILES = [
    "requirements.txt", "requirements-linux-py313.lock", "environment.yml",
    "LICENSE", "CITATION.cff", "DATA_CODE_AVAILABILITY.md",
]
DIRECTORIES = [
    "audit", "baseline_ablation", "baseline_external", "certified_benchmark",
    "boundary_law", "code", "cps_transfer_benchmark", "decision_benchmark", "discovery_extension", "extension_study",
    "external_physical_validation", "figures", "manuscript", "observer_rebuild",
    "observer_in_loop_certified", "parameter_audit", "corpus_audit_round32",
    "physical_e2e_round34", "public_data", "openmct_greybox_round42",
    "openmct_modelset_round46", "public_cluster_cases",
    "theory", "tools",
]
AUDIT_ALLOW = {
    "PUBLIC_RELEASE_RIGHTS_MATRIX.md", "provenance_graph.csv",
    "provenance_graph.json", "math_audit.md", "novelty_matrix.md",
    "UPSTREAM_LICENSE_EVIDENCE.md", "upstream_license_api_snapshot.json",
}

RTHS_DERIVED = {
    "cps_transfer_benchmark/results/SHA256SUMS.json",
    "cps_transfer_benchmark/results/rths_fault_metrics.csv",
    "cps_transfer_benchmark/results/rths_run_metrics.csv",
    "cps_transfer_benchmark/results/summary.json",
}
OPEN_CC_BY_DERIVED_PREFIXES = (
    "cps_transfer_benchmark/results/openmct_",
    "cps_transfer_benchmark/results/water_hil_",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def licence(rel: str) -> str:
    if rel in RTHS_DERIVED:
        return "CC-BY-SA-4.0-upstream-derived-author-confirmation-required"
    if rel.startswith(OPEN_CC_BY_DERIVED_PREFIXES):
        return "CC-BY-4.0-upstream-derived-author-confirmation-required"
    if rel.startswith("external_physical_validation/results/"):
        return "CC-BY-4.0-upstream-derived-author-confirmation-required"
    if rel.endswith((".py", ".sh", "Makefile")) or rel.startswith(".github/"):
        return "MIT-provisional-author-confirmation-required"
    return "CC-BY-4.0-provisional-author-confirmation-required"


def eligible(src: Path) -> tuple[bool, str]:
    rel = src.relative_to(ROOT).as_posix()
    if (not src.is_file() or src.suffix in TRANSIENT or src.name == ".DS_Store"
            or "__pycache__" in src.parts):
        return False, "transient"
    if (rel.startswith("figures/originals_round37/")
            or rel.startswith("extension_study/results_v1_pre_round4/")):
        return False, "superseded-internal-snapshot"
    if (rel.startswith("figures/Fig9_matlab_reproduction.")
            or rel == "figures/build_matlab_reproduction_figure.py"
            or rel == "figures/FIGURE_DATA_MAP.md"):
        return False, "rights-controlled-figure-chain"
    if (rel.startswith("external_physical_validation/raw/")
            or (rel.startswith("public_data/raw/") and src.name != "README.md")
            or (rel.startswith("public_cluster_cases/cache/") and src.name != "README.md")
            or rel.startswith("incoming/")
            or rel.startswith(("matlab_reproduction/",
                               "matlab_branch_reproduction/"))):
        return False, "third-party-or-rights-controlled"
    if rel.startswith("audit/") and src.name not in AUDIT_ALLOW:
        return False, "historical-internal-audit"
    if src.suffix.lower() in {".tif", ".tiff"}:
        return False, "large-release-asset"
    if src.stat().st_size > MAX_GIT_BYTES:
        return False, "large-release-asset"
    return True, "included"


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def filter_release_provenance(stage: Path) -> None:
    """Retain only claim paths whose complete evidence chain is in the release."""
    json_path = stage / "audit" / "provenance_graph.json"
    csv_path = stage / "audit" / "provenance_graph.csv"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    retained: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    for row in payload["rows"]:
        paths = [row["source_object"], row["derived_artifact"],
                 row["raw_identity_object"], *str(row["analysis_script"]).split(";")]
        missing = list(dict.fromkeys(
            rel for rel in paths if not (stage / rel).is_file()
        ))
        if missing:
            excluded.append({
                "claim_id": str(row["claim_id"]),
                "reason": "rights-controlled evidence chain excluded from public candidate",
                "missing_paths": ";".join(missing),
            })
        else:
            retained.append(row)
    payload["rows"] = retained
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    fields = list(retained[0])
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(retained)
    (stage / "audit" / "RELEASE_PROVENANCE_EXCLUSIONS.json").write_text(
        json.dumps({"schema_version": "1.0", "excluded": excluded}, indent=2) + "\n",
        encoding="utf-8",
    )


def filter_panel_provenance(stage: Path) -> None:
    """Remove records for figures whose rights-controlled inputs are excluded."""
    path = stage / "figures" / "PANEL_PROVENANCE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"] = [
        row for row in payload["records"]
        if row.get("file_stem") != "Fig9_matlab_reproduction"
    ]
    payload["record_count"] = len(payload["records"])
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing release candidate: {out}")

    out.mkdir(parents=False)
    stage = out
    exclusions: list[dict[str, object]] = []
    try:
        for rel in ROOT_FILES:
            src = ROOT / rel
            if not src.is_file():
                raise FileNotFoundError(src)
            copy_file(src, stage / rel)
        for directory in DIRECTORIES:
            for src in sorted((ROOT / directory).rglob("*")):
                ok, reason = eligible(src)
                if ok:
                    copy_file(src, stage / src.relative_to(ROOT))
                elif src.is_file() and reason == "large-release-asset":
                    exclusions.append({
                        "path": src.relative_to(ROOT).as_posix(),
                        "bytes": src.stat().st_size,
                        "sha256": sha256(src),
                        "reason": reason,
                    })

        copy_file(ROOT / "release" / "README_GITHUB.md", stage / "README.md")
        copy_file(ROOT / "release" / "DATA_CODE_AVAILABILITY.md",
                  stage / "DATA_CODE_AVAILABILITY.md")
        copy_file(ROOT / "release" / "AUTHOR_INPUT_NEEDED.md", stage / "AUTHOR_INPUT_NEEDED.md")
        copy_file(ROOT / "release" / "THIRD_PARTY_NOTICES.md", stage / "THIRD_PARTY_NOTICES.md")
        copy_file(ROOT / "release" / "AUTHOR_RELEASE_ATTESTATION.md", stage / "AUTHOR_RELEASE_ATTESTATION.md")
        copy_file(ROOT / "release" / "PUBLISHING_CHECKLIST.md", stage / "PUBLISHING_CHECKLIST.md")
        copy_file(ROOT / "release" / "Makefile", stage / "Makefile")
        copy_file(ROOT / "release" / ".gitignore", stage / ".gitignore")
        copy_file(ROOT / "release" / "workflows" / "reproduce.yml",
                  stage / ".github" / "workflows" / "reproduce.yml")

        filter_release_provenance(stage)
        filter_panel_provenance(stage)

        with (stage / "EXCLUDED_ASSETS.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "bytes", "sha256", "reason"])
            writer.writeheader()
            writer.writerows(exclusions)

        files = []
        for path in sorted(stage.rglob("*")):
            if not path.is_file() or path.name == "RELEASE_MANIFEST.json":
                continue
            rel = path.relative_to(stage).as_posix()
            files.append({"path": rel, "bytes": path.stat().st_size,
                          "sha256": sha256(path), "licence": licence(rel)})
        manifest = {
            "schema_version": "1.0",
            "status": "local GitHub release candidate; not a public deposit",
            "public_doi": None,
            "repository_url": None,
            "files": files,
        }
        (stage / "RELEASE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        print(f"incomplete candidate retained for inspection: {out}")
        raise
    print(f"PASS: GitHub candidate built at {out} with {len(files)} files; "
          f"{len(exclusions)} exclusions recorded")


if __name__ == "__main__":
    main()
