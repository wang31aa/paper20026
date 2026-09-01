#!/usr/bin/env python3
"""Build and validate the v7 claim-to-source provenance graph."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "audit" / "provenance_graph.csv"
OUT_JSON = ROOT / "audit" / "provenance_graph.json"
MANUSCRIPT = ROOT / "manuscript" / "NC_Rebuilt_Main.tex"


ROWS = [
    dict(claim_id="C01", claim="A weighted directed-network metric yields a one-sided ultimate residual certificate.", status="supported-under-assumptions", manuscript_anchor="A weighted metric bounds tracking error on directed graphs", display_item="Main text, Eqs. 1--8", panel="none", derived_artifact="observer_in_loop_certified/results_r3/validation_schemafix.json", source_object="observer_in_loop_certified/PROOF.md", analysis_script="observer_in_loop_certified/test_equations.py", external_identifier="none", boundary="The result is sufficient and asymptotic; it does not predict failure below the threshold or bound every finite-time sample."),
    dict(claim_id="C02", claim="The observer, controller and certificate close numerically on one dissipative comparison system.", status="supported-simulation-observer-in-loop", manuscript_anchor="We evaluated the complete observer--controller model", display_item="Main Figure 2", panel="a--e", derived_artifact="figures/Fig12_observer_in_loop_certified.svg", source_object="observer_in_loop_certified/results_r3/raw_trajectories.npz", analysis_script="observer_in_loop_certified/run_benchmark.py;observer_in_loop_certified/validate_r3_schemafix.py;figures/build_observer_in_loop_certified_figure.py", external_identifier="none", boundary="Fixed simulated graphs and a protocol-known target field; no physical controller deployment is claimed."),
    dict(claim_id="C03", claim="A 576-trajectory mechanism scan separates topology, forcing and observer-gain effects.", status="supported-simulation-mechanism-scan", manuscript_anchor="The parameter scan resolved the predicted division of roles", display_item="Main Figure 3", panel="a--d", derived_artifact="figures/Fig13_mechanism_scan.svg", source_object="observer_in_loop_certified/results_r4/raw_runs.csv", analysis_script="observer_in_loop_certified/run_mechanism_scan.py;observer_in_loop_certified/validate_r4.py;figures/build_r4_mechanism.py", external_identifier="none", boundary="The scan tests the stated simulated model class and is not a cross-domain physical validation."),
    dict(claim_id="C04", claim="The dimensionless threshold organizes 2,304 paired tolerance evaluations from 576 trajectories.", status="supported-one-sided-empirical-consistency", manuscript_anchor="A dimensionless number defines a one-sided certificate threshold", display_item="Main Figure 1", panel="a,b,d", derived_artifact="figures/Fig1_dimensionless_boundary.svg", source_object="boundary_law/results/dimensionless_boundary_rows.csv", analysis_script="boundary_law/derive_boundary_law.py;boundary_law/validate.py", external_identifier="none", boundary="The finite-tail absence of exceedances is empirical consistency, not a theorem-excluded finite-time quadrant; the 2,304 rows are paired evaluations, not independent trajectories."),
    dict(claim_id="C05", claim="The coefficient two is attained exactly in a scalar member of the theorem class.", status="supported-analytic-sharpness", manuscript_anchor="The numerical conservatism does not mean", display_item="Main Figure 1", panel="c", derived_artifact="boundary_law/results/scalar_sharpness.csv", source_object="boundary_law/results/scalar_sharpness.csv", analysis_script="boundary_law/derive_boundary_law.py;boundary_law/validate.py", external_identifier="none", boundary="Sharpness is for the uniform coefficient within the theorem class, not an information-theoretic lower bound over all controllers."),
    dict(claim_id="C06", claim="A measurement-side residual orders independently published coupled-circuit traces.", status="supported-measurement-side-only", manuscript_anchor="In 303 recordings", display_item="Main text and Source Data", panel="none", derived_artifact="external_physical_validation/results/physical_metrics.csv", source_object="external_physical_validation/results/validation.json", analysis_script="external_physical_validation/analyse_physical_timeseries.py;external_physical_validation/validate_results.py", external_identifier="doi:10.5281/zenodo.3521009", boundary="No paper observer, controller or hardware-in-the-loop deployment is tested."),
    dict(claim_id="C07", claim="Public motor records constrain a grey-box dissipative application model.", status="supported-offline-model-bridge", manuscript_anchor="We next used the public motor measurements", display_item="Extended Data Figure 3", panel="a--d", derived_artifact="figures/Fig14_openmct_greybox_bridge.svg", source_object="openmct_greybox_round42/results/holdout_metrics.csv", analysis_script="openmct_greybox_round42/run_qualification.py;openmct_greybox_round42/validate.py;figures/build_openmct_greybox_bridge.py", external_identifier="doi:10.17632/5xvg43r9r8.1", boundary="Offline data-constrained modelling; not a distributed multi-motor observer-controller experiment."),
    dict(claim_id="C08", claim="Structural and water-controller records test transportability of residual modelling.", status="supported-mixed-transfer", manuscript_anchor="Structural real-time hybrid-test and water-controller", display_item="Extended Data Figure 1", panel="b--d", derived_artifact="figures/Fig10_cps_transfer.svg", source_object="cps_transfer_benchmark/results/summary.json", analysis_script="cps_transfer_benchmark/run_rths.py;cps_transfer_benchmark/run_water_hil.py;cps_transfer_benchmark/validate_results.py;figures/build_cps_transfer_figure.py", external_identifier="doi:10.5281/zenodo.17296336;doi:10.6084/m9.figshare.28735547.v2", boundary="The records constrain candidate residual models; they do not execute the distributed observer-controller."),
    dict(claim_id="C09", claim="Three public-data-informed mobile-system cases illustrate the disturbance-ledger interpretation.", status="supported-simulation-application-cases", manuscript_anchor="We next asked whether", display_item="Extended Data Figure 4", panel="a--c", derived_artifact="figures/Fig15_public_cluster_cases.svg", source_object="public_cluster_cases/results/summary.csv", analysis_script="public_cluster_cases/run_cases.py;public_cluster_cases/validate.py", external_identifier="DaRUS robot validation data; Hugging Face UAV synthetic data; US DOT naturalistic driving data", boundary="These are public-data-informed simulations with normalized radii, not physical-unit end-to-end validations or a universal cross-domain threshold."),
    dict(claim_id="C10", claim="Nontrivial five-node directed members occupy most of the uniform certificate and domain dynamics determine intervention effects.", status="supported-analytic-and-computational-extension", manuscript_anchor="Directed-network lower bounds and domain-specific interventions", display_item="Main Figure 4", panel="a--d", derived_artifact="figures/Fig16_discovery_extension.svg", source_object="discovery_extension/results/domain_interventions.csv", analysis_script="discovery_extension/run_extension.py;discovery_extension/validate.py", external_identifier="DaRUS robot validation data; Hugging Face UAV synthetic data; US DOT naturalistic driving data", boundary="The directed lower bound is analytic; the domain results are public-data-constrained surrogates with calibration proxies, not physical controller trials or safety probabilities."),
]

RAW_IDENTITY = {
    "C06": "external_physical_validation/RAW_REBUILD_RECORD.md",
    "C07": "openmct_greybox_round42/PROTOCOL.json",
    "C08": "cps_transfer_benchmark/PREREGISTRATION.md",
    "C09": "public_cluster_cases/results/provenance.json",
    "C10": "public_cluster_cases/results/provenance.json",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def licence(rel: str) -> str:
    if rel.startswith("incoming/"):
        return "author-redistribution-authorized-no-standard-licence"
    if rel.endswith((".py", ".sh")):
        return "MIT-provisional"
    return "CC-BY-4.0-provisional"


def build() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for spec in ROWS:
        if spec["claim_id"] in seen:
            raise ValueError(f"duplicate claim id: {spec['claim_id']}")
        seen.add(spec["claim_id"])
        manuscript_path = ROOT / spec.get("manuscript_file", "manuscript/NC_Rebuilt_Main.tex")
        manuscript = manuscript_path.read_text(encoding="utf-8")
        if spec["manuscript_anchor"] not in manuscript:
            raise ValueError(f"missing manuscript anchor: {spec['manuscript_anchor']}")
        source = ROOT / spec["source_object"]
        derived = ROOT / spec["derived_artifact"]
        raw_identity_rel = RAW_IDENTITY.get(spec["claim_id"], spec["source_object"])
        raw_identity = ROOT / raw_identity_rel
        if not source.is_file() or not derived.is_file() or not raw_identity.is_file():
            raise FileNotFoundError(source if not source.is_file() else derived)
        scripts = spec["analysis_script"].split(";")
        for script in scripts:
            if not (ROOT / script).is_file():
                raise FileNotFoundError(ROOT / script)
        row = dict(spec)
        row.setdefault("manuscript_file", "manuscript/NC_Rebuilt_Main.tex")
        row.update(
            source_bytes=source.stat().st_size,
            source_sha256=sha256(source),
            source_licence=licence(spec["source_object"]),
            raw_identity_object=raw_identity_rel,
            raw_identity_sha256=sha256(raw_identity),
            derived_sha256=sha256(derived),
        )
        out.append(row)
    return out


def main() -> None:
    rows = build()
    fields = list(rows[0])
    with OUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(
        json.dumps({"schema_version": "1.0", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: {len(rows)} claim-to-source paths written and validated")


if __name__ == "__main__":
    main()
