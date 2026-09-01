#!/usr/bin/env python3
"""Build a deterministic, panel-resolved source/script/output hash ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "PANEL_PROVENANCE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(relative: str) -> dict[str, object]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


# Each tuple is: panel, source paths, analysis scripts, data class,
# independent/display unit. Figure-builder paths are specified once per figure.
FIGURES: list[dict[str, object]] = [
    {
        "figure": "Main Figure 1", "stem": "Fig1_dimensionless_boundary",
        "builder": "boundary_law/derive_boundary_law.py",
        "panels": [
            ("a", ["boundary_law/results/dimensionless_boundary_rows.csv"],
             ["boundary_law/derive_boundary_law.py", "boundary_law/validate.py"],
             "paired_simulation_evaluations", "four tolerance evaluations per trajectory; 2,304 rows from 576 trajectories"),
            ("b", ["boundary_law/results/dimensionless_boundary_rows.csv"],
             ["boundary_law/derive_boundary_law.py", "boundary_law/validate.py"],
             "synthetic_topology_scan", "independent initialized trajectory; n=144 per topology"),
            ("c", ["boundary_law/results/scalar_sharpness.csv"],
             ["boundary_law/derive_boundary_law.py", "boundary_law/validate.py"],
             "analytic_sharpness_construction", "deterministic scalar theorem-class construction; 120 parameter points"),
            ("d", ["boundary_law/results/dimensionless_boundary_rows.csv"],
             ["boundary_law/derive_boundary_law.py", "boundary_law/validate.py"],
             "synthetic_parameter_scan", "independent initialized trajectory; n=576"),
        ],
    },
    {
        "figure": "Extended Data Figure 1", "stem": "Fig10_cps_transfer",
        "builder": "figures/build_cps_transfer_figure.py",
        "panels": [
            ("a", [], [], "schematic_no_values", "not applicable"),
            ("b", ["cps_transfer_benchmark/results/rths_run_metrics.csv"],
             ["cps_transfer_benchmark/run_rths.py"], "external_physical_hil", "held-out actuator record nested in run; n=14"),
            ("c", ["cps_transfer_benchmark/results/rths_fault_metrics.csv"],
             ["cps_transfer_benchmark/run_rths.py"], "synthetic_replay_of_external_hil", "synthetic actuator-record replay; n=224 total"),
            ("d", ["cps_transfer_benchmark/results/water_hil_session_metrics.csv"],
             ["cps_transfer_benchmark/run_water_hil.py"], "external_hil_weak_label", "complete spoof-labelled session; n=1 per condition"),
        ],
    },
    {
        "figure": "Extended Data Figure 2", "stem": "Fig11_openmct_motor",
        "builder": "figures/build_openmct_figure.py",
        "panels": [
            ("a", ["cps_transfer_benchmark/results/openmct_run_metrics.csv"],
             ["cps_transfer_benchmark/run_openmct.py"], "external_physical", "complete held-out controller record; n=7"),
            ("b", ["cps_transfer_benchmark/results/openmct_representative_10ms.csv"],
             ["cps_transfer_benchmark/run_openmct.py"], "external_physical", "prespecified complete 10-ms PI record"),
            ("c", ["cps_transfer_benchmark/results/openmct_representative_10ms.csv"],
             ["cps_transfer_benchmark/run_openmct.py"], "external_physical", "prespecified complete 10-ms PI record"),
            ("d", ["cps_transfer_benchmark/results/openmct_run_metrics.csv"],
             ["cps_transfer_benchmark/run_openmct.py"], "external_physical", "complete continuous-PI record; n=4"),
        ],
    },
    {
        "figure": "Extended Data Figure 7", "stem": "Fig6_ablation",
        "builder": "figures/build_figures.py",
        "panels": [
            ("a", ["baseline_ablation/results/raw_runs.csv", "baseline_ablation/results/metadata.json"],
             ["baseline_ablation/run_baselines.py"], "synthetic_simulation", "independent initialized run; n=20 per cell"),
            ("b", ["baseline_ablation/results/raw_runs.csv"],
             ["baseline_ablation/run_baselines.py"], "synthetic_simulation", "independent initialized run; n=60 per variant"),
            ("c", ["baseline_ablation/results/raw_runs.csv", "baseline_ablation/results/metadata.json"],
             ["baseline_ablation/run_baselines.py"], "synthetic_simulation", "independent initialized run; n=20 per cell"),
        ],
    },
    {
        "figure": "Extended Data Figure 1", "stem": "Fig2_robustness",
        "builder": "figures/build_figures.py",
        "panels": [
            ("a", ["code/results/raw_runs.csv"], ["code/run_experiments.py"],
             "synthetic_simulation", "independent initialized run; n=20 per cell"),
            ("b", ["code/results/raw_runs.csv"], ["code/run_experiments.py"],
             "synthetic_simulation", "independent initialized run; n=20 per cell"),
            ("c", ["code/results/raw_runs.csv"], ["code/run_experiments.py"],
             "synthetic_simulation", "independent initialized run; n=100 per heterogeneity group"),
        ],
    },
    {
        "figure": "Extended Data Figure 2", "stem": "Fig3_numerics_certificates",
        "builder": "figures/build_figures.py",
        "panels": [
            ("a", ["code/results/dt_convergence.csv"], ["code/run_experiments.py"],
             "synthetic_numerical_audit", "matched initial-condition seed; n=5"),
            ("b", ["code/results/raw_runs.csv"], ["code/run_experiments.py"],
             "synthetic_simulation", "independent initialized run; n=20 per cell"),
            ("c", ["theory/result.json"], ["theory/solve_static.py"],
             "synthetic_analytic", "deterministic solver/theory output"),
        ],
    },
    {
        "figure": "Extended Data Figure 3", "stem": "Fig4_oracle_diagnostic",
        "builder": "figures/build_figures.py",
        "panels": [
            ("a", ["code/results/raw_runs.csv", "code/results/raw_representative_trajectories.npz"],
             ["code/run_experiments.py"], "synthetic_simulation", "prespecified representative run, seed 0"),
            ("b", ["code/results/raw_representative_trajectories.npz"],
             ["code/run_experiments.py"], "synthetic_simulation", "prespecified representative run, seed 0"),
            ("c", ["code/results/raw_runs.csv", "code/results/raw_representative_trajectories.npz"],
             ["code/run_experiments.py"], "synthetic_simulation", "declared final 20% of representative run"),
            ("d", ["code/results/topologies.npz"], ["code/run_experiments.py"],
             "synthetic_topology", "exact stored directed topology"),
        ],
    },
    {
        "figure": "Extended Data Figure 4", "stem": "Fig9_matlab_reproduction",
        "builder": "figures/build_matlab_reproduction_figure.py",
        "panels": [
            ("a", ["matlab_reproduction/PREREGISTRATION.md", "matlab_reproduction/results/graph1_full/raw_checkpoints.csv", "matlab_reproduction/results/graph1_full/parity_summary.json"],
             ["matlab_reproduction/literal_static_graph.py"], "author_supplied_computation_audit", "frozen variable group; n=37"),
            ("b", ["matlab_reproduction/results/graph2_heldout/pointwise_errors.csv", "matlab_reproduction/results/graph2_heldout/parity_summary.json", "matlab_reproduction/posthoc_correction/graph2_full/pointwise_errors.csv", "matlab_reproduction/posthoc_correction/graph2_full/parity_summary.json"],
             ["matlab_reproduction/literal_static_graph.py", "matlab_reproduction/posthoc_correction/run_graph2_ab3_ma4.py"],
             "author_supplied_computation_audit", "frozen variable group; n=37 per branch"),
            ("c", ["matlab_reproduction/phase2/graph1_summary/legacy_ab.csv", "matlab_reproduction/phase2/graph2_summary/legacy_ab.csv"],
             ["matlab_reproduction/intended_continuous.py", "matlab_reproduction/summarize_phase2.py"],
             "author_supplied_computation_audit", "deterministic formulation comparison"),
            ("d", ["matlab_reproduction/phase2/graph1_summary/run_metrics.csv", "matlab_reproduction/phase2/graph2_summary/run_metrics.csv"],
             ["matlab_reproduction/intended_continuous.py", "matlab_reproduction/summarize_phase2.py"],
             "author_supplied_computation_audit", "three registered grids per graph"),
        ],
    },
    {
        "figure": "Extended Data Figure 5", "stem": "Fig7_extension",
        "builder": "figures/build_figures.py",
        "panels": [
            ("a", ["extension_study/results/raw_runs.csv", "extension_study/results/metadata.json"],
             ["extension_study/run_extension.py"], "synthetic_stress_test", "independent initialized run; n=10 per cell"),
            ("b", ["extension_study/results/raw_runs.csv"],
             ["extension_study/run_extension.py"], "synthetic_stress_test", "independent initialized run; n=10 per cell"),
            ("c", ["extension_study/results/raw_runs.csv"],
             ["extension_study/run_extension.py"], "synthetic_stress_test", "independent initialized run; n=10 per cell"),
        ],
    },
    {
        "figure": "Main Figure 2", "stem": "Fig12_observer_in_loop_certified",
        "builder": "figures/build_observer_in_loop_certified_figure.py",
        "panels": [
            ("a", ["observer_in_loop_certified/PREREGISTRATION_R3.md",
                   "observer_in_loop_certified/PROOF.md",
                   "observer_in_loop_certified/results_r3/metadata.json"],
             ["observer_in_loop_certified/run_benchmark.py"],
             "analytic_certificate", "analytic identity; no empirical unit"),
            ("b", ["observer_in_loop_certified/results_r3/raw_trajectories.npz"],
             ["observer_in_loop_certified/validate_r3_schemafix.py"],
             "synthetic_simulation", "independent initialized observer-loop run; n=120"),
            ("c", ["observer_in_loop_certified/results_r3/raw_trajectories.npz"],
             ["observer_in_loop_certified/validate_r3_schemafix.py"],
             "synthetic_simulation", "independent initialized observer-loop run; n=120"),
            ("d", ["observer_in_loop_certified/results_r3/matched_policy_runs.csv"],
             ["observer_in_loop_certified/run_benchmark.py",
              "observer_in_loop_certified/validate_r3_schemafix.py"],
             "synthetic_matched_control", "matched topology-level-seed triplet; n=120"),
            ("e", ["observer_in_loop_certified/results_r3/matched_policy_dt_audit.csv"],
             ["observer_in_loop_certified/run_benchmark.py",
              "observer_in_loop_certified/validate_r3_schemafix.py"],
             "synthetic_step_audit", "policy-topology-seed-step arm; n=72"),
        ],
    },
    {
        "figure": "Main Figure 3", "stem": "Fig13_mechanism_scan",
        "builder": "figures/build_r4_mechanism.py",
        "panels": [
            ("a", ["observer_in_loop_certified/results_r4/raw_runs.csv",
                   "observer_in_loop_certified/PREREGISTRATION_R4.md"],
             ["observer_in_loop_certified/run_mechanism_scan.py"],
             "synthetic_parameter_scan", "complete prescribed run; n=576"),
            ("b", ["observer_in_loop_certified/results_r4/raw_runs.csv"],
             ["observer_in_loop_certified/run_mechanism_scan.py"],
             "synthetic_parameter_scan", "complete prescribed run; n=576"),
            ("c", ["observer_in_loop_certified/results_r4/raw_runs.csv"],
             ["observer_in_loop_certified/run_mechanism_scan.py"],
             "synthetic_parameter_scan", "complete prescribed run; n=576"),
            ("d", ["observer_in_loop_certified/results_r4/raw_runs.csv",
                   "observer_in_loop_certified/results_r4/validation.json"],
             ["observer_in_loop_certified/run_mechanism_scan.py"],
             "synthetic_parameter_scan", "complete prescribed run; n=576"),
        ],
    },
    {
        "figure": "Extended Data Figure 3", "stem": "Fig14_openmct_greybox_bridge",
        "builder": "figures/build_openmct_greybox_bridge.py",
        "panels": [
            ("a", ["openmct_greybox_round42/PROTOCOL.json"],
             ["openmct_greybox_round42/run_qualification.py"],
             "model_design_schematic", "three training and seven held-out physical records"),
            ("b", ["openmct_greybox_round42/results/holdout_metrics.csv"],
             ["openmct_greybox_round42/run_qualification.py"],
             "external_physical_model_holdout", "complete held-out motor record; n=7"),
            ("c", ["openmct_greybox_round42/results/identifiability.json"],
             ["openmct_greybox_round42/run_qualification.py"],
             "external_physical_model_identification", "pooled fit plus three leave-one-record fits"),
            ("d", ["openmct_greybox_round42/results/holdout_metrics.csv"],
             ["openmct_greybox_round42/run_qualification.py"],
             "external_physical_model_holdout", "complete held-out motor record; n=7"),
        ],
    },
    {
        "figure": "Extended Data Figure 4", "stem": "Fig15_public_cluster_cases",
        "builder": "public_cluster_cases/run_cases.py",
        "panels": [
            ("a", ["public_cluster_cases/results/robot_trace.csv",
                   "public_cluster_cases/results/provenance.json"],
             ["public_cluster_cases/run_cases.py", "public_cluster_cases/validate.py"],
             "public_simulation_informed_cluster_case",
             "complete DaRUS validation run 0; five robot roles and 295 rows"),
            ("b", ["public_cluster_cases/results/uav_trace.csv",
                   "public_cluster_cases/results/provenance.json"],
             ["public_cluster_cases/run_cases.py", "public_cluster_cases/validate.py"],
             "public_synthetic_uav_informed_cluster_case",
             "20 UAVs and 20,000 public synthetic rows"),
            ("c", ["public_cluster_cases/results/vehicle_trace.csv",
                   "public_cluster_cases/results/provenance.json"],
             ["public_cluster_cases/run_cases.py", "public_cluster_cases/validate.py"],
             "public_naturalistic_vehicle_informed_cluster_case",
             "ADAS record 6; three vehicle roles and 1,719 rows"),
        ],
    },
    {
        "figure": "Main Figure 4", "stem": "Fig16_discovery_extension",
        "builder": "discovery_extension/run_extension.py",
        "panels": [
            ("a", ["discovery_extension/results/directed_network_lower_bound.csv"],
             ["discovery_extension/run_extension.py", "discovery_extension/validate.py"],
             "analytic_directed_network_lower_bound",
             "graph-gain combination; 3 graph classes x 80 gains"),
            ("b", ["discovery_extension/results/domain_interventions.csv",
                   "public_cluster_cases/results/provenance.json"],
             ["discovery_extension/run_extension.py", "discovery_extension/validate.py"],
             "public_data_constrained_domain_surrogate",
             "one held-out sequence segment per domain"),
            ("c", ["discovery_extension/results/domain_interventions.csv"],
             ["discovery_extension/run_extension.py", "discovery_extension/validate.py"],
             "single_factor_computational_intervention",
             "domain-intervention run; 3 domains x 5 conditions"),
            ("d", ["discovery_extension/results/domain_interventions.csv"],
             ["discovery_extension/run_extension.py", "discovery_extension/validate.py"],
             "single_factor_computational_intervention",
             "domain-intervention run; 3 domains x 4 contrasts"),
        ],
    },
]

# Journal-facing provenance contains only figures cited by the current
# manuscript. Historical diagnostic figures remain reproducible in the
# repository but are not represented as submission displays.
CURRENT_STEMS = {
    "Fig1_dimensionless_boundary", "Fig12_observer_in_loop_certified",
    "Fig13_mechanism_scan", "Fig10_cps_transfer", "Fig11_openmct_motor",
    "Fig14_openmct_greybox_bridge", "Fig15_public_cluster_cases",
    "Fig16_discovery_extension",
}
FIGURES = [figure for figure in FIGURES if figure["stem"] in CURRENT_STEMS]


def main() -> None:
    records: list[dict[str, object]] = []
    for figure in FIGURES:
        builder = artifact(str(figure["builder"]))
        svg = artifact(f"figures/{figure['stem']}.svg")
        for panel, sources, analyses, data_class, unit in figure["panels"]:  # type: ignore[index]
            records.append({
                "display_id": f"{figure['figure']} panel {panel}",
                "figure": figure["figure"],
                "panel": panel,
                "file_stem": figure["stem"],
                "data_class": data_class,
                "independent_or_display_unit": unit,
                "source_files": [artifact(path) for path in sources],
                "analysis_scripts": [artifact(path) for path in analyses],
                "figure_builder": builder,
                "derived_svg": svg,
            })
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "record_count": len(records),
        "records": records,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE: {len(records)} panel provenance records to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
