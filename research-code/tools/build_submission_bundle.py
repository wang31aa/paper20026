#!/usr/bin/env python3
"""Build the journal-facing manuscript and figure bundle."""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "manuscript/NC_Rebuilt_Main.pdf",
    "manuscript/NC_Rebuilt_SI_Submission.pdf",
    "manuscript/NC_Rebuilt_Extended_Data.pdf",
    "manuscript/SIGuide.doc",
    "manuscript/COVER_LETTER.md",
    "DATA_CODE_AVAILABILITY.md",
]
FILES += [f"figures/Fig{i}_{name}.pdf" for i, name in [
    (1, "dimensionless_boundary"), (10, "cps_transfer"),
    (11, "openmct_motor"),
    (12, "observer_in_loop_certified"),
    (13, "mechanism_scan"),
    (14, "openmct_greybox_bridge"),
    (15, "public_cluster_cases"),
    (16, "discovery_extension"),
]]
FILES += [
    "figures/Fig1_v55_decisive_evidence.pdf",
    "figures/Fig1_v55_decisive_evidence.svg",
    "vehicle_v53_finite_game/V53_FROZEN_CONTRACT.json",
    "vehicle_v53_finite_game/results/V53_REPORT.json",
    "uav_v54_sixdof/UAV_V54_FROZEN_CONTRACT.json",
    "uav_v54_sixdof/results/V54_REPORT.json",
    "uav_v54_sixdof/results/V54_HELDOUT.csv",
    "cross_domain_v55_transfer/V55_FROZEN_CONTRACT.json",
    "cross_domain_v55_transfer/results/V55_REPORT.json",
    "cross_domain_v55_transfer/results/V55_REGRET.csv",
    "cross_domain_v43_response_regimes/results/v43_curve_summary.csv",
    "cross_domain_v43_response_regimes/results/v43_regime_summary.csv",
    "cross_domain_v43_response_regimes/results/V43_RESPONSE_REGIME_AUDIT.json",
    "vehicle_v47_dense/results/V47_DEVELOPMENT.csv",
    "vehicle_v47_dense/results/V47_HELDOUT.csv",
    "vehicle_v47_dense/results/V47_FROZEN_PREDICTIONS.json",
    "vehicle_v47_dense/results/V47_HELDOUT_EVALUATION.json",
    "vehicle_v47_dense/results/V47_VALIDATION.json",
    "cross_domain_v56_frozen_closed_loop/V56_FROZEN_CONTRACT.json",
    "cross_domain_v56_frozen_closed_loop/results/V56_REPORT.json",
    "cross_domain_v56_frozen_closed_loop/results/V56_VALIDATION.json",
    "cross_domain_v56_frozen_closed_loop/results/V56_ALL_POLICY_RUNS.csv",
    "cross_domain_v56_frozen_closed_loop/results/V56_SELECTED_POLICY_EVALUATION.csv",
    "uav_v57_task_filter/UAV_V57_FROZEN_CONTRACT.json",
    "uav_v57_task_filter/results/V57_REPORT.json",
    "uav_v57_task_filter/results/V57_VALIDATION.json",
    "uav_v57_task_filter/results/V57_HELDOUT.csv",
    "uav_v57_task_filter/results/Fig_V57_task_filter.svg",
    "marine_v58_unseen_domain/MARINE_V58_FROZEN_CONTRACT.json",
    "marine_v58_unseen_domain/results/V58_REPORT.json",
    "marine_v58_unseen_domain/results/V58_HELDOUT.csv",
    "marine_v59_dynamic_observer/MARINE_V59_FROZEN_CONTRACT.json",
    "marine_v59_dynamic_observer/results/V59_REPORT.json",
    "marine_v59_dynamic_observer/results/V59_HELDOUT.csv",
    "marine_v60_coordinate_invariant/MARINE_V60_FROZEN_CONTRACT.json",
    "marine_v60_coordinate_invariant/results/V60_REPORT.json",
    "marine_v60_coordinate_invariant/results/V60_HELDOUT.csv",
    "marine_v61_official_mss_qualification/MSS_OTTER_QUALIFICATION.json",
    "marine_v62_mss_runtime_replay/run_official_mss_replay.m",
    "marine_v62_mss_runtime_replay/results/MSS_REPLAY_REPORT.json",
    "marine_v62_mss_runtime_replay/results/MSS_OTTER_DERIVATIVE_WITNESSES.csv",
    "marine_v62_mss_runtime_replay/results/MSS_OTTER_PID_REPLAY.csv",
    "marine_v63_official_heterogeneous_fleet/V63_FROZEN_CONTRACT.json",
    "marine_v63_official_heterogeneous_fleet/run_v63.m",
    "marine_v63_official_heterogeneous_fleet/validate_v63.py",
    "marine_v63_official_heterogeneous_fleet/results/V63_REPORT.json",
    "marine_v63_official_heterogeneous_fleet/results/V63_RUNS.csv",
    "marine_v63_official_heterogeneous_fleet/results/V63_NODE_TIMESERIES.csv",
    "audit/PRINCIPAL_PLATFORM_THEOREM_OBSERVATION_MAP.json",
    "audit/PRINCIPAL_PLATFORM_THEOREM_OBSERVATION_MAP_VALIDATION.json",
    "audit/FULL_MANUSCRIPT_SIMULATION_EVIDENCE_STATUS.json",
    "audit/NATURE_DECISIVE_GATES_ROUND30_2026-09-01.md",
    "audit/CONTINUOUS_VEHICLE_KERNEL_CERTIFICATE.json",
    "audit/CONTINUOUS_VEHICLE_KERNEL_CERTIFICATE_VALIDATION.json",
    "audit/NATURE_DECISIVE_GATES_ROUND31_2026-09-01.md",
    "audit/PROSPECTIVE_INTERVENTION_VALUE_CERTIFICATE.json",
    "audit/PROSPECTIVE_INTERVENTION_VALUE_CERTIFICATE_VALIDATION.json",
    "audit/NATURE_DECISIVE_GATES_ROUND32_2026-09-01.md",
    "vehicle_v46/results/V46_POLICY_SUMMARY.csv",
    "vehicle_v46/results/V46_PAIRED_RESULTS.csv",
    "figures/Fig1_two_network_discovery.pdf",
    "figures/Fig1_two_network_discovery.svg",
    "figures/source_data/Fig1_panel_b_capability_kernels.json",
    "figures/source_data/Fig1_panel_c_uav_frozen_curve.csv",
    "figures/source_data/Fig1_panel_d_out_of_grid_prediction.json",
    "figures/source_data/Fig1_panel_d_system_specific_prediction.json",
    "hil_bridge/results/repeated_timing_status.json",
    "uav_causal_validation/results/uav_v8_curve_summary.csv",
    "uav_causal_validation/results/uav_v8_mechanism_summary.csv",
    "uav_causal_validation/UAV_V12_FROZEN_HELDOUT_CONTRACT.json",
    "uav_causal_validation/results/uav_v12_frozen_heldout_runs.csv",
    "uav_causal_validation/results/UAV_V12_QUALIFICATION_REGISTRY.json",
    "cross_domain_v17/V17_FROZEN_CONTRACT.json",
    "cross_domain_v17/results/v17_analysis.json",
    "cross_domain_v17/results/V17_QUALIFICATION_REGISTRY.json",
    "cross_domain_v18/V18_FROZEN_CONTRACT.json",
    "cross_domain_v18/V18_FROZEN_PREDICTOR.json",
    "cross_domain_v18/results/V18_QUALIFICATION_REGISTRY.json",
    "cross_domain_v19/V19_FROZEN_CONTRACT.json",
    "cross_domain_v19/V19_FROZEN_PREDICTOR.json",
    "cross_domain_v19/results/V19_QUALIFICATION_REGISTRY.json",
    "figures/source_data/SOURCE_DATA_README.md",
    "figures/source_data/Fig1_dimensionless_boundary_rows.csv",
    "figures/source_data/Fig1_scalar_sharpness.csv",
    "figures/source_data/Fig12_observer_in_loop_certified_panel_b_envelope.csv",
    "figures/source_data/Fig12_observer_in_loop_certified_panel_c_observer.csv",
    "figures/source_data/Fig12_observer_in_loop_certified_panel_d_matched.csv",
    "figures/source_data/Fig12_observer_in_loop_certified_panel_e_dt.csv",
    "figures/source_data/Fig13_mechanism_scan_source_data.csv",
    "figures/source_data/Fig13_mechanism_scan_validation.json",
    "figures/source_data/Fig16_directed_network_lower_bound.csv",
    "figures/source_data/Fig16_domain_interventions.csv",
    "figures/source_data/Fig16_summary.json",
    "figures/source_data/Main_text_cluster_safety_budget_scan.csv",
    "figures/source_data/Main_text_public_circuit_summary.csv",
    "figures/source_data/Main_text_public_circuit_record_metrics.csv",
    "figures/source_data/Main_text_public_circuit_topology_metrics.csv",
    "figures/source_data/Extended_Data_Fig1_panel_b_rths_records.csv",
    "figures/source_data/Extended_Data_Fig1_panel_c_replay_records.csv",
    "figures/source_data/Extended_Data_Fig1_panel_d_water_sessions.csv",
    "figures/source_data/Extended_Data_Fig2_panels_a_d_motor_records.csv",
    "figures/source_data/Extended_Data_Fig2_panels_b_c_motor_trace.csv",
    "figures/source_data/Extended_Data_Fig3_motor_holdouts.csv",
    "figures/source_data/Extended_Data_Fig3_motor_identifiability.json",
    "figures/source_data/Extended_Data_Fig3_motor_qualification.json",
    "figures/source_data/Extended_Data_Fig4_summary.csv",
    "figures/source_data/Extended_Data_Fig4_provenance.json",
    "figures/source_data/Extended_Data_Table1_graph_constants.csv",
    "audit/MAIN_TEXT_RESEARCH_NARRATIVE_VALIDATION.json",
    "audit/NATURE_DECISIVE_GATES_ROUND33_2026-09-01.md",
    "audit/CURRENT_NATURE_GATE_STATUS_2026-09-01.json",
    "audit/NATURE_DECISIVE_GATES_ROUND34_2026-09-01.md",
    "observer_in_loop_certified/results_r3/validation_portable_v2.json",
    "audit/NATURE_DECISIVE_GATES_ROUND35_2026-09-01.md",
    "audit/AUTHOR_SIDE_ISOLATED_RELEASE_REPLAY_2026-09-01.json",
    "audit/NATURE_DECISIVE_GATES_ROUND36_2026-09-01.md",
    "audit/OFFICIAL_UAV_REPLAY_ATTRIBUTION_STATUS.json",
    "audit/NATURE_DECISIVE_GATES_ROUND37_2026-09-01.md",
    "audit/CODE_ONLY_PUBLIC_CANDIDATE_STATUS.json",
    "audit/NATURE_DECISIVE_GATES_ROUND38_2026-09-01.md",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "submission_bundle")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    expected = set()
    for rel in FILES:
        src = ROOT / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst_name = "NC_Rebuilt_SI.pdf" if src.name == "NC_Rebuilt_SI_Submission.pdf" else src.name
        dst = out / dst_name
        shutil.copy2(src, dst)
        expected.add(dst.name)
    for path in out.iterdir():
        if path.is_file() and path.name not in expected | {"SHA256SUMS.txt", "README.md"}:
            path.unlink()
    rows = [f"{digest(out / name)}  {name}" for name in sorted(expected)]
    (out / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    total = sum((out / name).stat().st_size for name in expected | {"SHA256SUMS.txt"})
    limit = 150_000_000
    if total >= limit:
        raise RuntimeError(f"bundle is {total} bytes, exceeding the 150 MiB gate")
    (out / "README.md").write_text(
        "# Nature initial-submission upload bundle\n\n"
        "This directory contains the manuscript, Supplementary Information, "
        "Extended Data, cover letter, availability statement and cited vector "
        "figures and Source Data. Code and validation workflows are maintained "
        "in a separate local project archive and are available to editors and "
        "reviewers on request; complete node-level "
        "trajectories are not claimed for every analysis.\n\n"
        f"Verified payload size: {total} bytes ({total / 1_000_000:.2f} MB), below "
        "the journal's decimal 150 MB combined-upload gate.\n",
        encoding="utf-8",
    )
    print(f"PASS: {len(expected)} payload files, {total / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()
