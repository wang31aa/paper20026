#!/usr/bin/env python3
"""Export the record-level tables plotted in the submission figures."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "cps_transfer_benchmark" / "results"
PHYSICAL_RESULTS = ROOT / "external_physical_validation" / "results"
OUT = ROOT / "figures" / "source_data"

TABLES = {
    "rths_run_metrics.csv": "Extended_Data_Fig1_panel_b_rths_records.csv",
    "rths_fault_metrics.csv": "Extended_Data_Fig1_panel_c_replay_records.csv",
    "water_hil_session_metrics.csv": "Extended_Data_Fig1_panel_d_water_sessions.csv",
    "openmct_run_metrics.csv": "Extended_Data_Fig2_panels_a_d_motor_records.csv",
    "openmct_representative_10ms.csv": "Extended_Data_Fig2_panels_b_c_motor_trace.csv",
}

PHYSICAL_TABLES = {
    "physical_metrics.csv": "Main_text_public_circuit_record_metrics.csv",
    "topology_metrics.csv": "Main_text_public_circuit_topology_metrics.csv",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in TABLES.items():
        source = RESULTS / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, OUT / output_name)
    for source_name, output_name in PHYSICAL_TABLES.items():
        source = PHYSICAL_RESULTS / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, OUT / output_name)
    print(f"Exported {len(TABLES) + len(PHYSICAL_TABLES)} source-data tables to {OUT}")


if __name__ == "__main__":
    main()
