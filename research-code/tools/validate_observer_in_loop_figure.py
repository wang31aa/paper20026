#!/usr/bin/env python3
"""Fail-closed numerical, provenance and export checks for the new R3 figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
RES = ROOT / 'observer_in_loop_certified' / 'results_r3'
SOURCE = FIG / 'source_data'
STEM = 'Fig12_observer_in_loop_certified'
TOL = 2e-15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def max_abs(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, float) - np.asarray(right, float))))


def main() -> int:
    checks = {}; diagnostics = {}
    ledger_path = FIG / f'{STEM}_PANEL_PROVENANCE.json'
    ledger = json.loads(ledger_path.read_text())
    panels = ledger.get('panels', [])
    checks['five_panel_ledger_exact'] = bool(
        ledger.get('schema') == 'cert-oil-r3-panel-provenance-v1'
        and ledger.get('figure') == STEM
        and [panel.get('panel') for panel in panels] == list('abcde'))
    provenance_ok = True; provenance_checks = 0
    for panel in panels:
        for item in panel.get('sources', []) + [panel.get('builder', {})] + panel.get('outputs', []):
            path = ROOT / item.get('path', '')
            provenance_ok &= bool(path.is_file() and path.stat().st_size == item.get('bytes')
                                  and sha256(path) == item.get('sha256'))
            provenance_checks += 1
    checks['all_panel_source_builder_output_hashes_match'] = bool(provenance_ok)
    diagnostics['panel_provenance_hash_checks'] = provenance_checks

    validation = json.loads((RES / 'validation_schemafix.json').read_text())
    checks['upstream_CERT_OIL_R3_validation_passed'] = bool(
        validation.get('passed') and all(validation.get('checks', {}).values()))

    b = pd.read_csv(SOURCE / f'{STEM}_panel_b_envelope.csv')
    c = pd.read_csv(SOURCE / f'{STEM}_panel_c_observer.csv')
    d = pd.read_csv(SOURCE / f'{STEM}_panel_d_matched.csv')
    e = pd.read_csv(SOURCE / f'{STEM}_panel_e_dt.csv')
    checks['source_data_row_counts_exact'] = (len(b) == 120 and len(c) == 120 * 201
        and len(d) == 120 and len(e) == 3 * 8 * 2)
    checks['all_source_data_values_finite'] = all(
        np.all(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()))
        for frame in (b, c, d, e))

    runs = pd.read_csv(RES / 'raw_runs.csv')
    raw = np.load(RES / 'raw_trajectories.npz')
    b_diff = 0.0; c_diff = 0.0
    for row in runs.itertuples(index=False):
        rid = f'{row.topology}_h{row.heterogeneity:.2f}_seed{row.seed}'
        t = raw[f'{rid}__time']; ratio = raw[f'{rid}__tracking_envelope_ratio']
        stored_b = b[b.run_id == rid]
        if len(stored_b) != 1:
            b_diff = float('inf'); break
        b_diff = max(b_diff, abs(float(stored_b.iloc[0].max_positive_time_tracking_envelope_ratio)
                                 - float(ratio[t > 0].max())))
        stored_c = c[c.run_id == rid].sort_values('time')
        if len(stored_c) != len(t):
            c_diff = float('inf'); break
        c_diff = max(c_diff, max_abs(stored_c.time, t),
            max_abs(stored_c.observer_weighted, raw[f'{rid}__observer_weighted']),
            max_abs(stored_c.observer_envelope, raw[f'{rid}__observer_envelope']))
    checks['panel_b_recomputes_from_all_120_raw_trajectories'] = b_diff < TOL
    checks['panel_c_recomputes_from_all_120_raw_trajectories'] = c_diff < TOL
    diagnostics['panel_b_max_recompute_difference'] = b_diff
    diagnostics['panel_c_max_recompute_difference'] = c_diff

    policy = pd.read_csv(RES / 'matched_policy_runs.csv')
    expected_d = policy.pivot(index=['topology', 'heterogeneity', 'seed'],
        columns='policy', values='tail20_max_tracking').reset_index()
    expected_d['observer_to_oracle'] = expected_d.observer_loop / expected_d.oracle_target
    expected_d['no_target_to_oracle'] = expected_d.no_target / expected_d.oracle_target
    d_key = ['topology', 'heterogeneity', 'seed']
    d_compare = d.merge(expected_d, on=d_key, suffixes=('_stored', '_raw'), validate='one_to_one')
    d_diff = max(max_abs(d_compare.observer_to_oracle_stored,
                         d_compare.observer_to_oracle_raw),
                 max_abs(d_compare.no_target_to_oracle_stored,
                         d_compare.no_target_to_oracle_raw))
    checks['panel_d_recomputes_from_360_actual_policy_runs'] = d_diff < TOL
    diagnostics['panel_d_max_recompute_difference'] = d_diff

    dt = pd.read_csv(RES / 'matched_policy_dt_audit.csv')
    expected_e = []
    for key, group in dt.groupby(['policy', 'topology', 'seed'], sort=True):
        values = group.set_index('dt').final_tracking
        base = float(values.loc[.001])
        for step in (.004, .002):
            expected_e.append((*key, step,
                abs(float(values.loc[step]) - base) / max(abs(base), 1e-12)))
    expected_e = pd.DataFrame(expected_e, columns=[
        'policy', 'topology', 'seed', 'dt', 'endpoint_relative_difference_raw'])
    e_compare = e.merge(expected_e, on=['policy', 'topology', 'seed', 'dt'],
                        validate='one_to_one')
    e_diff = max_abs(e_compare.endpoint_relative_difference,
                     e_compare.endpoint_relative_difference_raw)
    checks['panel_e_recomputes_from_all_72_dt_rows'] = e_diff < TOL
    diagnostics['panel_e_max_recompute_difference'] = e_diff

    svg = (FIG / f'{STEM}.svg').read_text()
    checks['svg_has_editable_text_and_no_embedded_raster'] = (
        '<text' in svg and '<image' not in svg)
    checks['all_vector_and_raster_exports_present'] = all(
        (FIG / f'{STEM}.{ext}').is_file() for ext in ('svg', 'pdf', 'png', 'tiff'))
    with Image.open(FIG / f'{STEM}.tiff') as image:
        dpi = image.info.get('dpi', (0, 0)); size = image.size
    checks['tiff_is_at_least_600_dpi'] = min(dpi) >= 599
    diagnostics['tiff_dpi'] = [float(value) for value in dpi]
    diagnostics['tiff_pixels'] = list(size)
    diagnostics['svg_text_nodes'] = svg.count('<text')
    diagnostics['svg_embedded_image_nodes'] = svg.count('<image')

    report = dict(passed=all(checks.values()), checks=checks,
                  diagnostics=diagnostics,
                  output_sha256={ext: sha256(FIG / f'{STEM}.{ext}')
                    for ext in ('svg', 'pdf', 'png', 'tiff')})
    report_path = FIG / f'{STEM}_QA.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
