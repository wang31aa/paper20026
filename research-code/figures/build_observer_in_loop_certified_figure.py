#!/usr/bin/env python3
"""Build a new, standalone R3 observer-in-loop evidence figure.

All quantitative marks are derived from immutable CERT-OIL-R3 raw outputs.
The script never imports or executes the simulator.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd

# Mandatory editable-text and publication settings.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
mpl.rcParams.update({
    'font.size': 7.0, 'axes.titlesize': 7.5, 'axes.labelsize': 7.0,
    'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0,
    'legend.fontsize': 7.0, 'axes.linewidth': 0.75,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.hashsalt': 'cert-oil-r3-figure-v1',
    'savefig.dpi': 600, 'savefig.bbox': 'tight',
})

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'observer_in_loop_certified' / 'results_r3'
OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'source_data'
STEM = 'Fig12_observer_in_loop_certified'

BLUE = '#0F4D92'
BLUE2 = '#3775BA'
TEAL = '#42949E'
ORANGE = '#D97732'
GREY = '#767676'
DARK = '#272727'
LIGHT = '#D8DDE4'
PALE_BLUE = '#E6EEF7'
PALE_TEAL = '#E4F1F1'
PALE_ORANGE = '#F5E8DE'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(ax, grid: bool = True) -> None:
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis='y', color='#D9DDE2', lw=.45, alpha=.65)
    ax.set_axisbelow(True)


def panel_label(ax, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=8.5,
            fontweight='bold', ha='left', va='bottom')


def stable_offsets(n: int, width: float = .18) -> np.ndarray:
    """Deterministic display-only offsets; source values are unchanged."""
    base = np.linspace(-width, width, n)
    order = np.argsort((np.arange(n) * 37) % max(n, 1))
    return base[order]


def box(ax, xy, width, height, text, face, edge=GREY, fontsize=7.0):
    patch = FancyBboxPatch(xy, width, height, boxstyle='round,pad=0.018',
        transform=ax.transAxes, facecolor=face, edgecolor=edge, lw=.8)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            transform=ax.transAxes, ha='center', va='center', fontsize=fontsize,
            color=DARK, linespacing=1.25)
    return patch


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
        arrowstyle='-|>', mutation_scale=8, lw=.8, color=color,
        shrinkA=2, shrinkB=2))


def load_and_derive():
    required = [
        RES / 'raw_runs.csv', RES / 'raw_trajectories.npz',
        RES / 'matched_policy_runs.csv', RES / 'matched_policy_trajectories.npz',
        RES / 'matched_policy_dt_audit.csv', RES / 'metadata.json',
        RES / 'validation_schemafix.json',
    ]
    if not all(path.is_file() for path in required):
        raise FileNotFoundError('complete validated CERT-OIL-R3 inputs are required')
    validation = json.loads((RES / 'validation_schemafix.json').read_text())
    if not validation.get('passed') or not all(validation.get('checks', {}).values()):
        raise ValueError('CERT-OIL-R3 validation gate is not fully passed')
    runs = pd.read_csv(RES / 'raw_runs.csv')
    policies = pd.read_csv(RES / 'matched_policy_runs.csv')
    dt = pd.read_csv(RES / 'matched_policy_dt_audit.csv')
    trajectories = np.load(RES / 'raw_trajectories.npz')
    if len(runs) != 120 or len(policies) != 360 or len(dt) != 72:
        raise ValueError('unexpected R3 result dimensions')
    if not runs['finite'].all() or not policies['finite'].all() or not dt['finite'].all():
        raise ValueError('non-finite registered run retained; figure build refused')

    panel_b = []
    panel_c = []
    for row in runs.itertuples(index=False):
        rid = f'{row.topology}_h{row.heterogeneity:.2f}_seed{row.seed}'
        time = trajectories[f'{rid}__time']
        ratio = trajectories[f'{rid}__tracking_envelope_ratio']
        observer = trajectories[f'{rid}__observer_weighted']
        observer_envelope = trajectories[f'{rid}__observer_envelope']
        if not (np.all(np.isfinite(time)) and np.all(np.isfinite(ratio))
                and np.all(np.isfinite(observer)) and np.all(np.isfinite(observer_envelope))):
            raise ValueError(f'non-finite raw trajectory: {rid}')
        panel_b.append(dict(run_id=rid, topology=row.topology,
            heterogeneity=row.heterogeneity, seed=row.seed,
            max_positive_time_tracking_envelope_ratio=float(ratio[time > 0].max()),
            tail20_delta_ratio=row.tail20_delta_ratio))
        panel_c.extend(dict(run_id=rid, topology=row.topology,
            heterogeneity=row.heterogeneity, seed=row.seed, time=float(t),
            observer_weighted=float(h), observer_envelope=float(he))
            for t, h, he in zip(time, observer, observer_envelope))
    panel_b = pd.DataFrame(panel_b)
    panel_c = pd.DataFrame(panel_c)

    pivot = policies.pivot(index=['topology', 'heterogeneity', 'seed'],
        columns='policy', values='tail20_max_tracking').reset_index()
    pivot['observer_to_oracle'] = pivot['observer_loop'] / pivot['oracle_target']
    pivot['no_target_to_oracle'] = pivot['no_target'] / pivot['oracle_target']
    panel_d = pivot

    dt_key = ['policy', 'topology', 'seed']
    dt_wide = dt.pivot(index=dt_key, columns='dt', values='final_tracking').reset_index()
    panel_e = []
    for row in dt_wide.itertuples(index=False):
        values = row._asdict()
        # Pandas sanitizes numeric column names to positional fields; recover by
        # selecting the original grouped rows instead of depending on them.
        group = dt[(dt.policy == row.policy) & (dt.topology == row.topology)
                   & (dt.seed == row.seed)].set_index('dt').final_tracking
        base = float(group.loc[.001])
        for step in (.004, .002):
            panel_e.append(dict(policy=row.policy, topology=row.topology,
                seed=row.seed, dt=step,
                endpoint_relative_difference=abs(float(group.loc[step]) - base)
                    / max(abs(base), 1e-12)))
    panel_e = pd.DataFrame(panel_e)

    SOURCE.mkdir(parents=True, exist_ok=True)
    panel_b.to_csv(SOURCE / f'{STEM}_panel_b_envelope.csv', index=False,
                   float_format='%.17g')
    panel_c.to_csv(SOURCE / f'{STEM}_panel_c_observer.csv', index=False,
                   float_format='%.17g')
    panel_d.to_csv(SOURCE / f'{STEM}_panel_d_matched.csv', index=False,
                   float_format='%.17g')
    panel_e.to_csv(SOURCE / f'{STEM}_panel_e_dt.csv', index=False,
                   float_format='%.17g')
    return runs, policies, panel_b, panel_c, panel_d, panel_e, trajectories


def draw_panel_a(ax):
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    stages = [
        (.025, 'Target dynamics', '$s(t)$', PALE_BLUE, BLUE),
        (.225, 'Distributed estimate', '$\\hat{s}_i$', PALE_TEAL, TEAL),
        (.425, 'Local pinning', '$B_{pin}\\hat{s}$', PALE_BLUE, BLUE),
        (.625, 'Observer perturbation', '$+\\alpha B_{pin}\\eta$', PALE_ORANGE, ORANGE),
        (.825, 'Certified envelope', '$z\\leq E_0+E_w+E_\\eta$', '#F0EEF7', '#6B5B95'),
    ]
    width, height, y = .15, .54, .20
    for index, (x, title, formula, face, edge) in enumerate(stages):
        box(ax, (x, y), width, height, f'{title}\n{formula}', face, edge,
            fontsize=7.0)
        if index < len(stages) - 1:
            arrow(ax, (x + width + .008, y + height / 2),
                  (stages[index + 1][0] - .008, y + height / 2), edge)
    ax.text(.025, .94, 'Observer information enters the tracking certificate',
            transform=ax.transAxes, ha='left', va='top', fontsize=7.8,
            fontweight='bold')


def draw_panel_b(ax, panel_b):
    topo_order = ['chain', 'star', 'branch', 'cyclic']
    levels = [.05, .15, .30]
    cells = [(topology, level) for topology in topo_order for level in levels]
    for group in range(4):
        if group % 2 == 0:
            ax.axvspan(group * 3 - .48, group * 3 + 2.48,
                       color='#F5F7F9', zorder=0)
    for index, (topology, level) in enumerate(cells):
        values = panel_b[(panel_b.topology == topology)
                         & np.isclose(panel_b.heterogeneity, level)][
                             'max_positive_time_tracking_envelope_ratio'].to_numpy()
        if len(values) != 10: raise ValueError('panel b cell is not n=10')
        ax.scatter(index + stable_offsets(len(values), .16), values,
            s=9, color=BLUE2, alpha=.72, edgecolor='white', linewidth=.25, zorder=3)
        ax.plot([index - .20, index + .20], [np.median(values)] * 2,
                color=DARK, lw=1.2, zorder=4)
    ax.axhline(1, color=ORANGE, ls='--', lw=1, label='coverage threshold')
    ax.set_ylim(.30, 1.035); ax.set_xlim(-.6, 11.6)
    ax.set_ylabel('Maximum positive-time\nenvelope ratio')
    ax.set_xticks([1, 4, 7, 10], ['Chain', 'Star', 'Branch', 'Cyclic'])
    ax.tick_params(axis='x', pad=2)
    ax.set_title('All 120 runs remain below the certificate', pad=4)
    # Line roles are stated in the caption; a legend would cover the complete
    # trajectory bundle at final manuscript width.
    clean(ax)


def draw_panel_c(ax, panel_c):
    rids = panel_c.run_id.drop_duplicates().tolist()
    matrix = panel_c.pivot(index='run_id', columns='time',
                           values='observer_weighted').loc[rids]
    envelope = panel_c.pivot(index='run_id', columns='time',
                             values='observer_envelope').loc[rids]
    time = matrix.columns.to_numpy(float); values = matrix.to_numpy(float)
    env = envelope.to_numpy(float)
    floor = 1e-18
    for series in values:
        ax.plot(time, np.maximum(series, floor), color=BLUE2, lw=.28, alpha=.12)
    q25, median, q75 = np.quantile(values, [.25, .5, .75], axis=0)
    ax.fill_between(time, np.maximum(q25, floor), np.maximum(q75, floor),
                    color=BLUE2, alpha=.24, linewidth=0, label='observed IQR')
    ax.plot(time, np.maximum(median, floor), color=BLUE, lw=1.35,
            label='observed median')
    ax.plot(time, np.maximum(np.median(env, axis=0), floor), color=ORANGE,
            lw=1.05, ls='--', label='analytic envelope median')
    ax.set_yscale('log'); ax.set_ylim(5e-19, 8); ax.set_xlim(0, 4)
    ax.set_xlabel('Time'); ax.set_ylabel('Weighted observer error')
    ax.set_title('Observer error decays within its envelope', pad=4)
    # Series roles are defined in the caption; no legend is placed over data.
    clean(ax)


def draw_panel_d(ax, panel_d):
    values = [panel_d.observer_to_oracle.to_numpy(),
              panel_d.no_target_to_oracle.to_numpy()]
    colors = [BLUE, GREY]
    bp = ax.boxplot(values, positions=[0, 1], widths=.45, patch_artist=True,
        showfliers=False, medianprops=dict(color=DARK, lw=1.2),
        whiskerprops=dict(color=GREY, lw=.8), capprops=dict(color=GREY, lw=.8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(.26); patch.set_edgecolor(color)
    for index, (series, color) in enumerate(zip(values, colors)):
        ax.scatter(index + stable_offsets(len(series), .20), series,
            s=7, color=color, alpha=.46, edgecolor='none', zorder=3)
    ax.axhline(1, color=ORANGE, ls='--', lw=1)
    ax.set_yscale('log'); ax.set_ylim(.78, 6.4)
    ax.set_xticks([0, 1], ['Observer / oracle', 'No target / oracle'])
    ax.set_ylabel('Matched tail-error ratio')
    ax.set_title('Target information closes the performance gap', pad=4)
    clean(ax)


def draw_panel_e(ax, panel_e):
    policy_order = ['observer_loop', 'oracle_target', 'no_target']
    labels = {'observer_loop': 'Observer loop', 'oracle_target': 'Oracle target',
              'no_target': 'No target'}
    colors = {'observer_loop': BLUE, 'oracle_target': TEAL, 'no_target': GREY}
    markers = {'observer_loop': 'o', 'oracle_target': 's', 'no_target': '^'}
    steps = [.004, .002]
    for pindex, policy in enumerate(policy_order):
        medians = []
        for sindex, step in enumerate(steps):
            values = panel_e[(panel_e.policy == policy)
                             & np.isclose(panel_e.dt, step)][
                                 'endpoint_relative_difference'].to_numpy()
            if len(values) != 8: raise ValueError('panel e group is not n=8')
            x = sindex + (pindex - 1) * .16
            ax.scatter(x + stable_offsets(len(values), .045), values,
                s=12, marker=markers[policy], color=colors[policy], alpha=.68,
                edgecolor='white', linewidth=.25, zorder=3)
            medians.append(np.median(values))
        ax.plot(np.arange(2) + (pindex - 1) * .16, medians,
                color=colors[policy], marker=markers[policy], ms=3.2, lw=1,
                label=labels[policy], zorder=4)
    ax.axhline(.01, color=ORANGE, ls='--', lw=1, label='1% gate')
    ax.set_yscale('log'); ax.set_ylim(1e-14, 3e-2); ax.set_xlim(-.42, 1.42)
    ax.set_xticks([0, 1], ['0.004', '0.002'])
    ax.set_xlabel('RK4 step compared with 0.001')
    ax.set_ylabel('Relative endpoint difference')
    ax.set_title('Endpoint error is insensitive to step size', pad=4)
    clean(ax)


def write_provenance(outputs: list[Path], source_csvs: dict[str, Path]):
    builder = Path(__file__).resolve()
    shared = {
        'protocol': ROOT / 'observer_in_loop_certified' / 'PREREGISTRATION_R3.md',
        'proof': ROOT / 'observer_in_loop_certified' / 'PROOF.md',
        'validation': RES / 'validation_schemafix.json',
        'metadata': RES / 'metadata.json',
    }
    panel_sources = {
        'a': [shared['protocol'], shared['proof'], shared['metadata'], shared['validation']],
        'b': [RES / 'raw_runs.csv', RES / 'raw_trajectories.npz', source_csvs['b']],
        'c': [RES / 'raw_trajectories.npz', source_csvs['c']],
        'd': [RES / 'matched_policy_runs.csv', RES / 'matched_policy_trajectories.npz',
              source_csvs['d']],
        'e': [RES / 'matched_policy_dt_audit.csv', source_csvs['e']],
    }
    units = {
        'a': 'analytic identity; no empirical unit',
        'b': 'independently initialized computational run; n=120',
        'c': 'independently initialized computational run trajectory; n=120',
        'd': 'matched graph-heterogeneity-seed case; n=120',
        'e': 'matched graph-seed step comparison; n=8 per policy and displayed step',
    }
    panels = []
    for label in 'abcde':
        panels.append(dict(panel=label, independent_or_display_unit=units[label],
            sources=[dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                          sha256=sha256(path)) for path in panel_sources[label]],
            builder=dict(path=str(builder.relative_to(ROOT)), bytes=builder.stat().st_size,
                         sha256=sha256(builder)),
            outputs=[dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                          sha256=sha256(path)) for path in outputs]))
    ledger = dict(schema='cert-oil-r3-panel-provenance-v1', figure=STEM,
                  panels=panels)
    (OUT / f'{STEM}_PANEL_PROVENANCE.json').write_text(
        json.dumps(ledger, indent=2) + '\n')


def build():
    _, _, panel_b, panel_c, panel_d, panel_e, _ = load_and_derive()
    fig = plt.figure(figsize=(7.2, 6.35), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[.58, 1.0, 1.0],
                          left=.085, right=.985, bottom=.105, top=.97,
                          wspace=.34, hspace=.56)
    ax_a = fig.add_subplot(gs[0, :]); ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1]); ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_panel_a(ax_a); draw_panel_b(ax_b, panel_b); draw_panel_c(ax_c, panel_c)
    draw_panel_d(ax_d, panel_d); draw_panel_e(ax_e, panel_e)
    panel_label(ax_a, 'a', x=-.055, y=.93); panel_label(ax_b, 'b', x=-.15)
    panel_label(ax_c, 'c', x=-.15); panel_label(ax_d, 'd', x=-.15)
    panel_label(ax_e, 'e', x=-.15)

    handles = [
        mpl.lines.Line2D([], [], color=BLUE, marker='o', ms=3, lw=1,
                         label='Observer loop'),
        mpl.lines.Line2D([], [], color=TEAL, marker='s', ms=3, lw=1,
                         label='Oracle target'),
        mpl.lines.Line2D([], [], color=GREY, marker='^', ms=3, lw=1,
                         label='No target'),
        mpl.lines.Line2D([], [], color=ORANGE, ls='--', lw=1,
                         label='1% gate'),
    ]
    ax_e.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.24),
                frameon=False, ncol=4, columnspacing=.75, handlelength=1.2,
                fontsize=7.0)

    outputs = [OUT / f'{STEM}.{ext}' for ext in ('svg', 'pdf', 'png', 'tiff')]
    fig.savefig(OUT / "Fig12_observer_in_loop_certified.svg", metadata={"Date": None})
    fig.savefig(OUT / "Fig12_observer_in_loop_certified.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "Fig12_observer_in_loop_certified.png", dpi=600)
    fig.savefig(OUT / "Fig12_observer_in_loop_certified.tiff", dpi=600)
    plt.close(fig)
    source_csvs = {label: SOURCE / f'{STEM}_panel_{label}_{suffix}.csv'
        for label, suffix in [('b', 'envelope'), ('c', 'observer'),
                              ('d', 'matched'), ('e', 'dt')]}
    write_provenance(outputs, source_csvs)


if __name__ == '__main__':
    build()
