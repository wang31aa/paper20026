#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

R=Path(__file__).resolve().parent
d=pd.read_csv(R/'results/v10_runs.csv')
def summarize(cols,name):
    g=d.groupby(cols,as_index=False).agg(successes=('success','sum'),runs=('success','size'),
        success_rate=('success','mean'),median_task_error=('task_error','median'),
        median_control_energy=('control_energy','median'))
    g.to_csv(R/f'results/{name}.csv',index=False)
summarize(['domain','policy','eta'],'v10_eta_summary')
summarize(['domain','policy','n'],'v10_size_summary')
summarize(['domain','policy','topology'],'v10_topology_summary')
print('analysis PASS')
