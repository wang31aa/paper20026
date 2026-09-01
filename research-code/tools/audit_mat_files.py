#!/usr/bin/env python3
"""Read-only structural/numeric audit for MATLAB v5 data files."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def worker(path: Path, root: Path) -> dict:
    import numpy as np
    from scipy.io import loadmat, whosmat

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "format": "MATLAB 5",
        "whosmat_status": None,
        "loadmat_status": None,
    }
    try:
        listing = whosmat(path)
        record["whosmat_status"] = "PASS"
        record["whosmat_count"] = len(listing)
    except Exception as exc:  # SciPy 1.18.0 compressed MAT regression
        record["whosmat_status"] = f"FAIL: {type(exc).__name__}: {exc}"

    variables = []
    data = loadmat(
        path,
        squeeze_me=False,
        struct_as_record=True,
        verify_compressed_data_integrity=True,
    )
    for name, array in data.items():
        if name.startswith("__"):
            continue
        array = np.asarray(array)
        numeric = np.issubdtype(array.dtype, np.number)
        item = {
            "file": record["path"],
            "variable": name,
            "shape": "x".join(map(str, array.shape)),
            "dtype": str(array.dtype),
            "elements": int(array.size),
            "numeric": bool(numeric),
            "real": None,
            "finite_count": None,
            "nonfinite_count": None,
            "min": None,
            "max": None,
            "min_abs": None,
            "max_abs": None,
        }
        if numeric:
            finite = np.isfinite(array)
            item["finite_count"] = int(finite.sum())
            item["nonfinite_count"] = int(array.size - finite.sum())
            item["real"] = not bool(np.iscomplexobj(array))
            if finite.any():
                values = array[finite]
                if np.iscomplexobj(array):
                    values = np.abs(values)
                    item["min_abs"] = float(values.min())
                    item["max_abs"] = float(values.max())
                else:
                    item["min"] = float(values.min())
                    item["max"] = float(values.max())
        variables.append(item)
    record["loadmat_status"] = "PASS"
    record["variable_count"] = len(variables)
    numeric = [item for item in variables if item["numeric"]]
    record["numeric_variable_count"] = len(numeric)
    record["numeric_element_count"] = sum(item["elements"] for item in numeric)
    record["nonfinite_count"] = sum(item["nonfinite_count"] for item in numeric)
    record["all_numeric_finite"] = record["nonfinite_count"] == 0
    del data
    gc.collect()
    return {"file": record, "variables": variables}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("json_out", type=Path)
    parser.add_argument("csv_out", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    if args.worker:
        print(json.dumps(worker(args.worker.resolve(), root), allow_nan=False))
        return

    files, variables = [], []
    for path in sorted(root.rglob("*.mat")):
        run = subprocess.run(
            [sys.executable, __file__, str(root), str(args.json_out), str(args.csv_out), "--worker", str(path)],
            check=True,
            text=True,
            capture_output=True,
        )
        result = json.loads(run.stdout)
        files.append(result["file"])
        variables.extend(result["variables"])

    by_name: dict[str, list[dict]] = {}
    for item in files:
        by_name.setdefault(Path(item["path"]).name, []).append(item)
    for group in by_name.values():
        if len(group) > 1:
            identical = len({item["sha256"] for item in group}) == 1
            for item in group:
                item["same_basename_count"] = len(group)
                item["same_basename_byte_identical"] = identical

    report = {
        "schema_version": "1.0",
        "method": "read-only scipy.io.whosmat/loadmat; no MATLAB code executed",
        "source_root": str(root),
        "files": files,
        "variables": variables,
    }
    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = list(variables[0])
    with args.csv_out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(variables)
    print(f"PASS: {len(files)} files; {len(variables)} variables")


if __name__ == "__main__":
    main()
