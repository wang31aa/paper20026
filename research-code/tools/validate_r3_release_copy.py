#!/usr/bin/env python3
"""Replay the frozen R3 validator without modifying release evidence bytes."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "observer_in_loop_certified"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cert-oil-r3-release-") as tmp:
        copied = Path(tmp) / MODULE.name
        shutil.copytree(MODULE, copied)
        for report in ("validation_schemafix.json", "validation_portable.json",
                       "validation_portable_v2.json"):
            path = copied / "results_r3" / report
            if path.exists():
                path.unlink()
        lock = copied / "results_r3.completed.lock"
        payload = json.loads(lock.read_text(encoding="utf-8"))
        payload["target"] = str((copied / "results_r3").resolve())
        lock.chmod(0o600)
        lock.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(copied / "validate_r3_portable.py"),
             str(copied / "results_r3")],
            cwd=copied,
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    print("PASS: R3 replay completed in a temporary copy; release bytes unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
