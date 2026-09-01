#!/usr/bin/env python3
"""Post-hoc correction of one provable Python transcription error.

This does not replace or overwrite the preregistered graph-2 held-out failure.
MATLAB graph-2 line 790 uses MA4 in the AB3 update after each step, whereas the
frozen v1 Python port inferred a uniform graph operation and therefore used
MB4.  No parameter or tolerance changes are made.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from literal_static_graph import run  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.mat, "graph2", posthoc_graph2_ab3_ma4=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    meta = {"graph": "graph2", "mat": str(args.mat), "steps": int(result["xx"].shape[-1]),
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "implementation": "posthoc-ab3-ma4-v1",
            "changes": ["AB3[k] = 2*MB3[k] - MB2[k] - MA4[k] for k>=2"]}
    args.output.with_suffix(".environment.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
