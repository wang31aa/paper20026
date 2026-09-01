#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/wbh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3}"
RUNTIME="$ROOT/runtime_py312"
WORKSPACE="${1:?usage: run_compatible_fallback.sh OFFICIAL_WORKSPACE.mat}"

export PYTHONPATH="$RUNTIME:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - <<'PY'
import casadi, numpy, scipy
assert casadi.__version__ == "3.7.2"
assert tuple(map(int, scipy.__version__.split(".")[:2])) == (1, 16)
print({"casadi": casadi.__version__, "numpy": numpy.__version__,
       "scipy": scipy.__version__})
PY

"$PYTHON_BIN" "$ROOT/test_python_translation.py"
"$PYTHON_BIN" "$ROOT/test_double_integrator_translation.py"
"$PYTHON_BIN" "$ROOT/smoke_test_casadi_nmpc.py" "$WORKSPACE"

test -s "$ROOT/results/casadi_nmpc_smoke_test.json"
"$PYTHON_BIN" - "$ROOT/results/casadi_nmpc_smoke_test.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
assert p["success"] is True
PY
