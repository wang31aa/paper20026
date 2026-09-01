#!/bin/sh
set -eu
PYTHON=${PYTHON:-../../aris_nature_repro_env/bin/python3}
"$PYTHON" run_v13.py development
"$PYTHON" fit_v13.py
"$PYTHON" run_v13.py confirmation
"$PYTHON" analyse_v13.py
"$PYTHON" test_feature_identifiability.py
"$PYTHON" validate_v13.py
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-v13}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/tmp/cache-v13}
export MPLCONFIGDIR XDG_CACHE_HOME
"$PYTHON" plot_eight_domain.py
"$PYTHON" run_v14_baselines.py
"$PYTHON" analyse_v14_baselines.py
"$PYTHON" analyse_information_physical_separation.py
"$PYTHON" validate_v14_baselines.py
"$PYTHON" ../audit/validate_eight_domain_heterogeneity.py
