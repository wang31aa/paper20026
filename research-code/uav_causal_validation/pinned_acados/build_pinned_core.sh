#!/usr/bin/env bash
set -euo pipefail

ACADOS_COMMIT="91067daebe12c07d76d32a6aed0b8db00b3a54e1"
SOURCE_DIR="${1:-/private/tmp/acados-${ACADOS_COMMIT}}"

if [ ! -d "${SOURCE_DIR}/.git" ]; then
  git clone --recursive https://github.com/acados/acados.git "${SOURCE_DIR}"
fi

git -C "${SOURCE_DIR}" fetch origin "${ACADOS_COMMIT}"
git -C "${SOURCE_DIR}" checkout --detach "${ACADOS_COMMIT}"
git -C "${SOURCE_DIR}" submodule update --init --recursive
# The pinned Makefile creates all three archives before its GNU-specific
# `cp --parents` install step. macOS /bin/cp lacks that option, so accept the
# target's exit only after independently verifying the archives below.
make -C "${SOURCE_DIR}" -j4 static_library \
  BLASFEO_TARGET=GENERIC HPIPM_TARGET=GENERIC CC=clang || true

for archive in libacados.a libhpipm.a libblasfeo.a; do
  test -s "${SOURCE_DIR}/lib/${archive}"
done

shasum -a 256 \
  "${SOURCE_DIR}/lib/libacados.a" \
  "${SOURCE_DIR}/lib/libhpipm.a" \
  "${SOURCE_DIR}/lib/libblasfeo.a"
