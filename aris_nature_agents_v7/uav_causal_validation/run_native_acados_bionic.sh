#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates git curl wget unzip \
  octave liboctave-dev cmake build-essential gfortran

git clone --recursive https://github.com/acados/acados.git acados
git -C acados checkout 91067daebe12c07d76d32a6aed0b8db00b3a54e1
git -C acados submodule update --init --recursive
wget -q https://github.com/casadi/casadi/releases/download/3.4.5/casadi-linux-octave-v3.4.5.tar.gz
mkdir -p acados/external/casadi-octave
tar -xf casadi-linux-octave-v3.4.5.tar.gz -C acados/external/casadi-octave

mkdir -p acados/build
cd acados/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DBLASFEO_TARGET=GENERIC \
  -DHPIPM_TARGET=GENERIC \
  -DACADOS_OCTAVE=ON \
  -DACADOS_INSTALL_DIR=/work/acados
cmake --build . -- -j2
cmake --build . --target install
cd /work

curl -L --retry 15 -o matlab_code.zip \
  'https://zenodo.org/records/4379503/files/matlab_code.zip?download=1'
echo '02cb805c7ed6d00cac410f84a9a1a3c5da6dd228c1cb580e5038f85804b06e45  matlab_code.zip' | sha256sum -c -
unzip -q matlab_code.zip -d official_code
curl -L --retry 15 -o dataset.zip \
  'https://zenodo.org/records/4379168/files/dataset.zip?download=1'
echo 'ff012bec9c5206a312eb18d664f760fe821e8cb9b4c2c65ea8f6bf9462960934  dataset.zip' | sha256sum -c -
unzip -q dataset.zip 'dataset/data/mpc/comparison/01/workspace.mat'

export ENV_RUN=true
export ACADOS_INSTALL_DIR=/work/acados
export LD_LIBRARY_PATH=/work/acados/lib
export OCTAVE_PATH=/work/acados/interfaces/acados_matlab_octave:/work/acados/external/casadi-octave:/work/aris_nature_agents_v7/uav_causal_validation
mkdir -p aris_nature_agents_v7/uav_causal_validation/results
octave --no-gui --quiet --eval "addpath(genpath('/work/official_code/matlab_code')); native_octave_replay('dataset/data/mpc/comparison/01/workspace.mat','aris_nature_agents_v7/uav_causal_validation/results/native_acados_replay.mat')"
chmod -R a+rX aris_nature_agents_v7/uav_causal_validation/results
