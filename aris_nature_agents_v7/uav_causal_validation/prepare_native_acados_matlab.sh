#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y --no-install-recommends cmake build-essential gfortran curl unzip
git clone --recursive https://github.com/acados/acados.git acados
git -C acados checkout 91067daebe12c07d76d32a6aed0b8db00b3a54e1
git -C acados submodule update --init --recursive
curl -L --retry 15 -o casadi-matlab.tar.gz https://github.com/casadi/casadi/releases/download/3.4.5/casadi-linux-matlabR2014b-v3.4.5.tar.gz
mkdir -p acados/external/casadi-matlab
tar -xf casadi-matlab.tar.gz -C acados/external/casadi-matlab
cmake -S acados -B acados/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBLASFEO_TARGET=GENERIC -DHPIPM_TARGET=GENERIC -DACADOS_INSTALL_DIR="$GITHUB_WORKSPACE/acados"
cmake --build acados/build --parallel 2
cmake --build acados/build --target install
curl -L --retry 15 -o matlab_code.zip 'https://zenodo.org/records/4379503/files/matlab_code.zip?download=1'
echo '02cb805c7ed6d00cac410f84a9a1a3c5da6dd228c1cb580e5038f85804b06e45  matlab_code.zip' | sha256sum -c -
unzip -q matlab_code.zip -d official_code
curl -L --retry 15 -o dataset.zip 'https://zenodo.org/records/4379168/files/dataset.zip?download=1'
echo 'ff012bec9c5206a312eb18d664f760fe821e8cb9b4c2c65ea8f6bf9462960934  dataset.zip' | sha256sum -c -
unzip -q dataset.zip 'dataset/data/mpc/comparison/01/workspace.mat'
mkdir -p aris_nature_agents_v7/uav_causal_validation/results
echo "ENV_RUN=true" >> "$GITHUB_ENV"
echo "ACADOS_INSTALL_DIR=$GITHUB_WORKSPACE/acados" >> "$GITHUB_ENV"
echo "LD_LIBRARY_PATH=$GITHUB_WORKSPACE/acados/lib:$GITHUB_WORKSPACE/build" >> "$GITHUB_ENV"
