#!/usr/bin/env bash
# This script installs the llvm shipped together with Phism.

set -o errexit
set -o pipefail
set -o nounset

echo ""
echo ">>> Install LLVM for Phism"
echo ""

TARGET="${1:-"local"}"

# The absolute path to the directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ "${TARGET}" == "local" ]; then
  "${DIR}/check-vitis.sh" || { echo "Xilinx Vitis check failed."; exit 1; }
fi

# Make sure llvm submodule is up-to-date.
git submodule sync
git submodule update --init --recursive

# Go to the llvm directory and carry out installation.
LLVM_DIR="${DIR}/../llvm"

cd "${LLVM_DIR}"
mkdir -p build
cd build

# Configure CMake
CC=gcc CXX=g++ cmake ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;llvm;clang;clang-extra-tools" \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DBUILD_POLYMER=ON \
  -DPLUTO_LIBCLANG_PREFIX="$(llvm-config --prefix)"
 
# Run building
cmake --build . --target all -- -j "$(nproc)"

if [ "${TARGET}" == "ci" ]; then
  # Run test
  cmake --build . --target check-llvm -- -j "$(nproc)"
fi
