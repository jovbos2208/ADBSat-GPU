#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nvcc_wrapper=$(command -v nvcc_wrapper || true)
if [[ -z "$nvcc_wrapper" ]]; then
  echo "nvcc_wrapper not found; install Kokkos or point CMAKE_CXX_COMPILER to it" >&2; exit 1
fi
cmake -S "$root_dir" -B "$root_dir/build-cuda" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$nvcc_wrapper" \
  -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON "$@"
cmake --build "$root_dir/build-cuda" -j "$(nproc)"
ctest --test-dir "$root_dir/build-cuda" --output-on-failure
