#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmake -S "$root_dir" -B "$root_dir/build-openmp" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_SERIAL=ON "$@"
cmake --build "$root_dir/build-openmp" -j "$(nproc)"
ctest --test-dir "$root_dir/build-openmp" --output-on-failure
