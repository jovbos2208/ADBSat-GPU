#!/usr/bin/env bash
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cmake -S "$root_dir" -B "$root_dir/build-hip" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=hipcc \
  -DKokkos_ENABLE_HIP=ON -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_ROCTHRUST=OFF "$@"
cmake --build "$root_dir/build-hip" -j "$(nproc)"
ctest --test-dir "$root_dir/build-hip" --output-on-failure
