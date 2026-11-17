# Performance report

Measured 2026-11-17 on:

- CPU: AMD Ryzen 5 PRO 8540U, 6 cores / 12 threads
- GPU: integrated AMD Radeon 740M (`gfx1100`, reported as HawkPoint2)
- ROCm/HIP 7.1.52801, Clang 21.1.8, GCC 15.2.0
- Kokkos 5.2.2, Release (`-O3`), double precision
- model: Sentman; quasi-uniform synthetic sphere
- five warmups, 15 synchronized repetitions; median shown
- device-resident geometry; allocation and initial H2D transfer excluded

### Initial baseline (before optimization)

| panels | HIP median | HIP Mpanels/s | OpenMP median | OpenMP Mpanels/s | HIP/OpenMP |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.160 ms | 8.62 | 0.211 ms | 47.31 | 0.18x |
| 100,000 | 8.731 ms | 11.45 | 5.573 ms | 17.94 | 0.64x |
| 1,000,000 | 47.624 ms | 21.00 | 28.705 ms | 34.84 | 0.60x |
| 5,000,000 | 199.664 ms | 25.04 | 126.227 ms | 39.61 | 0.63x |

At 1,000,000 panels, a separate single-thread OpenMP run measured 114.240 ms
(8.75 Mpanels/s), making HIP 2.40x faster than one CPU thread. It is nevertheless
slower than the full 12-thread CPU on this low-power integrated GPU. The kernel
contains `exp`, `erf`, trigonometry, and a six-value reduction; those costs plus
GPU launch overhead dominate small meshes. A discrete Radeon or NVIDIA GPU is
expected to have a different crossover, but no claim is made without measuring it.

Correctness gates passed on both HIP and OpenMP. The analytical tests cover a
front/back Newton plate, symmetry and moment cancellation, plus finite outputs
for every implemented model. HIP and OpenMP produced matching benchmark drag
coefficients (`C_D = 2.31900863` at one million panels).

## Optimized result

The panel kernel now carries `cos(delta)` directly from the normal/flow dot
product and computes `sin(delta)=sqrt(1-cos(delta)^2)`. This removes one `acos`
and all subsequent `sin/cos` calls per panel. Coefficient arrays are no longer
written to global memory unless the caller requests panel-level output.

At 1,000,000 Sentman panels (20 repetitions):

| implementation | HIP median | throughput |
|---|---:|---:|
| initial | 47.624 ms | 21.00 Mpanels/s |
| optimized math | 22.380 ms | 44.68 Mpanels/s |
| optimized math + conditional stores | 21.802 ms | 45.87 Mpanels/s |

This is a **2.18x kernel speedup**. The optimized 12-thread OpenMP median is
21.203 ms, so the integrated GPU is now within about 3% of the full CPU rather
than 40% slower.

## Shadowing redesign

The Python/MATLAB algorithm projects every candidate triangle for each flow
direction and scans potential front/back pairs. Its worst-case work is O(N^2)
and it creates multiple temporary candidate arrays.

The GPU implementation builds a balanced, flat triangle BVH once in body
coordinates (O(N log N)). Each flow-facing panel then casts one upstream ray and
traverses only intersected BVH nodes, typically O(log N + hits). It keeps the
Python semantics: only back-facing panels occlude, the 90.0001-degree split is
preserved, and the third barycentric edge remains open. A two-triangle fixture
returns panel 0 as shadowed in both the original Python implementation and the
new HIP/OpenMP implementation.

Stress case: two aligned layers, 100,000 total triangles, exactly 50,000
shadowed, five warmups and 15 synchronized repetitions:

| backend | median | throughput | detected shadows |
|---|---:|---:|---:|
| HIP / Radeon 740M | 10.343 ms | 9.67 Mpanels/s | 50,000 |
| OpenMP / 12 threads | 9.269 ms | 10.79 Mpanels/s | 50,000 |

The remaining shadow cost is dominated by divergent BVH traversal, irregular
node loads, ray/AABB tests, and ray/triangle tests. On this small integrated GPU
those costs are close to the 12-thread CPU; batching orientations or using a
discrete GPU should improve occupancy, but requires measurement.

## Profiling and next optimization targets

1. Establish production mesh-size and orientation-batch distributions.
2. Batch many attitudes per launch; this should improve GPU occupancy far more
   than micro-optimizing a single small mesh.
3. Profile with `rocprofv3 --stats build-hip/adbsat-benchmark ...` when the ROCm
   profiler package is installed. It was not available on this host.
4. Evaluate polynomial/special-function alternatives only under an explicit
   error budget; `erf` is a likely hotspot but numerical behavior is contractual.
5. Add per-model Python-vs-C++ golden fixtures before production integration.

Do not include initialization or H2D transfer only when production reuses a
`Solver`. For one-shot calls, end-to-end time must include those costs.

## Method reference

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
