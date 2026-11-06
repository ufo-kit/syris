CUDA optimization ledger
========================

Record a method here as useful only after a measured speedup and accuracy check.
Ratios apply to the listed workload, arithmetic mode, and reference, not to all
projections. Raw results include the source hashes used for that experiment.
The hardware so far is an RTX 2000 Ada. See :doc:`cuda-performance` for the first
round's detailed methodology and reproduction commands.

Confirmed useful
----------------

* **Do not initialize unused hit records** (2026-11-05).
  Float32 Armadillo and repaired Dragon, six 2048-square views: 1.55–1.65x for
  center sampling and 1.69–1.76x for adaptive depth 1 relative to ``f75c9af``.
  All compared pixels agree exactly. Only the written hit-list prefix is read.
  Evidence: ``benchmarks/results/rtx2000ada-float32.json``, ``uninitialized``.

* **Batch overflow events instead of searching once per crossing** (2026-11-05).
  Keep the next 32 distinct distances, including every duplicate orientation at
  retained distances, on each complete BVH traversal. No crossing-count limit.
  The head-on 140-layer stress case improved 58–61x in Float32, 44–63x in Float64,
  and 43–60x in mixed precision versus ``f75c9af``, with exact output agreement.
  These large gains require overflow; they do not describe ordinary mesh rays.
  Evidence: ``benchmarks/results/rtx2000ada-{float32,float64,mixed}.json``.

* **Combine those changes while retaining the 256-hit default** (2026-11-05).
  Float32 ordinary mesh views improved 1.52–1.67x. Float64 ordinary views were
  effectively unchanged. Out-of-line fallback storage gives up some common-ray
  speed compared with removing initialization alone, while accelerating dense
  rays. Precision, triangle predicates, and adaptive integration were retained.
  The first-round focused suite passed 120 tests and GPU memcheck/initcheck.

* **Signed accumulation for dense rays through a known material boundary**
  (2026-11-06). At 37 degrees, the 80- and 140-layer Float32 scenes improved
  **1.79–1.99x** for center sampling and **2.53–3.10x** for adaptive depth 1
  versus the already optimized ordered implementation. These are full projection
  times, with 64-square detectors and seven interleaved warmed repeats. All
  compared pixels match exactly. The 2048-square Armadillo/Dragon views did
  not show a material general gain (0.95–1.03x); some head-on layered views
  regressed. This is a conditional fast path, not a default replacement.
  Evidence: ``benchmarks/results/rtx2000ada-signed-float32.json``, ``signed``.
  A Float64 comparison (512-square meshes, 32-square layers, five repeats)
  confirms 1.29–1.71x center / 1.65–1.98x adaptive speedups on those oblique
  dense scenes. Float64 outputs differ by at most 4.75e-16 relatively and
  preserve zero and finite masks; exact equality is not claimed for that mode.
  Evidence: ``benchmarks/results/rtx2000ada-signed-float64.json`` and its
  separately retained ``-exact-attempt.json`` failure.
  The corresponding mixed-precision run matches every pixel exactly and gives
  1.32–1.70x center / 1.50–2.15x adaptive speedups on the oblique dense scenes.
  Evidence: ``benchmarks/results/rtx2000ada-signed-mixed.json``.

* **Separate the signed retry kernel for mixed-precision cone-beam views**
  (2026-11-06). On Armadillo and repaired Dragon at 1024 square, the deferred
  variant gives about **1.50x** for center sampling and **1.55–1.58x** for
  adaptive depth 1 versus the optimized ordered projector. Every one of the
  seven paired repeats improves, and all 12,582,912 compared pixels match
  exactly. Parallel views are roughly unchanged (0.99–1.04x). This retains the
  same material-boundary contract as signed accumulation; it is not a general
  geometry mode. The smaller 512-square run also showed the cone-beam benefit.
  Evidence: ``benchmarks/results/rtx2000ada-signed-deferred-mixed1024.json``.

* **Select box components once per axis** (2026-11-06). This is now the default
  box predicate. It preserves the original component values, slab order,
  directed arithmetic, and short-circuit decisions. On 1024-square Armadillo
  and repaired Dragon, projection speedups are 1.14–1.27x in Float32,
  0.98–1.13x in Float64, and 1.04–1.21x in mixed precision versus the indexed
  predicate in the already optimized worktree. Layered stress scenes have
  mixed results, including regressions; see the detailed comparison below.

* **Assign compact detector tiles to warps** (2026-11-06). With the scalar
  predicate, 4-column by 8-row tiles improve ordinary mesh projection times
  by another 1.08–1.26x in Float32, 1.10–1.35x in Float64, and 1.15–1.34x in
  mixed precision. Each lane retains its original rays and adaptive samples.
  Enable this optional scheduler with ``syris.init(compute_backend="cuda",
  warp_tile_width=4)``. No shared intersection list or sample approximation
  is involved. All compared final images match the indexed reference exactly.
  Evidence: ``benchmarks/results/rtx2000ada-scheduling-final-*.json``.

Not adopted as defaults
-----------------------

* **32-hit common list with a 256-hit device retry:** retries can make moderately
  layered geometry slower. Retained as a compile-time experiment.
* **Shared root samples and a compact adaptive refinement queue:** extra passes,
  global storage, and repairs for differently rounded shared corners outweighed
  reuse. It was 11–31% slower than the optimized local sampler on the six
  Float32 adaptive mesh views. Retained as ``sampling_mode="staged"``.
* **A capped persistent worker pool:** no consistent additional gain on the
  full mesh projections. The default launch grid was retained.
* **Selecting each near/far box plane independently:** the first scalar
  prototype raised the mixed parallel normals kernel from 124 to 130
  registers, crossing the two-block residency limit. Some parallel mesh
  projections fell to 0.61–0.78x. Selecting both components once per axis
  reduced the final kernel to 128 registers and removed that regression in
  the final mesh comparison. Earlier ``scalar`` and ``tile4`` benchmark
  variants retain this prototype; ``scalar_axes`` and ``tile4_axes`` use
  the adopted predicate.
* **A separate signed exception pass as a general default:** reduces local
  storage in the Float32 normals kernel from 5328 to 1776 bytes, but Float32
  mesh projection gains were only 0.99–1.11x in the 2048-square run, and head-on
  dense adaptive cases could be substantially slower. The confirmed mixed
  cone-beam gains above do not establish a benefit for every workload.
* **128-thread blocks and depth-specific caches:** no consistent gain across
  mesh views. Depth-specific builds retain the requested sample pattern and
  matched every compared pixel. These remain benchmark experiments; neither
  changes the ordinary launch or sampling configuration.
  Evidence: ``benchmarks/results/rtx2000ada-signed-ablations-float32.json``
  and ``benchmarks/results/rtx2000ada-depth-specialization-float32.json``.

Scalar selection and warp tiles
-------------------------------

The final comparison uses seven interleaved warmed repeats per variant on
RTX 2000 Ada, CuPy 13.6.0, CUDA runtime 13.0. Each precision includes Armadillo
(345,944 triangles) and repaired Dragon (871,506 triangles), 1024-square
detectors, parallel views at 0 and 37 degrees, a cone view at 37 degrees,
and center sampling / adaptive depth 1. The table gives ranges across the
12 mesh cases, using median full projection time. Timing includes launch
preparation and output scaling, but excludes compilation, BVH construction,
and image downloads.

.. list-table:: Projection speedup on the ordinary meshes
   :header-rows: 1

   * - Precision
     - Scalar vs indexed
     - Tiles vs scalar
     - Scalar + tiles vs indexed
   * - Float32
     - 1.14–1.27x
     - 1.08–1.26x
     - 1.32–1.48x
   * - Float64
     - 0.98–1.13x
     - 1.10–1.35x
     - 1.12–1.44x
   * - Mixed
     - 1.04–1.21x
     - 1.15–1.34x
     - 1.24–1.47x

The reference is ``indexed``, which retains the earlier hit-storage and
overflow optimizations. These are incremental gains, not comparisons against
``f75c9af``; the JSON's ``reference_variant`` identifies this distinction.
The benchmark compares the final image for each variant/view, with zero
absolute and relative comparison tolerance. All 72 scene/view/depth cases
across the three precisions pass, with unchanged finite and zero masks.

The additional 20- and 140-layer stress scenes use 64-square detectors.
Scalar-only ratios range from 0.87–1.09x in Float32, 0.85–1.18x in Float64,
and 0.98–1.21x in mixed precision. Combining scalar selection and 4x8 tiles
gives 1.07–1.99x, 1.05–2.12x, and 1.07–1.85x respectively against indexed.
Tiling is still optional: scene, resolution, and adaptive divergence affect
the balance between locality and waiting for the slowest lane.

Four tile shapes were screened with the first scalar prototype. The smaller
512-square cases were less consistently improved, and a capped worker pool
gave no consistent additional benefit. Those exploratory results are retained
in ``rtx2000ada-warp-tiles-screen-float32.json`` and
``rtx2000ada-warp-tiles-{float32,float64,mixed}.json``; they use the earlier
predicate and must not be confused with the final ``scheduling-final`` files.
See :doc:`cuda-performance` for reproduction commands.

Inspection of the actual cached Float32 parallel normals kernel gives
128 / 124 / 122 registers and 5328 / 5296 / 4400 bytes of local storage for
indexed / scalar / scalar plus tiles. Static local-load/store instruction
counts are 655/885, 545/806, and 544/836 respectively. These counts describe
the named kernel's instructions, not executed memory traffic, fallback callees,
or measured stalls. Evidence: ``rtx2000ada-scheduling-codegen.json``.

The final focused CUDA suite passes 217 tests without skips, including 491,520
box-decision comparisons across three precisions and 210 scheduler coverage
combinations. Projection comparisons cover four tile shapes, both camera
types, normals and even/odd crossings, partial edges, the deepest supported
adaptive sampling, overflow, and NaN preservation. CUDA memcheck and synccheck
both report zero errors. The profiler driver also passes a width-4 line-info
capture with exact warmup agreement. Evidence is in the
``rtx2000ada-scheduling-{validation,checks,profile-smoke,audit}.json`` reports.

A narrow-layer cone stress fixture also exposed one existing Float64 NaN,
at both center sampling and adaptive depth 1. The preserved pre-change CUDA
snapshot, indexed predicate, scalar predicate, and all tested tile shapes
agree on that result. The fixture remains in the scheduling equality checks.
A separate wider fixture makes every ray cross all 140 layers and checks
finite outputs and analytic center-ray thickness. The original failed
finite-output assertion is retained in
``rtx2000ada-scheduling-validation-first-attempt.json`` and
``scheduling-regressions-first-attempt.log``; the diagnosis is in
``rtx2000ada-scheduling-fixture-check.json``. This numerical limitation has
not been repaired by the scheduling changes.

Signed-mode contract
--------------------

Compile-time ``SYRIS_SIGNED_THICKNESS`` uses compensated Float64 accumulation
and the same rebased triangle distances. Shared-edge/vertex hits, uncertain
orientations, and inconsistent endpoint counts use the ordered fallback. It
requires a closed, correctly oriented 0/1 material boundary: overlapping or
duplicate shells violate the contract even if endpoint counts balance. It is
not enabled in ordinary library initialization. The benchmark requires explicit
``--assume-material-boundary`` to run it; this asserts rather than validates
the input contract. See :doc:`cuda-projection` for the two fallback strategies.

Profiling status
----------------

A warmed Nsight Systems capture works through the existing CuPy setup. For
1024-square Armadillo at adaptive depth 1, the projector accounts for 99.4% of
GPU kernel time. A subsequent administrator-authorized Nsight Compute capture
completed 25 replay passes and preserved the checked output exactly.

The counter report shows 6.47 active threads per warp, 31.79% achieved occupancy
limited by registers, and substantial memory-dependency waits. Source mapping
places 76.94% of theoretical local-memory sectors at the dynamically indexed
arrays in the box predicate, and 96.19% of excessive theoretical global-memory
sectors at BVH node accesses. These motivated the scalar selection and warp
tile experiments above. The counters themselves are not measured speedups: sampled
wait locations include dependencies on preceding loads, theoretical sectors
are not physical post-cache traffic, and cache/clock control was disabled.
See :doc:`profiling` for the evidence and interpretation limits.
No post-change hardware-counter capture has been collected. SAH and node-layout
experiments remain deferred.
