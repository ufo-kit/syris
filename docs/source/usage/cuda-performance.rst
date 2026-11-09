CUDA performance experiments
============================

The retained changes remove unused hit-record initialization and process
overflow rays in batches of distinct events. They retain the 256-hit default
buffer, ray/triangle predicates, precision modes, boundary grouping, and pixel
integration rules. The small-buffer and shared-sampling experiments remain
available for comparison, rather than becoming defaults.
See :doc:`cuda-performance-ledger` for the running record of confirmed methods
and experiments that have not justified a default change.

These measurements predate the centred-coordinate precision modes and refer
to the source hashes in their reports. In particular, the mixed mode measured
here still stored hit distances in Float32. See :doc:`cuda-projection` for the
current mode definitions.

Why batch overflow events?
--------------------------

Previously, a ray with more than 256 triangle hits restarted the whole BVH
search for each successive crossing. The new fallback retains the next 32
distinct distances per traversal. It combines every orientation at each retained
distance, sorts that batch, integrates it, and continues beyond its last event.

Once the batch is full, its largest retained distance can only decrease. An
evicted event therefore cannot become part of that batch again. A retained
event keeps every duplicate hit from the complete traversal, including groups
larger than the buffer. There is no limit on the number of batches. This avoids
both truncation and the per-crossing traversal cost, with bounded storage.

Measurements
------------

Results are saved under ``benchmarks/results``. The benchmark compares against
the CUDA sources at ``f75c9af`` and records source hashes, mesh hashes, GPU and
CuPy versions, kernel attributes, individual timings, finite-mask differences,
and exact pixel comparisons. The repaired Dragon input is a local precomputed
NPZ; it is not regenerated or downloaded by the benchmark.

The main experiment uses an RTX 2000 Ada, Float32, Armadillo and repaired Dragon,
2048 by 2048 detectors, parallel views at 0 and 37 degrees and a cone-beam view
at 37 degrees, with center sampling and adaptive depth 1. Five warmed repeats
use interleaved variant order. The GPU also drives a desktop; clocks are not
locked. Small images of layered boxes exercise dense intersections separately.

The default changes give **1.52–1.65x** speedups for center sampling and
**1.56–1.67x** for adaptive depth 1 across the six mesh views. All 50,331,648
mesh pixels per variant match the baseline exactly; none are nonfinite.
Removing initialization alone gives 1.55–1.65x and 1.69–1.76x respectively:
the batched fallback's device-call storage has a modest common-ray cost in
exchange for much faster overflow handling.

For 140 separated layers viewed head-on, using a 16 by 16 detector:

.. list-table:: Dense-ray stress case (median full projection time)
   :header-rows: 1

   * - Sampling
     - Baseline
     - Optimized
     - Speedup
   * - Center ray
     - 877.530 ms
     - 15.126 ms
     - 58.0x
   * - Adaptive depth 1
     - 5343.417 ms
     - 87.896 ms
     - 60.8x

These large multiples are specific to the overflow stress case. The 20- and
80-layer cases mostly fit the ordinary buffer and do not show that benefit.
The staged sampler is 11–31% slower than the optimized local sampler on the
six adaptive mesh views, so sharing samples is not a demonstrated additional
speedup in this implementation. Dense-case outputs also match exactly.

Additional 512 by 512 mesh comparisons, with three warmed repeats, show the
precision dependence:

.. list-table:: Default speedup relative to the same-precision baseline
   :header-rows: 1

   * - Precision
     - Center sampling
     - Adaptive depth 1
   * - Float64
     - 0.95–1.12x
     - 0.99–1.05x
   * - Mixed (Float32 storage, Float64 triangle arithmetic)
     - 1.04–1.49x
     - 1.08–1.54x

Ordinary Float64 projections are effectively unchanged on this GPU. The
head-on 140-layer overflow cases improve 44–63x in Float64 and 43–60x in mixed
precision, using 8 by 8 detectors. All 30 cases in each of the three precision
reports match their respective baseline outputs exactly. No precision reduction
was used to obtain these speedups.

Wall time includes argument preparation, device allocations, geometry scaling,
projection, and output scaling. It excludes JIT compilation, mesh loading,
repair, BVH construction, download, and file writing. CUDA-event timings sum
the projection kernels on the actual launch stream. BVH construction is recorded
separately. These are not complete scan-throughput measurements.

Other prototypes and priorities
-------------------------------

* **Small common hit buffer:** ``SYRIS_HIT_CAPACITY=32`` retries overflowing rays
  with a separate 256-hit device function before event batching. It avoids the
  immediate repeated-traversal penalty of simply setting the capacity to 16.
  Nevertheless, extra traversal and device-call storage can outweigh the smaller
  list. It is an ablation, not the default configuration.
* **Shared adaptive samples:** ``sampling_mode="staged"`` traces a shared corner
  grid and pixel centers, repairs mismatched corners, classifies pixels, and
  refines a compact queue. Neighboring pixels can compute different floating-point
  coordinates for a nominally shared corner. Reuse is permitted only for exactly
  matching coordinates; the remaining rays use the original calculation.
  Additional launches, global scratch storage, corner repairs, and a large
  refinement kernel offset the saved rays on the measured meshes. Deeper levels
  still use the local sampler. This prototype remains opt-in.
* **Persistent workers:** capping the dynamic scheduler's worker pool did not
  show a consistent gain on the full mesh projections. The default launch grid
  is retained. Staged work queues use a bounded worker pool because later stages
  may have very little work. The initial ablation report includes both schedules.
* **Rasterized signed lengths:** worthwhile as a separate backend experiment.
  A signed sum integrates winding multiplicity, whereas material thickness
  requires the intended material boundary. Coincident triangle edges also need
  single ownership; the current inclusive predicate cannot simply feed an
  unordered sum. Cavities, overlapping shells, internal ray origins, precision,
  and cancellation need independent validation. `gVirtualXRay
  <https://gvirtualxray.sourceforge.io/>`_ is a relevant existing GLSL backend.
  No rasterization speedup is claimed here.
* **Hardware traversal:** OptiX or `Vulkan ray queries
  <https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html>`_ need a
  benchmark of all crossings and thickness, including boundary behavior and
  precision. Rendering throughput alone does not answer this question.
* **Scan overhead:** cached scaled geometry, reusable scratch buffers and
  overlapping transfers with file output remain useful follow-ups. They do not
  remove the measured kernel cost; this prototype leaves scan orchestration
  unchanged.

Signed accumulation experiment (2026-11-06)
-------------------------------------------

For a closed, outward-oriented material boundary, each occupied ray interval
contributes its exit distance minus its entry distance. Thus the total is
``sum(exits) - sum(entries)``; traversal order is irrelevant in exact arithmetic.
An origin inside material contributes an initial interval starting at zero.
The implementation rebases the ray, uses the original triangle distances, and
accumulates in compensated Float64 arithmetic. It falls back to the ordered
method on exact triangle boundaries, uncertain normal signs, inconsistent
endpoint counts, or invalid totals.

This identity measures winding multiplicity. It represents material thickness
only when occupancy is zero or one. For example, two overlapping unit-thickness
boxes can have signed thickness 2 while their union has thickness 1.5. Endpoint
counts alone cannot distinguish them. Consequently the prototype requires an
explicit material-boundary assumption and remains outside ordinary library
initialization.

The Float32 comparison uses the already optimized ordered projector as its
reference, seven interleaved warmed repeats, 2048-square ordinary mesh images,
and 64-square images of separated boxes. Both signed variants match all
50,405,376 compared pixels per variant exactly, including the finite masks.

.. list-table:: Signed method versus optimized ordered projection, full wall time
   :header-rows: 1

   * - Workload
     - Center sampling
     - Adaptive depth 1
   * - Armadillo and repaired Dragon, six views
     - 0.95–1.02x
     - 0.95–1.03x
   * - 80 and 140 layers, 37-degree parallel/cone views
     - 1.79–1.99x
     - 2.53–3.10x

Head-on layered views are mixed and can regress; exact diagonal hits trigger
the ordered fallback. These gains establish a useful dense-ray case, not a
general replacement for the ordered projector. The full report, including
20-layer cases, is ``benchmarks/results/rtx2000ada-signed-float32.json``.

In Float64, a separate 512-square mesh / 32-square layer comparison with five
repeats gives 1.29–1.71x for center rays and 1.65–1.98x for adaptive sampling in
the oblique dense cases. Ordinary mesh views remain roughly unchanged. The
signed sum is not bit-for-bit identical in Float64: its largest observed
relative difference is 4.75e-16 (largest absolute difference 2.85e-14), with
identical zero and finite masks. The initial exact comparison stopped and is
preserved as ``rtx2000ada-signed-float64-exact-attempt.json``. The completed
``rtx2000ada-signed-float64.json`` uses ``atol=0`` and ``rtol=8*epsilon(Float64)``;
it still records every exact difference. These comparison tolerances do not
alter projector arithmetic or adaptive refinement tolerances.

The same smaller comparison in mixed precision matches every pixel exactly.
Signed accumulation gives 1.32–1.70x center / 1.50–2.15x adaptive speedups on
the oblique dense scenes. Its in-kernel fallback can regress ordinary parallel
mesh views, so the Float32 result should not be extrapolated between arithmetic
modes. See ``rtx2000ada-signed-mixed.json`` for all views and both fallback modes.

``signed_deferred`` moves exceptional pixels to a device queue and repeats
their complete original sampling in a separate kernel. Its fast Float32 kernel
uses 1776 bytes of local storage instead of 5328 in the ordered kernel (4496
for ``signed`` with its in-kernel fallback). The ordinary mesh projection ratios
are only 0.99–1.11x, and head-on dense adaptive cases can be slower. No automatic
choice between the methods is made.

The separate retry pass does help the mixed-precision cone-beam mesh views.
A second run at 1024 square with seven interleaved repeats gives about 1.50x
for center sampling and 1.55–1.58x for adaptive depth 1 on Armadillo and repaired
Dragon. Every paired repeat improves; all 12,582,912 compared pixels match
exactly. Parallel views remain roughly unchanged (0.99–1.04x). See
``rtx2000ada-signed-deferred-mixed1024.json``. This is a confirmed conditional
benefit of separating fallback storage, with the same geometry contract.

Compiling for exactly depth 0 or 1 and using 128-thread blocks were also tested.
Both preserved the compared outputs, but neither produced a consistent gain
across mesh views. Their reports are
``rtx2000ada-depth-specialization-float32.json`` and
``rtx2000ada-signed-ablations-float32.json``. Reports capture the source hashes at
measurement time; earlier ablations need not have the latest checkout's hash.

Reproduce
---------

Run from this checkout with its CUDA/CuPy environment and PyVista installed::

    python -m benchmarks.cuda_projection \
        --mesh /path/to/Armadillo.ply --mesh /path/to/repaired-dragon.npz \
        --size 2048 --layer-size 16 --depths 0 1 --repeats 5 \
        --output /tmp/cuda-float32.json

The benchmark extracts baseline sources into a temporary directory and compares
``baseline``, ``uninitialized``, ``optimized``, ``hybrid`` (32 hits), and ``staged``.
Use ``--precision float64`` or ``--precision mixed`` for the other arithmetic
modes, and ``--variants baseline optimized staged`` for a shorter comparison.
Without mesh paths it uses synthetic layered boxes. An input NPZ must contain
indexed ``vertices`` and triangle ``faces`` arrays. All comparisons use the same
uploaded geometry and BVH; no mesh simplification is involved.

For the conditional signed comparison::

    python -m benchmarks.cuda_projection \
        --mesh /path/to/Armadillo.ply --mesh /path/to/repaired-dragon.npz \
        --size 2048 --layer-size 64 --layers 20 80 140 \
        --depths 0 1 --repeats 7 --reference optimized \
        --variants optimized signed signed_deferred --assume-material-boundary \
        --output /tmp/cuda-signed-float32.json

The signed flag asserts the geometry contract; it does not validate it. Omit
it for general meshes and use the default ordered projector. ``specialized``
and ``signed_specialized`` compile separate kernels for the requested depths;
their dispatcher rejects a depth it has not compiled. ``blocks128`` changes
only launch block size. Comparisons are exact by default. Explicit ``--atol``
and ``--rtol`` allow numerical comparisons while still reporting exact changes.
The Float64 study above adds ``--precision float64 --size 512 --layer-size 32
--layers 80 140 --repeats 5 --rtol 1.7763568394002505e-15``.
The mixed-precision study uses ``--precision mixed`` with those smaller sizes
and repeat count, keeping the default exact comparison.
The larger mixed retry comparison uses ``--precision mixed --size 1024 --layers
--repeats 7 --variants optimized signed_deferred``; the empty ``--layers``
list omits the synthetic scenes.

For the subsequent scalar-selection and warp-tile comparison::

    python -m benchmarks.cuda_projection \
        --mesh /path/to/Armadillo.ply --mesh /path/to/repaired-dragon.npz \
        --size 1024 --layer-size 64 --layers 20 140 \
        --depths 0 1 --repeats 7 --reference indexed \
        --variants indexed scalar_axes tile4_axes \
        --output /tmp/cuda-scheduling-float32.json

Repeat with ``--precision float64`` and ``--precision mixed``. ``indexed``
keeps the earlier hit-storage and overflow optimizations and selects the
original array-indexed box predicate. ``scalar_axes`` uses the adopted scalar
predicate with per-thread scheduling; ``tile4_axes`` adds 4-column by 8-row
warp tiles. The ``tile8_axes``, ``tile16_axes``, and ``tile32_axes`` variants
test other tile shapes. Unqualified ``scalar`` and ``tile4``/``tile8``/etc.
retain the first, unadopted scalar prototype for reproducing earlier results.
``optimized`` uses ordinary library initialization, whose box predicate now
matches ``scalar_axes``. See :doc:`cuda-performance-ledger` for measured gains
and regressions.

Validation
----------

The focused suite covers all arithmetic modes, normals and even/odd crossings,
thin layers, cavities, shared boundaries, reversed rays, shuffled geometry,
both hit-buffer limits, duplicate groups larger than the buffer, internal ray
origins, every supported sampling depth, rectangular images, NaN propagation,
and staged/local equality::

    SYRIS_REQUIRE_CUDA=1 python -m unittest \
        syris.tests.test_cuda_projection syris.tests.test_mesh_normals \
        syris.tests.test_mesh_repair -q

CUDA memcheck and initcheck additionally cover representative staged and
overflow paths: 120 focused tests passed, and both sanitizer runs reported zero
errors. A separate seven-test camera suite has one existing constructor failure
for a missing shape; both that test and the camera implementation are unchanged
from the baseline. Local include contents now participate in the CuPy module-cache
key: changing a header previously could reuse a stale kernel binary. A regression
test compiles the same source twice with different header contents.

The signed study adds analytic and contract-limit tests in
``syris.tests.test_cuda_signed_projection``, plus queue and depth-specialization
checks in ``syris.tests.test_cuda_projection_variants``. Together with the
focused suite above, **219 tests passed** for that study's source state. The retry
tests cover empty, partial, and full queues, every supported adaptive depth,
rectangular/singleton images, cavities, and preservation of invalid NaNs in all
three arithmetic modes. The overlap counterexample explicitly tests the
signed mode's contract limitation, rather than pretending balanced counts
detect an arbitrary invalid mesh.
Five additional memcheck cases and three initcheck cases report zero errors.
Commands, counts, comparison tolerances, and the audit of saved results are
recorded in ``benchmarks/results/validation-signed.txt``.

The subsequent scalar-selection and warp-tile study passes 217 focused CUDA
tests with no skips::

    SYRIS_REQUIRE_CUDA=1 python -m unittest \
        syris.tests.test_cuda_projection syris.tests.test_cuda_signed_projection \
        syris.tests.test_cuda_projection_variants syris.tests.test_cuda_scheduling -v

This count excludes the mesh-repair and mesh-normal modules in the earlier
study. The new checks compare box decisions exactly and exercise tile coverage,
partial warps, camera types, sampling depths, overflow, and invalid NaNs in
all three precisions. Two representative cases also pass both CUDA memcheck
and synccheck with zero errors. See
``benchmarks/results/validation-scheduling.txt`` for commands and
:doc:`cuda-performance-ledger` for the retained pre-existing Float64 cone
limitation found while constructing the overflow fixture.

Agreement with baseline outputs and analytic cases supports these implementation
changes; it is not a general accuracy certificate. Float32 input quantization and
adaptive sampling's heuristic feature detection retain their existing limits.
