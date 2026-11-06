Profiling GPU Code
==================

CUDA / CuPy
-----------

CuPy's ``RawModule`` kernels can be profiled directly with NVIDIA Nsight.
``syris.init(profiling=True)`` controls the older OpenCL profiler below; it is
not the switch for CUDA profiling.

Use **Nsight Systems** (``nsys``) to locate time spent in kernels, transfers,
launches, and synchronization. Use **Nsight Compute** (``ncu``) to investigate
the selected kernel's arithmetic throughput, memory/cache traffic, occupancy,
and warp stalls. A timeline cannot by itself separate BVH traversal from
triangle testing inside the projection kernel.

The profiling driver keeps initialization, compilation, BVH construction, and
warmup outside CUDA profiler start/stop. NVTX ranges identify the sampling depth
and repeat. It synchronizes the actual launch stream, downloads results after
capture, and compares the last captured image at each depth with warmup.
Run from the checkout with the CUDA Python environment::

    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --output=/tmp/syris-projection \
        python -m benchmarks.profile_cuda_projection \
        --mesh /path/to/mesh.ply --size 1024 --camera parallel \
        --depths 0 1 --repeats 3 --output /tmp/syris-projection.json

Inspect the resulting ``.nsys-rep`` in ``nsys-ui``, or export summaries::

    nsys stats --report cuda_gpu_kern_sum,nvtx_kern_sum,cuda_api_sum \
        /tmp/syris-projection.nsys-rep

Then collect counters for one warmed projection, rather than every setup and
CuPy helper kernel::

    CUPY_CACHE_SAVE_CUDA_SOURCE=1 ncu --set detailed \
        --profile-from-start off --clock-control none --cache-control none \
        --kernel-name 'regex:^project_parallel_normals_kernel$' \
        --launch-count 1 --export /tmp/syris-projection-kernel \
        python -m benchmarks.profile_cuda_projection \
        --mesh /path/to/mesh.ply --size 1024 --camera parallel \
        --depths 1 --repeats 1 --line-info --output /tmp/syris-kernel.json

``--line-info`` compiles the same source set as ordinary initialization with
``-lineinfo`` for source correlation. It does not enable the ``-G`` debug build.
``CUPY_CACHE_SAVE_CUDA_SOURCE=1`` retains newly compiled source in the CuPy
cache. For cone-beam profiling use ``--camera cone`` and filter
``project_conebeam_normals_kernel``. ``--precision`` selects ``float32``,
``float64``, or ``mixed``. Add ``--warp-tile-width 4`` to capture the optional
4-column by 8-row scheduler; this is also retained in a line-info build.

Profiler instrumentation and kernel replay can change runtime and cache state.
Use the separate warmed benchmark for speedup measurements. These commands
leave GPU clocks unchanged and request no cache flush between replay passes;
report that policy when comparing counter captures.

If ``ncu`` reports ``ERR_NVGPUCTRPERM``, the driver denies access to performance
counters. A single administrator-authorized profiler run can provide access;
changing system-wide driver settings is not necessary for that option. A sudo
run should use the absolute path to the project's Python interpreter and
explicit cache directories. See `NVIDIA's counter permission instructions
<https://developer.nvidia.com/ERR_NVGPUCTRPERM>`_ and `CuPy's profiling guide
<https://docs.cupy.dev/en/stable/user_guide/performance.html#in-depth-profiling>`_.

First local capture (2026-11-06)
--------------------------------

Nsight Systems 2025.6.3 successfully captured CuPy 13.6.0 on the RTX 2000 Ada.
The case was Armadillo (345,944 triangles), a 1024-square parallel detector at
37 degrees, Float32, two warmups and three captured repeats at depths 0 and 1.

.. list-table:: Instrumented timeline, not a standalone speed benchmark
   :header-rows: 1

   * - Sampling
     - Median projection kernel
     - Median projection range
     - Projector share of GPU kernel time
   * - Center
     - 6.99 ms
     - 7.93 ms
     - 94.2%
   * - Adaptive depth 1
     - 73.01 ms
     - 74.04 ms
     - 99.4%

The first captured range is longer than later repeats. The combined capture
attributes 99.0% of GPU kernel time to ``project_parallel_normals_kernel``.
The long CPU ``cudaEventSynchronize`` calls are waits for GPU execution; their
time must not be added again as independent CPU overhead. Captured outputs
checked after profiling match warmup exactly and are finite.

Metadata and CSV summaries are in ``benchmarks/results/profiling``. The local
trace is ``/tmp/syris-nsys-armadillo-float32.nsys-rep``. The first Nsight Compute
attempt was blocked by ``ERR_NVGPUCTRPERM`` with ``RmProfilingAdminOnly: 1``.
Running ``benchmarks/results/profiling/run-ncu-local.sh`` from an authenticated
terminal subsequently collected the counters below.

Counter capture (2026-11-06)
----------------------------

Nsight Compute 2026.1.0 captured one warmed Float32 Armadillo projection at
adaptive depth 1 with line information. Its 25 replay passes completed, and
the checked output matched warmup exactly with no nonfinite pixels. The
14.8-second instrumented host range includes profiling and replay overhead;
the reported kernel duration was 72.08 ms. Neither is a new speedup benchmark.

.. list-table:: Hardware counters for this capture
   :header-rows: 1

   * - Metric
     - Value
     - Interpretation
   * - Average active threads per warp
     - 6.47 of 32; 5.81 after predication
     - Many instruction lanes are inactive.
   * - Achieved occupancy
     - 31.79%; theoretical limit 33.33%
     - 128 registers per thread limit residency to two blocks per SM.
   * - Scheduler cycles without an eligible warp
     - 69.68%
     - Resident work often cannot issue its next instruction.
   * - Long-scoreboard waits
     - About 51.6% of average warp cycles between issued instructions
     - Dependencies on local/global memory operations are prominent.
   * - L2 throughput / DRAM throughput
     - 82.75% / 1.75% of their respective peak sustained rates
     - Cache and access-pattern costs deserve attention; DRAM is lightly used.
   * - L1 / L2 hit rate
     - 84.71% / 99.41%
     - High cache hit rates coexist with significant access costs.

Source correlation identifies two concrete experiments:

* ``WatertightRay.cu:161-168`` constructs ``b[6]`` and ``d[3]`` and indexes them
  with runtime axis choices for every box test. These lines account for 76.94%
  of source-instrumented theoretical local-memory sectors. The generated SASS
  contains local stores and loads even though the register-spill request metric
  is zero. Test scalar component selection while preserving the axis order,
  directed rounding, and zero-direction handling of the conservative predicate.
* Bounding-box loads account for 78.81% of excessive theoretical global-memory
  sectors. Including rope and child accesses raises the BVH-node share to
  96.19%. This supports investigating coherent ray scheduling, node layout, and
  conservative shared candidates. A candidate list must cover every ray using
  it; per-ray intersection tests and sampling accuracy still need to be retained.

These are source locations and access-pattern estimates, not exclusive runtime
fractions. In particular, the leading wait mapped to box-array construction is
an instruction consuming the preceding global bounding-box loads. The 54.03%
of long-scoreboard samples mapped to the array lines cannot all be charged to
local arrays. Theoretical sectors do not measure physical traffic after cache
filtering. Likewise, low average lane activity does not establish a 32/6.47
speedup, and Nsight's automatic speedup suggestions must not be added together.
No 10x speedup follows from this capture.

At analysis time, the source hashes matched the captured hashes. The source
line numbers above describe that capture, before the subsequent optimizations.
The CuPy-generated source
was recovered from its saved cache and checked against the concatenated source
files before mapping line numbers. Evidence under
``benchmarks/results/profiling/armadillo-float32-d1-ncu*`` includes metadata,
the details and raw counter exports, source-aggregated metrics, mapping hashes,
and SASS around the eight leading sampled memory-dependency waits. The binary
report remains at ``/tmp/syris-ncu-armadillo-float32-d1-lineinfo.ncu-rep``.

This is one mesh, view, precision, and sampling depth. Both cache and clock
control were disabled, so cross-pass cache state and variable clocks limit
precise comparisons. Confirm any optimization with the separate warmed
benchmarks and numerical checks. For a subsequent cache-sensitive comparison,
application replay with the same explicit warmup can reproduce cache priming
on each pass; see `NVIDIA's cache-control guidance
<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#cache-control>`_.

Candidate solutions
-------------------

Scalar selection and warp tiles have since been implemented; see
:doc:`cuda-projection` for usage and :doc:`cuda-performance-ledger` for results.
The design below records their rationale and the remaining proposals. SAH and
node-layout changes remain unimplemented.
The current code distinguishes four opportunities: remove avoidable per-node
work, reduce the number of visited nodes, improve memory access locality, and
keep more warp lanes doing useful work. A single change need not improve all
four, and higher occupancy alone is not the objective.

The adopted scalar predicate replaces the runtime-indexed ``b[6]`` and
``d[3]`` arrays with inlined component selection. It selects the two box
components once per axis and retains the existing near/far indices, Kx/Ky/Kz
evaluation order, short-circuit behavior, directed arithmetic, and signed-zero
handling. Constructor and triangle-predicate arrays are unchanged. The first
prototype selected each plane separately and increased mixed-precision register
use enough to regress some views; the revised selector avoids that threshold.

Morton ordering, SAH, and physical layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BvhCupyAccelerator.build`` sorts triangle-centroid Morton codes, leaf bounds,
and a triangle-index permutation. Vertex and normal buffers retain their input
order. Internal bounds, child indices, and ropes occupy separate arrays;
leaves and internal nodes occupy separate index ranges. Morton locality of
centroids therefore does not ensure contiguous node or triangle accesses by
the rays currently executing together. The default projector assigns pixels by
independent per-thread ``atomicAdd`` calls, without a guarantee that warp lanes
keep neighboring pixels as work progresses. The optional tile scheduler gives
lanes a defined detector neighborhood.

SAH addresses tree quality: which primitives should share a subtree to reduce
expected traversal and intersection work. It is a reasonable alternative to
the current Morton-based topology. A binned SAH builder with the same single-
triangle leaves is a useful comparison because it can retain the existing
leaf/internal index convention, triangle mapping, and traversal kernel. It
must regenerate child links and escape ropes for its topology and preserve
conservative bounds. Keep the node storage format fixed for this comparison.
Morton treelets joined by SAH, or SAH-based treelet optimization after the
current build, are later ways to limit construction cost. See `PBRT's BVH
construction discussion
<https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies>`_
and `Karras and Aila's tree optimization work
<https://research.nvidia.com/publication/2013-07_fast-parallel-construction-high-quality-bounding-volume-hierarchies>`_.

Our projector needs every crossing, so nearest-hit termination and shrinking
the search interval after an early hit do not apply. SAH can still avoid empty
subtrees, but its usual surface-area probability model is not tailored to a
particular parallel or cone-beam view. A later view-specific heuristic could
weight projected node footprints or measured ray visits; that is a proposal
requiring its own validation. Compare actual node/triangle visits and runtime,
not just an SAH score, and include build cost over the number of views sharing
the tree. A changed traversal order must retain canonical event grouping and
the current ordered accumulation; reordering signed sums can change rounding.

Separately, test a compact traversal-oriented node layout and geometry reordered
into leaf order, with normals and original triangle identities remapped
consistently. Node layout is the higher-priority memory experiment for this
capture because node fields dominate its excess global-sector metric. Repacking
does not guarantee coalescing when neighboring lanes traverse different nodes.
Avoid changing bound precision or adding spatial-split duplicate primitives in
the initial experiments.

Detector tiles and warp activity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented scheduler lets one warp dynamically claim a compact tile of
32 pixels. Each lane processes its assigned pixel using the original ray
generation and adaptive sampler; completed warps claim another tile. The
initial comparison screened 8 by 4, 4 by 8, 16 by 2, and 32 by 1. This retains
dynamic balancing between warps and gives a defined spatial relationship
between their lanes.
It changes scheduling only, with no coarsening of the detector or sharing of
sample values. Divergent per-pixel refinement and waiting for the slowest lane
can still outweigh the coherence gain. The final 4-column by 8-row comparison
improves ordinary mesh projections by 1.12–1.48x across the three precisions
when combined with scalar selection, relative to the already optimized indexed
projector. These are separate warmed measurements, not inferred from counters.

If tiling helps, a separate packet traversal experiment can share node loads
and use a per-ray hit mask. A shared candidate set must conservatively cover
all participating rays; triangle predicates still run per ray, and lists must
have a complete overflow path. Large or divergent packets can perform more
total work. Ordering packets in detector Morton order is another independent
option; it serves ray scheduling, separately from the geometry's Morton sort.

The 6.47 active lanes per instruction and 31.79% occupancy describe different
effects. Tiles target agreement between lanes' work. Occupancy depends on
registers and other per-block resources; it needs reductions in live state or
carefully chosen kernel boundaries. Blind register caps can introduce spills.
The existing shared-root/refinement queue was slower on the recorded mesh
cases, and block-size/depth-specialization changes gave no consistent benefit.
An individual-ray queue separating traversal from adaptive bookkeeping would
be a different, more costly design; investigate it only if subsequent profiles
justify its queue traffic and synchronization. Preserve original sample
coordinates and per-pixel accumulation order. See the existing ablation results
in :doc:`cuda-performance-ledger` and `Aila and Laine's work-distribution study
<https://research.nvidia.com/publication/2009-08_understanding-efficiency-ray-traversal-gpus>`_.

Scalar selection and warp tile scheduling have been evaluated in that order.
SAH and node layout remain deferred; compare them independently against this
baseline before combining changes.
Collect diagnostic visit counts separately from timing runs, and compare
center/adaptive sampling, parallel/cone cameras, multiple views and meshes,
and each supported precision. Numerical checks must include shared boundaries,
nearly parallel rays, thin features, and hit-buffer overflow. None of these
changes may reduce the sample set or discard a required crossing.

OpenCL profiler
---------------

.. automodule:: syris.profiling
    :members:
