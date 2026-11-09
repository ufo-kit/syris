CUDA projection checks
======================

Run the focused regression suite with a CUDA device and CuPy installed::

    SYRIS_REQUIRE_CUDA=1 python -m unittest syris.tests.test_cuda_projection -v

Without ``SYRIS_REQUIRE_CUDA=1``, these tests skip when CUDA is unavailable.
They compile the production kernels in all three modes and exercise the Python
BVH builder and projector as well as individual device predicates.

The references are analytic: shared triangle edges, box intersection lengths,
thin layers, a hollow solid, and pixel integrals through a wedge and slab.
Additional cases cover hit-list overflow, transformed normals, sorted BVH
leaves, conservative box clipping, and singleton trees. Checking BVH structure
repeatedly can expose publication failures, but is not a proof of race freedom;
the builder uses device-scope acquire/release publication for that purpose.

Precision modes
---------------

Select a mode before constructing meshes and cameras::

    syris.init(compute_backend="cuda", precision="float32")
    syris.init(compute_backend="cuda", precision="mixed")
    syris.init(compute_backend="cuda", precision="float64")

These are alternative initializations. ``float32`` is the default.
``mixed_precision=True`` and ``double_precision=True`` remain aliases for
``mixed`` and ``float64``; conflicting settings raise ``ValueError``.
OpenCL supports ``float32`` and ``float64`` with its existing projection path.

.. list-table:: CUDA mesh projection
   :header-rows: 1

   * - Mode
     - Geometry, rays and BVH
     - Triangle arithmetic / stored hits
     - Output
   * - ``float32``
     - Float32
     - Float32 with existing double intermediates and edge fallback / Float32
     - Float32
   * - ``mixed``
     - Float32
     - Float64 / Float64
     - Float32
   * - ``float64``
     - Float64
     - Float64 / Float64
     - Float64

All CUDA modes use the original mesh's bounding-box centre as a stable local
origin. Vertices are centred before the host transform; the matching transformed
origin is subtracted from the camera before GPU conversion. Lengths are
normalized with a power of two, preserving significands when scaling normal floating-point
values. The normalized vertices and bounding boxes are cached when building
the BVH. Returned thicknesses retain the configured length unit, and public
mesh and camera poses retain their world coordinates.

Mixed and Float64 modes compute sample positions as offsets from the detector
centre. Parallel, cone-beam, local adaptive and staged adaptive sampling use
the same formula. These are the same intended physical rays; only their
floating-point evaluation changes. Mixed mode retains Float64 hit distances
through sorting, duplicate handling and accumulation: promoting triangle
arithmetic alone can still collapse distinct crossings when storing their
distances in Float32. Float32 mode retains the corner-based ray formula because
the revised formula exposed such a collision in the rounding experiments.

These modes reduce specific rounding errors, but do not guarantee a prescribed
accuracy or recover detail already lost in the input mesh. Float32 is intended
for speed; mixed mode trades more arithmetic and hit-buffer memory for accuracy.
Measure performance on the target GPU and workload. Mesh repair and its optional
``float32_rounding`` step remain separate loading options.

The mesh examples accept the same names::

    python -m examples.mesh_scan_cuda --precision mixed

Public API and coordinate-frame checks can be run with::

    SYRIS_REQUIRE_CUDA=1 python -m unittest syris.tests.test_cuda_precision_modes -v

For direct kernel users, the detector-origin argument denotes its corner in
Float32 mode and its centre in mixed/Float64 mode. BVH arrays now contain local,
normalized coordinates: recover world coordinates with
``local / accelerator.coordinate_scale + accelerator.coordinate_origin``.

Projection assumptions
----------------------

Thickness requires closed surfaces. With normals enabled, normals must point
out of the material, including into cavities. Inconsistent crossing sequences
return NaN. Without normals, crossings use the even/odd rule, so overlapping
solids need an appropriate boundary representation. Mesh transforms also
transform supplied normals; computed normals follow the current vertices.

Shared triangle edges use separately rounded products and consistent fallback
arithmetic. Boundary hit distances are canonicalized so duplicate hits can be
combined without a distance-relative tolerance that would erase thin layers.
Ray origins are moved close to the mesh before measuring thickness. This
reduces cancellation but cannot recover detail already lost when input
coordinates were converted to the selected precision. The legacy triangle and
hit-grouping tolerance arguments remain in the raw kernel interface for
compatibility and no longer control these predicates.

The local hit buffer is a fast path. Rays exceeding its capacity use repeated
BVH traversals to collect the next 32 distinct events at a time. All duplicate
hits at each retained distance are combined before integration. The number of
passes is unbounded, so dense rays are never silently truncated. Unused hit
records are not initialized; only the written prefix is read.

Optional mesh repair
--------------------

Install the optional CPU repair dependency from this checkout::

    uv pip install -e '.[mesh-repair]'

Enable it when loading a mesh::

    syris.init(compute_backend="cuda", double_precision=True)
    mesh = Mesh.from_file(source, trajectory, unit=q.um,
                          use_normals=True, repair=True)

The example exposes the same option::

    uv run python -m examples.mesh_cuda --repair-mesh --double-precision

Repair uses libigl/CGAL Boolean self-union and retains regions with positive
winding number. Supply all shells of one material together, with faces oriented
out of material and into cavities. Correctly oriented cavities, nested material
islands, and small disconnected components are retained. Process distinct
materials separately. The method cannot infer whether an outward-facing nested
shell was intended to describe a void.

Exactly coincident vertices are welded and zero-area triangles are removed;
self-intersections and overlaps are resolved. Open boundaries and inconsistent
face orientation raise ``ValueError``. Repair changes connectivity, discards
input mesh attributes, and recomputes face normals without reorienting cavities.
The input PyVista object or file is not modified.

Calculations use Float64. The output is checked for degenerate faces, oriented
closure and self-intersections, including after conversion to Float32 when
selected. Conversion must also preserve distinct vertex positions. If it
invalidates the result, loading raises ``ValueError``
with instructions to use double precision. Validation covers the loaded
coordinates; later unit conversions and transforms still involve rounding.
Repair is performed once at loading and does not certify arbitrary projections.
The optional Boolean implementation is in libigl's copyleft module; see the
`upstream licenses <https://github.com/libigl/libigl-python-bindings>`_.

Pixel sampling
--------------

``iterations=0`` traces the pixel center. Adaptive sampling uses midpoint
quadrature over accepted subregions, with variation among their corners and
center deciding whether to subdivide. ``abs_tolerance`` is expressed in the
same length units as the returned thickness; ``rel_tolerance`` is dimensionless.
The supported maximum depth is 3 in single precision and 2 in double precision.
Larger depths and invalid tolerances raise ValueError.

These tolerances are refinement heuristics, not bounds on the pixel integration
error: a feature missed by all sampled points can remain undetected. The tests
verify the implemented sampling and analytic examples, not arbitrary-scene
accuracy or watertightness for every representable input.

Pixel scheduling
----------------

CUDA box intersections use scalar component selection instead of temporary
runtime-indexed arrays. The conservative arithmetic and intersection decisions
are retained. Pixel scheduling remains per thread by default.

To assign a compact 4-column by 8-row detector tile to each warp::

    syris.init(compute_backend="cuda", warp_tile_width=4)

Widths 1, 2, 4, 8, 16, and 32 are supported; each tile contains 32 pixels.
Zero selects the ordinary per-thread scheduler. This option is CUDA-only.
Each lane evaluates its own pixel with the original ray coordinates and
adaptive sampling. Completed warps fetch another tile dynamically. Partial
edge tiles are masked without dropping lanes from warp synchronization.
Raw launches with incomplete warps or multidimensional blocks use the ordinary
scheduler. Deferred signed retry queues also retain per-thread assignment.
The experimental staged sampler retains its own work queues.

Tiling is optional because its performance depends on the scene, detector
size, and sampling. It can improve locality while increasing time spent waiting
for the slowest pixel in a warp. See :doc:`cuda-performance-ledger` for measured
cases. The focused scheduling checks are::

    SYRIS_REQUIRE_CUDA=1 python -m unittest syris.tests.test_cuda_scheduling -v

Performance prototypes
----------------------

The default retains the 256-hit list and the local adaptive sampler. An
experimental staged sampler is available with::

    mesh.project(camera=camera, iterations=1, sampling_mode="staged")

It traces shared root corners and pixel centers, repairs corners whose computed
coordinates differ from the original pixel calculation, then refines a compact
pixel queue. It preserves the selected depth, midpoint quadrature, tolerance
checks, and NaN propagation. Deeper levels still use the local cache. It needs
approximately 28 additional bytes per pixel in Float32 or 52 in Float64 and
additional launches, and can be slower. ``iterations=0`` uses the usual center
ray in either mode.

A compile-time ``-DSYRIS_HIT_CAPACITY=32`` option experiments with a small common
list and an out-of-line 256-hit retry before event batching. It is not the default:
frequent retries can outweigh the smaller common list. The benchmark compiles
this variant separately without changing the installed configuration.

The signed-length prototype is available through the benchmark's ``signed``
and ``signed_deferred`` variants, with ``--assume-material-boundary`` required.
For a unit ray direction and outward material normals, it evaluates
``sum(exit distances) - sum(entry distances)`` without sorting. Distances are
rebased close to the scene and summed with compensated Float64 arithmetic.
The triangle predicate, geometry storage, and pixel sampling remain unchanged.

This requires a closed, consistently oriented boundary with material occupancy
zero or one. Cavities and disconnected components are permitted. Overlapping
or duplicate shells are not: the sum counts their multiplicity, even when the
entry and exit counts balance. The flag asserts this contract; it does not
validate a mesh. Ordinary library initialization does not enable this mode.

Exact edge/vertex hits, uncertain normal signs, and inconsistent endpoint counts
use the ordered fallback. ``signed`` falls back within the tracing kernel;
``signed_deferred`` queues affected pixels and repeats their complete sampling
in a separate ordered kernel. The latter supports only the local sampler.
Neither mode assumes that balanced endpoint counts prove valid geometry.

See :doc:`cuda-performance` for the measurements, tradeoffs, and reproduction
commands. These are projection benchmarks; unchanged output on tested scenes
is not a general certificate of geometric or pixel-integration accuracy.
