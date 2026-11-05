CUDA projection checks
======================

Run the focused regression suite with a CUDA device and CuPy installed::

    SYRIS_REQUIRE_CUDA=1 python -m unittest syris.tests.test_cuda_projection -v

Without ``SYRIS_REQUIRE_CUDA=1``, these tests skip when CUDA is unavailable.
They compile the production kernels in both precisions and exercise the Python
BVH builder and projector as well as individual device predicates.

The references are analytic: shared triangle edges, box intersection lengths,
thin layers, a hollow solid, and pixel integrals through a wedge and slab.
Additional cases cover hit-list overflow, transformed normals, sorted BVH
leaves, conservative box clipping, and singleton trees. Checking BVH structure
repeatedly can expose publication failures, but is not a proof of race freedom;
the builder uses device-scope acquire/release publication for that purpose.

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
BVH traversals to enumerate every crossing. Dense rays can therefore be slower;
they are no longer silently truncated.

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
