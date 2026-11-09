CUDA rounding experiments
=========================

The recorded experiments reproduce the four audited Coarse Bunny counterexamples
and compare coordinate preparation with higher-precision triangle arithmetic.
They use the existing benchmark repository's prepared meshes. The original
``probe``, ``validate``, ``investigate`` and ``wide_hits`` drivers patch scratch
copies of the pre-integration sources recorded in the result hashes. They require
that source snapshot and refuse to run against the integrated implementation.

For the current public API, run from this checkout with CUDA, PyVista and tifffile::

    python -m benchmarks.rounding_precision.public_modes \
        --benchmark-root /path/to/syris-benchmarks \
        --references /tmp/precision-references --write-references \
        --output /tmp/precision-modes.json

This checks all three modes, with synchronized warmed BVH-build and projection
timings. Projection includes GPU-to-host transfer and metre conversion. The
Float64 references use the same prepared Float32 input geometry; selected rays
are independently checked with CPU plane intersections. To compare against a
pre-change source snapshot, first run with ``--legacy-root /path/to/snapshot``
and ``--write-references``, then run the current implementation against those
same reference files without ``--write-references``. Keep separate output files.

The historical source-patching commands, for their matching source snapshot, were::

    python -m benchmarks.rounding_precision.probe \
        --benchmark-root /path/to/syris-benchmarks --output /tmp/rounding-initial.json
    python -m benchmarks.rounding_precision.validate \
        --benchmark-root /path/to/syris-benchmarks --output /tmp/rounding-validation.json
    python -m benchmarks.rounding_precision.investigate \
        --benchmark-root /path/to/syris-benchmarks --output /tmp/rounding-regressions.json
    python -m benchmarks.rounding_precision.wide_hits \
        --benchmark-root /path/to/syris-benchmarks --output /tmp/rounding-wide-hits.json
    python -m unittest benchmarks.rounding_precision.test_references

The benchmark repository must contain prepared baseline Bunny and Armadillo
Float32 NPZ files, their FOV metadata, the counterexample audit, and the retained
Bunny CUDA Float32 TIFF. There are no downloads. The probe requires exact
agreement between the current baseline and that TIFF before interpreting results.

Recorded experiment, 2026-11-09
-------------------------------

Results are in ``benchmarks/results/rounding/2026-11-09/``. Each JSON records
source and input hashes, the GPU and software versions, and the command used.
The GPU was an NVIDIA RTX 2000 Ada Generation.

The original four-pixel image was reproduced bit for bit. Absolute errors in nm:

================  ==========  ============================
Pixel (row, col)   Current     Centred + power-of-two scale
================  ==========  ============================
(636, 578)        1.354903      0.093605
(989, 1485)       2.351642      0.258380
(1165, 1304)      1.018484      0.101244
(1395, 603)       6.956988      0.104329
================  ==========  ============================

All four pass the original tolerance after this coordinate change. At the last
pixel, intersecting the actual encoded GPU coordinates with 90-digit arithmetic
gives a 6.957413 nm error against the intended rays/geometry. The remaining
arithmetic and output-conversion contribution is approximately 0.000425 nm.
Higher-precision triangle arithmetic cannot undo those earlier coordinate errors.

The full-frame validation covers Bunny at 2048 squared and Armadillo at 512
squared, each at 0, 17, 45, 90 and 135 degrees. Every mode, including the Float64
comparison, starts from the SAME prepared Float32 coordinates. The tolerance is
``1 nm + 0.01% * abs(reference thickness)``.

================================  ====================  ====================
Variant                           Exceeding pixels      Maximum error (nm)
================================  ====================  ====================
Current                           102                   10.124
Mixed arithmetic only             102                   10.084
Power-of-two scale only            53                    692.393
Centred + power-of-two scale        13                    9.601
Centred + power-of-two + mixed      13                    9.598
Centred + power-of-two + new rays   9                     54219.814
================================  ====================  ====================

Counts sum over ten images, or 22,282,240 detector pixels per variant. They are
comparison results for these views, not population error rates. The full-image
comparison uses Float64 output, independently checked with CPU plane intersections
on 3,300 sample rays and the 16 largest-error pixels for each frame/variant.
The largest checked reference discrepancy was below 0.00000005 nm. Exact-edge
and odd-crossing cases in the CPU reference are explicitly ambiguous. Separate
analytic tests cover units, disjoint intervals, oblique paths and edge ambiguity.

The centred/power-of-two candidate had no nonfinite, negative, missing-foreground
or extra-positive pixels under the recorded checks. It removes every tolerance
exceedance in the original 0-degree Bunny frame, but leaves 13 elsewhere. Its
count also increases from six to eight in the 90-degree Bunny frame. It is an
improvement on this sample, not a guarantee for arbitrary views or meshes.

Median warmed projection latency was approximately 9.46 versus 9.24 ms for Bunny,
and 3.24 versus 3.32 ms for Armadillo (current versus centred/power-of-two).
These are preliminary timings from three repetitions per view; differences this
small should not be interpreted as a speedup. Timings include GPU-to-host transfer
and metre conversion, exclude pose/BVH construction, and do not measure TIFF I/O.

What the variants change
------------------------

``power2`` replaces the arbitrary normalization factor by a power of two. For the
normal floating-point values in these experiments, this avoids another coordinate
rounding at scaling. ``centered_power2`` also removes the common positive detector
offset from BOTH mesh and camera, preserving the intended physical rays while
keeping coordinates nearer zero. Neither operation repairs or changes the
intended mesh surface. Rotated vertices and detector coordinates still undergo
Float32 conversion, so finite errors remain.

``mixed`` uses the existing CUDA mixed-precision triangle implementation.
Geometry, rays and stored hit distances remain Float32. The experimental new ray
formula computes offsets about the detector centre before adding the centre
position. It improves many ordinary errors but exposes a severe crossing-pairing
failure, so it is not a recommended default.

Adverse cases retained
----------------------

``investigate`` reproduces two regressions and captures their GPU inputs and
triangle hits. Distinct intersections can round to the same Float32 distance;
the paired-crossing path then merges them, changing crossing parity. Its odd-count
handling interprets the ray as starting inside material and can return a large
finite thickness error. This is separate from the original nanometre-scale
coordinate errors.

At Bunny 45 degrees, pixel (1291, 652), the new-ray variant returns 57.321471 um
instead of 3.101657 um. High-precision intersection of its encoded inputs gives
3.101656 um. Mixed triangle arithmetic still returns the bad thickness because
the hit distances are stored in Float32. At 90 degrees, pixel (1138, 1220), the
scale-only variant gives a 692 nm error; mixed arithmetic resolves that particular
collision. See ``regressions.json`` for the actual hit records.

Float64 hit-storage follow-up
-----------------------------

``wide_hits`` tests a further isolated prototype: centred coordinates, power-of-two
scaling, the new ray formula, mixed triangle arithmetic, and Float64 hit distances
through sorting, deduplication and accumulation. Geometry, rays, BVH and output images
remain Float32. The copied CUDA source changes the hit-record field, triangle
distance output, event records and accumulation state; production files are not
modified.

It passes all 25 existing ``ProjectionCases`` CUDA regression tests, covering
shared edges, thin layers, cavities, clipping, normals, overflow handling,
adaptive sampling and cone-beam paths. Across the same ten mesh/view combinations,
tolerance exceedances fall to eight, with maximum error 5.787 nm and no nonfinite,
negative, missing-foreground or extra-positive pixels in the recorded checks.
The Float64 image comparisons are independently checked on another 2,720 CPU
plane-intersection sample evaluations, including each image's largest errors.

The 45-degree regression pixel now returns 3.101655693 um against the reference
3.101656740 um, an error of approximately 0.00105 nm. This isolates the harmful
Float32 hit-distance collision: increasing only triangle arithmetic precision
had left the 57.321471 um result unchanged.

The broader change costs time: median warmed latency was about 16.72 ms for Bunny
and 4.41 ms for Armadillo, versus approximately 9.46 and 3.24 ms for the baseline
in the earlier run. These cross-run timings are preliminary, not a controlled
speedup claim. See ``wide_hits.json`` for raw measurements and precision scope.

These initial experiments established two useful directions: centred/power-of-two coordinates
give a low-cost improvement; retaining more precision through hit processing
prevents the demonstrated crossing collision and improves the combined result
further. Eight exceedances remain, so neither prototype establishes a universal
error guarantee.

Integrated public modes
-----------------------

``public-baseline.json`` and ``public-modes.json`` compare a preserved source
snapshot with the integrated ``syris.init(precision=...)`` API. The same ten
views and prepared Float32 inputs give the following results at the declared
1 nm + 0.01% thickness tolerance:

.. list-table:: Full-frame comparison against pre-change Float64 output
   :header-rows: 1

   * - Configuration
     - Pixels exceeding tolerance
     - Maximum error (nm)
   * - Previous Float32
     - 102
     - 10.124
   * - Improved Float32
     - 13
     - 9.601
   * - Mixed
     - 8
     - 5.787
   * - Float64
     - 0
     - 0.000000048

There are no nonfinite or negative pixels, and no missing/extra foreground
pixels under the recorded 0.1 nm foreground cutoff. Each run includes 1,440
CPU plane-reference checks, including the largest-error pixels for each
frame/mode. The largest reference discrepancy is below 0.00000005 nm.
The four original Bunny pixels all pass in improved Float32. The known
45-degree crossing-collision pixel passes in mixed mode at 3.101655693 um.

The stable origin is the original mesh's bounding-box centre, transformed with
the body. Centred host vertices are cached, and normalized GPU vertices/bounds
are reused between projections. Computing a new centre from each rotated bbox
was less accurate in Float32 on this sample and was not retained. All parallel,
cone-beam, local adaptive and staged adaptive paths share the detector formula.
Mixed mode carries Float64 hit distances through the bounded and overflow paths.

Median warmed projection times including GPU-to-host transfer were:

.. list-table:: Milliseconds on RTX 2000 Ada, seven repeats per view
   :header-rows: 1

   * - Mesh
     - Previous Float32
     - Improved Float32
     - Mixed
     - Float64
   * - Bunny, 2048 squared
     - 9.46
     - 9.45
     - 15.74
     - 43.69
   * - Armadillo, 512 squared
     - 3.15
     - 2.70
     - 3.81
     - 7.28

Warmed Float32 BVH-build medians were 1.01 to 1.07 ms for Bunny and 15.90 to
13.98 ms for Armadillo. These sequential runs are preliminary timings, not a
guarantee of zero overhead or a controlled speedup claim. They exclude initial
mesh preparation, JIT and disk I/O; the fixed centred host geometry also uses
additional memory. The ten-view sample leaves finite errors in both Float32
and mixed mode and does not replace the wider benchmark campaign.

The integrated checks passed 260 tests across projection predicates, overflow,
signed/deferred/depth-specialized kernels, precision selection, scene transforms,
example outputs, cameras, OpenCL precision and optional repair. The broader run
also encountered one existing failure: ``TestCamera.test_constructor`` constructs
a camera with ``shape=None``, which fails while computing viewport dimensions.
The same error was reproduced on the preserved pre-change source; it is unrelated
to the precision modes. Counts and log hashes are in ``public-validation.json``.
