Examples
========

In this section we describe the examples which come directly with *syris* in
order to demonstrate its usage.


Composite body
--------------

**composite_body.py**

.. automodule:: examples.composite_body
    :members:


Edge enhancement
----------------

**edge_enhancement.py**

.. automodule:: examples.edge_enhancement
    :members:


Energy Filter
-------------

**energy_filter.py**

.. automodule:: examples.energy_filter
    :members:


Experiment
----------

**experiment.py**

.. automodule:: examples.experiment
    :members:


Fresnel Propagation
-------------------

**propagator.py**

.. automodule:: examples.propagator
    :members:


Fresnel Propagation Accuracy
----------------------------

**fresnel.py**

.. automodule:: examples.fresnel
    :members:


Mesh Bodies
-----------

**mesh.py**

.. automodule:: examples.mesh
    :members:


Laminography of Samples Defined by Meshes
-----------------------------------------

**mesh_scan.py**

.. automodule:: examples.mesh_scan
    :members:


Comparing OpenCL and CUDA mesh examples
---------------------------------------

``mesh`` / ``mesh_cuda`` and ``mesh_scan`` / ``mesh_scan_cuda`` share their
geometry, argument parsing and output code. The CUDA modules select the backend.
Both load the same triangles, center the bounding box by default, rotate the
mesh in a fixed detector frame, and save thickness in **metres**. Float32 is the
default; ``--double-precision`` preserves Float64 through TIFF output.

The default mesh is a built-in cube, with no download. Both single-mesh examples
default to 256 by 256 pixels; ``--n-x`` and ``--n-y`` override the square ``--n``.
Automatic fields of view enclose rotation about the chosen mesh origin. Use
identical explicit pixel sizes for controlled comparisons. ``mesh`` pixel sizes
default to micrometres (overridden by ``--pixel-size-units``); ``mesh_scan`` pixel
sizes are in nanometres. Mesh coordinates use ``--mesh-units`` (default ``um``).

``--supersampling s`` now means the same thing in all four examples: trace a
uniform s-by-s grid of pixel centers, transfer the result, then average on the
CPU. The default is one ray per pixel. This replaces the earlier overloaded
meaning of this option in ``mesh`` (OpenCL median sampling) and ``mesh_cuda``
(CUDA adaptive refinement). Legacy median sampling remains available through
``--supersampling-projection`` on OpenCL; CUDA refinement uses ``--subsampling``.
Keep both at their defaults for comparisons. Cone beams, mixed precision and
``--use-normals`` are CUDA-only modes and are not equivalent OpenCL workloads.

From a checkout with the CUDA and example dependencies installed::

    mkdir -p comparison/opencl comparison/cuda
    uv run --no-sync python -m examples.mesh --n 256 --x-rotate 23 \
        --y-rotate 17 --num-y-rotations 4 --double-precision --no-show \
        --projection-filename comparison/opencl/projection
    uv run --no-sync python -m examples.mesh_cuda --n 256 --x-rotate 23 \
        --y-rotate 17 --num-y-rotations 4 --double-precision --no-show \
        --projection-filename comparison/cuda/projection
    uv run --no-sync python -m examples.compare_mesh_outputs \
        comparison/opencl comparison/cuda

For scans, the same command arguments work with either entry point::

    uv run --no-sync python -m examples.mesh_scan --input sample.ply \
        --n 256 --pixel-size 750 --num-projections 20 --lamino-angle 23 \
        --supersampling 2 --out-directory scan/opencl
    uv run --no-sync python -m examples.mesh_scan_cuda --input sample.ply \
        --n 256 --pixel-size 750 --num-projections 20 --lamino-angle 23 \
        --supersampling 2 --out-directory scan/cuda
    uv run --no-sync python -m examples.compare_mesh_outputs scan/opencl scan/cuda

Scan angles exclude the endpoint: four views over 180 degrees are 0, 45, 90 and
135 degrees. One-view scans are supported. Pixel-shift retries are disabled by
default; ``--test-pixels`` opts in and can change the workload and sample positions.
Multiple devices receive disjoint projection indices. Ground-truth slices
(``--make-gt`` / ``--compute-slice``) use the existing OpenCL slicer in both entry
points and are not a CUDA/OpenCL projection benchmark.

The logs identify the backend, device, precision and sampling. One warm-up
projection is excluded by default (``--warmup`` changes this). Reported projection
time includes pose changes, accelerator rebuilding, tracing, host transfer and
averaging, but excludes file writing and display. Both examples rotate the mesh
and rebuild the accelerator for each measured view, including the first. These
are application timings, not isolated kernel timings or timings for CUDA camera
motion with a cached, static BVH.

Matching inputs does not guarantee identical floating-point results. The old
OpenCL intersection predicates can miss rays on shared triangle edges even in
double precision. For example, the unrotated default cube at ``--n 48`` produced
zero-thickness pixels on face diagonals where CUDA returned the correct 2e-6 m
thickness. The ray-casting implementations are unchanged by these examples.
The comparison command checks filenames, shapes, dtypes, finite values and
every pixel, reports the maximum absolute difference, and exits with status 1
if any check fails. Its default tolerances are ``rtol=1e-5`` and ``atol=1e-12`` m;
choose tolerances appropriate to the sample before interpreting timings as an
equal-accuracy comparison.


Metaballs
---------

**metaballs.py**

.. automodule:: examples.metaballs
    :members:


Paganin Phase Retrieval
-----------------------

**paganin.py**

.. automodule:: examples.paganin
    :members:


Platform Benchmark
------------------

**speed.py**

.. automodule:: examples.speed
    :members:


Multi GPU Speedup
-----------------

**multigpu.py**

.. automodule:: examples.multigpu
    :members:


X-ray Source
------------

**source.py**

.. automodule:: examples.source
    :members:


X-ray Source Blur
-----------------

**source_blur.py**

.. automodule:: examples.source_blur
    :members:


Simple
------

**simple.py**

.. automodule:: examples.simple
    :members:


Tomographic Rotation
--------------------

**tomographic_rotation.py**

.. automodule:: examples.tomographic_rotation
    :members:


4D Tomography
-------------

**tomography_4D.py**

.. automodule:: examples.tomography_4D
    :members:


Trajectory
----------

**trajectory.py**

.. automodule:: examples.trajectory
    :members:


Transformation Order
--------------------

**transformation.py**

.. automodule:: examples.transformation
    :members:


Transmission Function Sampling
------------------------------

**transmission_function_sampling.py**

.. automodule:: examples.transmission_function_sampling
    :members:


Linear Shift Dependent Convolution
----------------------------------

**varconvolution.py**

.. automodule:: examples.varconvolution
    :members:
