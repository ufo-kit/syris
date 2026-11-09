Syris
=====

.. image:: https://img.shields.io/badge/Python-3.10+-blue
    :target: https://www.python.org/downloads/

.. image:: https://badge.fury.io/py/syris.png
    :target: http://badge.fury.io/py/syris

.. image:: https://github.com/ufo-kit/syris/actions/workflows/tests.yml/badge.svg?branch=master
    :target: https://github.com/ufo-kit/syris/actions

.. image:: https://readthedocs.org/projects/syris/badge/?version=latest
    :target: http://syris.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/ufo-kit/syris/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/ufo-kit/syris

*Syris* (**sy**\ nchrotron **r**\ adiation **i**\ maging **s**\ imulation) is a
framework for simulations of X-ray absorption and phase contrast dynamic imaging
experiments, like time-resolved radiography, tomography or laminography. It
includes X-ray sources, various sample shape creation possibilities, complex
refractive index lookup options, motion model and indirect detection model
(scintillator combined with a conventional camera). Phase contrast is simulated
by the Angular spectrum method, which enables one to include various optical
elements in the simulation, e.g. gratings and X-ray lenses.

Compute-intensive algorithms like Fourier transforms, sample shape creation and
free-space propagation use OpenCL. Mesh projection also supports CUDA through
CuPy.

There are numerous examples of how to use *syris* described below which ship
directly with the code. Enjoy!


Installation
------------

From this checkout, install with `uv <https://docs.astral.sh/uv/>`_::

    uv sync

This creates ``.venv`` and installs the project in editable mode using
``uv.lock``. Python 3.10 or newer is required; ``.python-version`` selects
Python 3.12 for development. OpenCL execution needs a working vendor OpenCL
runtime in addition to the Python packages.

For CUDA 13 mesh projection and the examples::

    uv sync --extra cuda13 --extra examples
    uv run --extra cuda13 --extra examples python -m examples.mesh_cuda

Use ``cuda12`` instead for CUDA 12. Install only one CUDA extra. The CUDA
projector compiles kernels with ``nvcc``, so a matching CUDA toolkit and an
NVIDIA driver are required; the extra installs the CuPy wheel. The BVH tree
viewer also requires the Graphviz ``dot`` executable.

Development commands::

    uv run ruff check .
    uv run ruff format --check .
    uv run python -m unittest discover -v
    uv run --group docs sphinx-build -b html docs/source docs/build/html
    uv build

Ruff handles linting, import sorting and formatting. Use ``make format`` to
sort imports and format Python files, or ``make lint`` to check both.

The test suite needs an OpenCL runtime; CUDA-specific tests skip when CUDA is
unavailable. Set ``SYRIS_REQUIRE_CUDA=1`` when CUDA tests must run. Use
``uv sync --locked`` to check an existing lockfile, and ``uv lock --upgrade``
when intentionally updating dependencies. Standard ``pip install .`` and
``pip install '.[cuda13]'`` also work.


Usage
-----

The first thing you have to do is to initialize *syris* by the ``syris.init()``
function. After that you only need to do whatever is necessary for your program.
A simple white beam propagation example looks like this:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    import quantities as q
    import syris
    from syris.physics import propagate
    from syris.bodies.simple import make_sphere
    from syris.materials import make_henke

    syris.init()
    energies = np.arange(10, 30) * q.keV
    n = 1024
    pixel_size = 0.4 * q.um
    distance = 2 * q.m
    material = make_henke('PMMA', energies)

    sample = make_sphere(n, n / 4 * pixel_size, pixel_size, material=material)
    image = propagate([sample], (n, n), energies, distance, pixel_size).get()
    plt.imshow(image)
    plt.show()

For more detailed information please see the `reference <https://syris.readthedocs.io/en/latest>`_.


Examples
********

Are stored in the **examples** directory. Run them from the checkout with,
for example, ``uv run --extra examples python -m examples.simple``.

Optional mesh repair
********************

Install libigl and the PyVista loader from this checkout::

    uv sync --extra cuda13 --extra examples --extra mesh-repair

The pinned Python 3.12 environment has prebuilt libigl wheels. On newer Python
versions, libigl 2.6.3 may require a source build.

Use ``Mesh.from_file(..., repair=True)`` to resolve self-intersections and
overlaps within one material. The input must be closed and oriented out of
material, including into cavities. Correctly oriented cavities and small
disconnected components are retained. Open boundaries and invalid repaired
results raise ``ValueError``.

For the CUDA dragon example::

    uv run --extra cuda13 --extra examples --extra mesh-repair \
        python -m examples.mesh_cuda --repair-mesh --double-precision

The same repair and precision flags are available in ``examples.mesh_scan_cuda``,
``examples.visualize_bvh_tree`` and ``examples.visualize_bvh_bboxes``. For example::

    uv run --extra cuda13 --extra examples --extra mesh-repair \
        python -m examples.visualize_bvh_bboxes --input mesh.ply \
        --show-nodes --repair-mesh --double-precision

Repair runs on the CPU in Float64 and validates the result after conversion to
the selected precision. Double precision avoids losing newly created fine
features during loading. The optional CGAL Boolean implementation uses libigl's
copyleft module; see its `licenses
<https://github.com/libigl/libigl-python-bindings>`_.

Select CUDA mesh precision with ``syris.init(compute_backend="cuda",
precision="float32")``, ``precision="mixed"`` or ``precision="float64"``.
All three modes use centred coordinates and power-of-two scaling. Mixed mode
keeps Float32 geometry, BVH and output images, with Float64 triangle arithmetic
and hit distances. Mixed and Float64 modes also calculate rays relative to the
detector centre. See ``docs/source/usage/cuda-projection.rst`` for their scope.

For mixed precision with optional repair and bounded Float32 rounding::

    uv run --extra cuda13 --extra examples --extra mesh-repair \
        python -m examples.mesh_cuda --repair-mesh --float32-rounding --precision mixed

The mesh-loading options are ``Mesh.from_file(..., repair=True,
float32_rounding=True)``. Bounded rounding remains experimental and independent
of projection precision. ``--mixed-precision`` and ``--double-precision`` remain
aliases for the named modes. Float64 cannot be combined with Float32 rounding.

Bounded rounding tries the two Float32 values bracketing affected coordinates
without changing the repaired connectivity. It checks for merged vertices,
collapsed or flipped triangles, closure and self-intersections. It raises
``ValueError`` if geometry collapses or the bounded search cannot find a valid
result. Search limits are 64 movable vertices, 2048 nearby triangles and 2048
candidate evaluations. No faces or small components are discarded to force
Float32 output.

Validation covers the returned mesh coordinates. Later vertex transformations,
including rotations and unit conversions, can introduce new rounding defects.
For rigid scans, keeping the mesh fixed and rotating the camera avoids repeatedly
rounding transformed vertices. Mixed mode reduces specific intersection and hit
rounding errors; it does not guarantee a valid or more accurate result for every
ray. Unresolved crossings can still produce NaN. Float32 remains the default.


Citation
--------

If you use this software for publishing your data, we kindly ask to cite the article below.

Faragó, T., Mikulík, P., Ershov, A., Vogelgesang, M., Hänschke, D. & Baumbach,
T. (2017). J. Synchrotron Rad. 24, https://doi.org/10.1107/S1600577517012255
