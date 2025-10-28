# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""
OpenCL, logging and precision configuration. This module also defines global variables which
hold the configuration objects for precision and OpenCL. Furthermore, pmasf path is specified
here and caching policy for data as well.
"""

import logging
import numpy as np
import pyopencl as cl
import pyopencl.cltypes as cltypes


LOG = logging.getLogger()
MAX_META_BODIES = 30


class Precision(object):
    """A precision object holds the precision settings of the floating point and complex numpy and
    OpenCL data types. If *double* is True, double precision is used.
    """

    def __init__(self, double=False):
        self.set_precision(double)

    def is_single(self):
        """Return True if the precision is single."""
        return self.cl_float == 4

    def set_precision(self, double):
        """If *double* is True set the double precision."""
        if double:
            self.cl_float = 8
            self.cl_cplx = 16
            self.np_float = np.float64
            self.np_cplx = np.complex128
        else:
            self.cl_float = 4
            self.cl_cplx = 8
            self.np_float = np.float32
            self.np_cplx = np.complex64
        self.numpy_to_opencl = {
            self.np_float: self.cl_float,
            self.np_cplx: self.cl_cplx,
        }
        self.opencl_to_numpy = dict(
            list(
                zip(
                    list(self.numpy_to_opencl.values()),
                    list(self.numpy_to_opencl.keys()),
                )
            )
        )

        self.float4 = np.dtype(
            {
                "names": ["x", "y", "z", "w"],
                "formats": [self.np_float] * 4,
            }
        )

        self.float2 = np.dtype(
            {
                "names": ["x", "y"],
                "formats": [self.np_float] * 2,
            }
        )

        self.float3 = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [self.np_float] * 3,
            }
        )

        self.uint2 = np.dtype(
            {
                "names": ["x", "y"],
                "formats": [np.uint32] * 2,
            }
        )

        dtype_base = "double" if double else "float"
        for i in [2, 3, 4, 8, 16]:
            setattr(self, "vfloat" + str(i), getattr(cltypes, dtype_base + str(i)))


class OpenCL(object):
    """OpenCL runtime information holder."""

    def __init__(self):
        self.ctx = None
        self.queues = []
        self.devices = []
        # Default command queue
        self.queue = None
        self.programs = {
            "improc": None,
            "physics": None,
            "geometry": None,
            "mesh": None,
        }
        # {command queue: {shape: plan}} dictionary
        self.fft_plans = {}


class RayCasting:
    """
    Holds epsilon and tolerance values for raycasting kernels.

    This class centralizes all the 'magic numbers' used for numerical
    stability and geometric robustness in the raycaster.
    """

    def __init__(self, double=False, dynamic_range=10):
        # --- Helper variables to set defaults based on precision ---
        if double:
            # 64-bit (double) precision uses smaller values
            default_abs_eps = 1e-15
            default_rel_eps = 1e-13
            default_floor = 1e-100
        else:
            # 32-bit (float) precision needs larger tolerances
            default_abs_eps = 1e-7
            default_rel_eps = 1e-5
            default_floor = 1e-10 / dynamic_range

        # --- Parameters for Watertight Ray-AABB Intersection ---

        # A numerical tolerance for the watertight ray-AABB intersection algorithm.
        # This number is derived from numerical analysis (5.0 * 2^-24)
        # Used for conservative BVH descent
        self.p_ray_box_epsilon = 5.0 * (2**-24)

        # --- Parameters for Watertight Ray-Triangle Intersection ---

        # A GEOMETRIC distance. Any hit closer than this (t < tmin) is discarded.
        # This is the primary value used to prevent self-intersection ("shadow acne").
        self.p_tri_ray_tmin = default_abs_eps

        # A NUMERICAL threshold. The 32-bit determinant is considered 'effectively zero'
        # if its absolute value is smaller than this. This triggers the 64-bit fallback.
        self.p_tri_abs_min_error = default_floor

        # Scales `p_tri_abs_min_error` to calculate the 32-bit *relative* error bound.
        # The final 32-bit bound is max(relative_bound, p_tri_abs_min_error).
        self.p_tri_gamma_multiplier = 256.0

        # --- Parameters for 64-bit (Double Precision) Fallback ---

        # Scales the 64-bit machine epsilon to calculate the 64-bit *relative* error bound.
        self.p_tri_d_gamma_multiplier = 24.0

        # A NUMERICAL threshold. The 64-bit determinant is considered 'effectively zero'
        # if its absolute value is smaller than this. It's an "almost-zero" floor.
        self.p_tri_d_abs_min_error = 1e-100

        # --- Parameters for Thickness Calculation ---

        # Used in the `match_pairs` (normal-based) kernel.
        # Defines the absolute component of the dynamic tolerance
        # for grouping nearly-coincident hits: max(abs_eps, rel_eps * t).
        self.p_group_abs_epsilon = default_floor

        # Defines the relative component of the dynamic tolerance.
        self.p_group_rel_epsilon = default_rel_eps

        # Used in the `traceRay` (no-normal) kernel.
        # Defines the tolerance for de-duplicating sorted t-values
        # in `unique_from_sorted_with_epsilon`.
        self.p_unique_abs_epsilon = default_abs_eps


def init_logging(level=logging.DEBUG, logger_file=None):
    """Initialize logging with output to *logger_file*."""
    LOG.setLevel(level)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=fmt)

    if logger_file:
        file_handler = logging.FileHandler(logger_file, "a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        LOG.addHandler(file_handler)


PRECISION = None
OPENCL = None
BACKEND = None
UNIT = None

# Refractive index calculation program path.
PMASF_FILE = "pmasf"

# OpenCL functions which are wrapped for profiling if profiling is enabled.
PROFILED_CL_FUNCTIONS = [cl.enqueue_nd_range_kernel, cl.enqueue_copy]

# Caching constants.
CACHE_HOST = 1
CACHE_DEVICE = 2
DEFAULT_CACHE = CACHE_HOST
