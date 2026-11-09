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

"""Synchrotron Radiation Imaging Simulation (SYRIS) initialization."""

__version__ = "0.4dev"


def init(
    compute_backend="opencl",
    platform_name=None,
    device_type=None,
    device_index=None,
    profiling=False,
    profiling_file="profile.dat",
    loglevel=None,
    logfile=None,
    double_precision=False,
    unit="um",
    mixed_precision=False,
    warp_tile_width=0,
    precision=None,
):
    """Initialize syris with the best available compute backend.

    ``precision`` selects ``"float32"`` (default), ``"mixed"`` (CUDA only),
    or ``"float64"``. The existing ``double_precision`` and ``mixed_precision``
    switches remain aliases; conflicting settings raise ValueError.

    CUDA centres mesh/camera coordinates before GPU conversion and normalizes
    lengths with a power of two. Mixed mode retains Float32 geometry, BVH and
    images, with Float64 triangle arithmetic and hit distances. Mixed and
    Float64 modes also compute detector samples relative to its centre.
    These reduce rounding errors but cannot recover lost input geometry.

    ``warp_tile_width`` is an optional CUDA pixel scheduler. A positive width
    assigns each warp a width-by-(32/width) detector tile. Zero retains the
    per-thread scheduler; the ray generation and sampling rules are identical.
    """
    from numbers import Integral

    if precision is not None:
        if precision not in ("float32", "mixed", "float64"):
            raise ValueError("precision must be 'float32', 'mixed', or 'float64'.")
        if (double_precision and precision != "float64") or (
            mixed_precision and precision != "mixed"
        ):
            raise ValueError("precision conflicts with double_precision or mixed_precision.")
        double_precision = precision == "float64"
        mixed_precision = precision == "mixed"
    if (
        isinstance(warp_tile_width, bool)
        or not isinstance(warp_tile_width, Integral)
        or warp_tile_width not in (0, 1, 2, 4, 8, 16, 32)
    ):
        raise ValueError("warp_tile_width must be 0 or an integer divisor of 32.")
    if warp_tile_width and compute_backend != "cuda":
        raise ValueError("warp_tile_width requires compute_backend='cuda'.")
    if mixed_precision and (double_precision or compute_backend != "cuda"):
        raise ValueError(
            "mixed_precision requires compute_backend='cuda' and double_precision=False."
        )
    import atexit
    import logging
    import os
    from pathlib import Path

    from quantities import Quantity

    import syris.config as cfg
    from syris.backend import HAS_OPENCL, ComputeBackend
    from syris.gpu.util import init_programs, make_opencl_defaults

    LOG = logging.getLogger(__name__)

    cfg.init_logging(
        level=logging.INFO if loglevel is None else loglevel,
        logger_file=logfile,
    )
    cfg.PRECISION = cfg.Precision(double_precision, mixed=mixed_precision)
    cfg.BACKEND = ComputeBackend(compute_backend=compute_backend)
    if mixed_precision and getattr(cfg.BACKEND, "name", None) != cfg.BACKEND.CUDA:
        raise ValueError("mixed_precision requires an available CUDA backend.")
    if warp_tile_width and getattr(cfg.BACKEND, "name", None) != cfg.BACKEND.CUDA:
        raise ValueError("warp_tile_width requires an available CUDA backend.")
    cfg.UNIT = Quantity(1, unit)

    if cfg.BACKEND.name == cfg.BACKEND.CUDA:
        from syris.gpu.cuda_utils import CudaPipeline

        kernel_dir = str(Path(__file__).resolve().parent / "gpu" / "cuda")

        source_files = ["WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu"]
        abs_source_files = [os.path.join(kernel_dir, f) for f in source_files]
        cuda_headers = [
            kernel_dir,
        ]

        options = ["-D__FP_T_D__"] if double_precision else []
        if warp_tile_width:
            options.append(f"-DSYRIS_WARP_TILE_WIDTH={int(warp_tile_width)}")
        if mixed_precision:
            options.append("-DSYRIS_MIXED_PRECISION")
            LOG.info(
                "Mixed precision: Float32 geometry/BVH/images, "
                "Float64 triangle arithmetic and hit distances."
            )
        cfg.BACKEND.pipeline = CudaPipeline(headers=cuda_headers, options=options)

        module_name = "bvh_kernels"

        try:
            cfg.BACKEND.pipeline.readModuleFromFiles(module_name, abs_source_files, jitify=False)
        except Exception as e:
            LOG.error(f"Failed to read modules: {e}")

        kernel_names = [
            "projectTriangleCentroid",
            "growTreeKernel",
            "project_parallel_kernel",
            "compute_thickness_kernel",
            "project_conebeam_kernel",
            "project_parallel_normals_kernel",
            "project_conebeam_normals_kernel",
            "project_shared_samples_kernel",
            "project_missing_samples_kernel",
            "classify_shared_samples_kernel",
            "refine_shared_samples_kernel",
        ]

        try:
            for kernel_name in kernel_names:
                cfg.BACKEND.pipeline.getKernelFromModule(module_name, kernel_name)
        except Exception as e:
            LOG.error(f"Failed to get kernel: {e}")

    if HAS_OPENCL:
        cfg.OPENCL = cfg.OpenCL()
        make_opencl_defaults(
            platform_name=platform_name,
            device_type=device_type,
            device_index=device_index,
            profiling=profiling,
        )
        cfg.BACKEND.queue = cfg.OPENCL.queue
        init_programs()

    if profiling:
        if HAS_OPENCL:
            from syris import profiling as prf
            from syris.gpu.util import _wrap_opencl

            _wrap_opencl()  # This function monkey-patches pyopencl, keep it specific
            prf.PROFILER = prf.Profiler(cfg.OPENCL.queues, profiling_file)
            prf.PROFILER.start()

            @atexit.register
            def exit_handler():
                """Shutdown the profiler on exit."""
                prf.PROFILER.shutdown()
        else:
            LOG.warning("Profiling is only supported for the OpenCL backend.")
