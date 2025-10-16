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
    unit="um"
):
    """Initialize syris with the best available compute backend."""
    import atexit
    import logging
    import syris.config as cfg
    import pkg_resources
    import os
    from syris.backend import ComputeBackend
    from syris.gpu.cuda_utils import CudaPipeline, CudaTimer
    from syris.gpu.util import make_opencl_defaults, init_programs
    from quantities import Quantity

    LOG = logging.getLogger(__name__)

    cfg.init_logging(level=logging.INFO if loglevel is None else loglevel, logger_file=logfile)
    cfg.PRECISION = cfg.Precision(double_precision)
    cfg.BACKEND = ComputeBackend(compute_backend=compute_backend)
    cfg.UNIT = Quantity(1, unit)

    if cfg.BACKEND.name == cfg.BACKEND.CUDA:
        kernel_dir = pkg_resources.resource_filename('syris', 'gpu/cuda/')

        source_files = ["WatertightRay.cu", "Ray.cu", "source.cu", "Legacy.cu"]
        abs_source_files = [os.path.join(kernel_dir, f) for f in source_files]
        cuda_headers = [kernel_dir,]

        options = ["-D__FP_T_D__", "-G"] if double_precision else []
        cfg.BACKEND.pipeline = CudaPipeline(headers=cuda_headers, options=options)

        module_name = "bvh_kernels"

        try:
            cfg.BACKEND.pipeline.readModuleFromFiles(
                module_name, abs_source_files, jitify=False
            )
        except Exception as e:
            LOG.error(f"Failed to read modules: {e}")

        kernel_names = [
            "projectTriangleCentroid", "growTreeKernel", "project_parallel_kernel",
            "compute_thickness_kernel",
            # "project_conebeam_kernel", 
            # "project_parallel_normals_kernel",
            # "project_conebeam_normals_kernel"
        ]

        try:
            for kernel_name in kernel_names:
                cfg.BACKEND.pipeline.getKernelFromModule(module_name, kernel_name)
        except Exception as e:
            LOG.error(f"Failed to get kernel: {e}")

    if cfg.BACKEND.name == cfg.BACKEND.OPENCL:
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
        if cfg.BACKEND.name == cfg.BACKEND.OPENCL:
            from syris import profiling as prf
            from syris.gpu.util import _wrap_opencl
            
            _wrap_opencl() # This function monkey-patches pyopencl, keep it specific
            prf.PROFILER = prf.Profiler(cfg.OPENCL.queues, profiling_file)
            prf.PROFILER.start()

            @atexit.register
            def exit_handler():
                """Shutdown the profiler on exit."""
                prf.PROFILER.shutdown()
        else:
            LOG.warning("Profiling is only supported for the OpenCL backend.")