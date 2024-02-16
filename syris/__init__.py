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
    platform_name=None,
    device_type=None,
    device_index=None,
    profiling=False,
    profiling_file="profile.dat",
    loglevel=None,
    logfile=None,
    double_precision=False,
):
    """Initialize syris with *device_index*."""
    import atexit
    import logging
    import pyopencl as cl
    import syris.config as cfg
    from syris.gpu.util import make_opencl_defaults, init_programs
    from syris import profiling as prf

    LOG = logging.getLogger(__name__)

    cfg.init_logging(level=logging.INFO if loglevel is None else loglevel, logger_file=logfile)
    cfg.PRECISION = cfg.Precision(double_precision)
    cfg.OPENCL = cfg.OpenCL()
    platforms = []
    try:
        platforms = cl.get_platforms()
    except Exception as e:
        LOG.exception(str(e))
    else:
        if not platforms:
            LOG.warning("No OpenCL platforms found, GPU computing will not be available")
        else:
            make_opencl_defaults(
                platform_name=platform_name,
                device_type=device_type,
                device_index=device_index,
                profiling=profiling,
            )
    if profiling:
        _wrap_opencl()
        prf.PROFILER = prf.Profiler(cfg.OPENCL.queues, profiling_file)
        prf.PROFILER.start()

        @atexit.register
        def exit_handler():
            """Shutdown the profiler on exit."""
            prf.PROFILER.shutdown()

    if platforms:
        init_programs()


def _wrap_opencl():
    import pyopencl as cl
    import syris.config as cfg
    from syris.gpu.util import execute_profiled

    for function in cfg.PROFILED_CL_FUNCTIONS:
        setattr(cl, function.__name__, execute_profiled(function))
