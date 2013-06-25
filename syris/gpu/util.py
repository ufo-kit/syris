"""
Utility functions concerning GPU programming.
"""

import pyopencl as cl
import time
from syris.profiling import profiler
import logging


logger = logging.getLogger(__name__)


def execute(function, *args, **kwargs):
    """Execute an OpenCL *function* and profile it."""
    event = function(*args, **kwargs)
    if function.__class__ == cl.Kernel:
        func_name = function.function_name
    else:
        func_name = function.__name__

    profiler.add(event, func_name)

    return event


def get_cuda_platform(platforms):
    for p in platforms:
        if p.name == "NVIDIA CUDA":
            return p
    return None


def get_cuda_context():
    platforms = cl.get_platforms()
    p = get_cuda_platform(platforms)
    devices = p.get_devices()

    logger.debug("Creating OpenCL context for %d devices." % (len(devices)))
    st = time.time()
    ctx = cl.Context(devices)
    logger.debug("OpenCL context created in %g s." % (time.time()-st))

    return ctx


def get_cuda_devices():
    """Get all CUDA devices."""
    return get_cuda_platform(cl.get_platforms()).get_devices()


def get_command_queues(context, devices=None,
                       queue_args=(), queue_kwargs={}):
    """Create command queues for each of the *devices* within a specified
    *context*. If *devices* is None, NVIDIA GPUs are automatically
    detected and used for creating the command queues.
    """
    if devices is None:
        devices = get_cuda_devices()

    logger.debug("Creating %d command queues." % (len(devices)))
    queues = []
    for device in devices:
        queues.append(cl.CommandQueue(context, device,
                                      *queue_args, **queue_kwargs))

    logger.debug("%d command queues created." % (len(devices)))

    return queues
