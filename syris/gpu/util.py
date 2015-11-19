"""
Utility functions concerning GPU programming.
"""

import pkg_resources
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.array import vec
import time
from syris import profiling as prf
from syris import config as cfg
import logging


LOG = logging.getLogger(__name__)


_SINGLE_HEADER = """
typedef float vfloat;
typedef float2 vfloat2;
typedef float3 vfloat3;
typedef float4 vfloat4;
typedef float8 vfloat8;
typedef float16 vfloat16;

"""

_DOUBLE_HEADER = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef double vfloat;
typedef double2 vfloat2;
typedef double3 vfloat3;
typedef double4 vfloat4;
typedef double8 vfloat8;
typedef double16 vfloat16;

"""


def init_programs():
    """Initialize all OpenCL kernels needed by syris."""
    cfg.OPENCL.programs['improc'] = get_program(get_source(['vcomplex.cl', 'imageprocessing.cl']))
    cfg.OPENCL.programs['physics'] = get_program(get_source(['vcomplex.cl', 'physics.cl']))
    cfg.OPENCL.programs['geometry'] = get_program(get_metaobjects_source())


def make_opencl_defaults(device_index=0, profiling=True):
    """Create default OpenCL context and a command queue based on *device_index* to the devices
    list. If *profiling* is True enable it.
    """
    if profiling:
        kwargs = {"properties": cl.command_queue_properties.PROFILING_ENABLE}
    else:
        kwargs = {}
    cfg.OPENCL.devices = [get_cuda_devices()[device_index]]
    cfg.OPENCL.ctx = get_cuda_context(devices=cfg.OPENCL.devices)
    cfg.OPENCL.queues = get_command_queues(cfg.OPENCL.ctx, cfg.OPENCL.devices, queue_kwargs=kwargs)
    cfg.OPENCL.queue = cfg.OPENCL.queues[0]


def get_program(src):
    """Create and build an OpenCL program from source string *src*."""
    if cfg.OPENCL.ctx is not None:
        return cl.Program(cfg.OPENCL.ctx, src).build()
    else:
        raise RuntimeError('OpenCL context has not been set yet')


def get_source(file_names, precision_sensitive=True):
    """Get source by concatenating files from *file_names* list and apply
    single or double precision parametrization if *precision_sensitive*
    is True.
    """
    string = ""
    for file_name in file_names:
        string += pkg_resources.resource_string(__name__, 'opencl/{}'.format(file_name))

    if precision_sensitive:
        header = _SINGLE_HEADER if cfg.PRECISION.is_single() else _DOUBLE_HEADER
        string = header + string

    return string


def get_metaobjects_source():
    """Get source string for metaobjects creation."""
    source = '#define MAX_OBJECTS {}'.format(cfg.MAX_META_BODIES)
    source += get_source(["polyobject.cl", "heapsort.cl",
                         "polynoms_heapsort.cl", "rootfinding.cl",
                         "metaobjects.cl"])

    return source


def get_cache(buf):
    """
    Get a device memory object from cache *buf*, which can reside either
    on host or on device.
    """
    if isinstance(buf, cl.MemoryObject):
        result = buf
    else:
        result = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)

    return result


def cache(mem, shape, dtype, cache_type=cfg.DEFAULT_CACHE):
    """
    Cache a device memory object *mem* with *shape* and numpy data type
    *dtype* on host or device based on *cache_type*.
    """
    if cache_type == cfg.CACHE_HOST:
        # We need to copy from device to host.
        result = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(cfg.OPENCL.queue, result, mem)
    else:
        result = mem

    return result


def execute_profiled(function):
    """Execute a *function* which can be an OpenCL kernel or other OpenCL
    related function and profile it.
    """
    def wrapped(*args, **kwargs):
        """Wrap a function for profiling."""
        event = function(*args, **kwargs)
        if hasattr(function, "function_name"):
            name = function.function_name
        else:
            name = function.__name__

        prf.PROFILER.add(event, name)
        return event

    # Preserve the name.
    wrapped.__name__ = function.__name__

    return wrapped


def get_cuda_platform(platforms):
    """Get the NVIDIA CUDA platform if any."""
    for plat in platforms:
        if plat.name == "NVIDIA CUDA":
            return plat
    return None


def get_cuda_devices():
    """Get all CUDA devices."""
    return get_cuda_platform(cl.get_platforms()).get_devices()


def get_cuda_context(devices=None, properties=None):
    """Create an NVIDIA CUDA context with *properties* for *devices*,
    if None are given create the context for all available."""
    if devices is None:
        devices = get_cuda_platform(cl.get_platforms()).get_devices()

    LOG.debug("Creating OpenCL context for %d devices." % (len(devices)))
    start = time.time()
    ctx = cl.Context(devices, properties)
    LOG.debug("OpenCL context created in %g s." % (time.time() - start))

    return ctx


def get_command_queues(context, devices=None,
                       queue_args=None, queue_kwargs=None):
    """Create command queues for each of the *devices* within a specified
    *context*. If *devices* is None, NVIDIA GPUs are automatically
    detected and used for creating the command queues.
    """
    if devices is None:
        devices = get_cuda_devices()
    if queue_args is None:
        queue_args = ()
    if queue_kwargs is None:
        queue_kwargs = {}

    LOG.debug("Creating %d command queues." % (len(devices)))
    queues = []
    for device in devices:
        queues.append(cl.CommandQueue(context, device,
                                      *queue_args, **queue_kwargs))

    LOG.debug("%d command queues created." % (len(devices)))

    return queues


def _make_vfloat_functions():
    """Make functions for creating OpenCL vfloat data types from host
    data types. Follow PyOpenCL make_floatn and make_doublen convention
    and use them for implementation.
    """
    def _wrapper(i):
        def make_vfloat(*args):
            if cfg.PRECISION.is_single():
                return getattr(vec, "make_float%d" % (i))(*args)
            else:
                return getattr(vec, "make_double%d" % (i))(*args)
        make_vfloat.__name__ = "make_vfloat%d" % (i)
        return make_vfloat

    for i in [2, 3, 4, 8, 16]:
        globals()[_wrapper(i).__name__] = _wrapper(i)

_make_vfloat_functions()


def make_vcomplex(value):
    """Make complex value for OpenCL based on the set floating point
    precision.
    """
    return make_vfloat2(value.real, value.imag)


def get_array(data, queue=None):
    """Get pyopencl.array.Array from *data* which can be a numpy array, a pyopencl.array.Array or a
    pyopencl.Image. *queue* is an OpenCL command queue.
    """
    if not queue:
        queue = cfg.OPENCL.queue

    if isinstance(data, cl_array.Array):
        result = data
    elif isinstance(data, np.ndarray):
        if data.dtype.kind == 'c':
            if data.dtype.itemsize != cfg.PRECISION.cl_cplx:
                data = data.astype(cfg.PRECISION.np_cplx)
            result = cl_array.to_device(queue, data.astype(cfg.PRECISION.np_cplx))
        else:
            if data.dtype.kind != 'f' or data.dtype.itemsize != cfg.PRECISION.cl_float:
                data = data.astype(cfg.PRECISION.np_float)
            result = cl_array.to_device(queue, data.astype(cfg.PRECISION.np_float))
    elif isinstance(data, cl.Image):
        result = cl_array.empty(queue, data.shape[::-1], np.float32)
        cl.enqueue_copy(queue, result.data, data, offset=0, origin=(0, 0),
                        region=result.shape[::-1])
        if result.dtype.itemsize != cfg.PRECISION.cl_float:
            result = result.astype(cfg.PRECISION.np_float)
    else:
        raise TypeError('Unsupported data type {}'.format(type(data)))

    return result


def get_image(data, access=cl.mem_flags.READ_ONLY, queue=None):
    """Get pyopencl.Image from *data* which can be a numpy array, a pyopencl.array.Array or a
    pyopencl.Image. The image channel order is pyopencl.channel_order.INTENSITY and channel_type is
    pyopencl.channel_type.FLOAT. *access* is either pyopencl.mem_flags.READ_ONLY or
    pyopencl.mem_flags.WRITE_ONLY. *queue* is an OpenCL command queue.
    """
    if not queue:
        queue = cfg.OPENCL.queue

    fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)
    mf = cl.mem_flags

    if isinstance(data, cl.Image):
        result = data
    else:
        if isinstance(data, cl_array.Array) or isinstance(data, np.ndarray):
            if data.dtype.kind == 'c':
                raise TypeError('Complex values are not supported')
            else:
                data = data.astype(np.float32)
        else:
            raise TypeError('Unsupported data type {}'.format(type(data)))

        if isinstance(data, cl_array.Array):
            result = cl.Image(cfg.OPENCL.ctx, access, fmt, shape=data.shape[::-1])
            cl.enqueue_copy(queue, result, data.data, offset=0, origin=(0, 0), region=result.shape)
        elif isinstance(data, np.ndarray):
            result = cl.Image(cfg.OPENCL.ctx, access | mf.COPY_HOST_PTR, fmt,
                              shape=data.shape[::-1], hostbuf=data)

    return result
