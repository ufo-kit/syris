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
Utility functions concerning GPU programming.
"""

import itertools
import pkg_resources
import queue
import sys
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
import quantities as q
from multiprocessing.pool import ThreadPool
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
    cfg.OPENCL.programs["improc"] = get_program(get_source(["vcomplex.cl", "imageprocessing.cl"]))
    cfg.OPENCL.programs["physics"] = get_program(get_source(["vcomplex.cl", "physics.cl"]))
    cfg.OPENCL.programs["geometry"] = get_program(get_metaobjects_source())
    cfg.OPENCL.programs["mesh"] = get_program(get_source(["heapsort.cl", "mesh.cl"]))
    cfg.OPENCL.programs["varconv"] = get_program(get_all_varconvolutions())


def make_opencl_defaults(platform_name=None, device_type=None, device_index=None, profiling=True):
    """Create default OpenCL context from *platform_name* and a command queue based on
    *device_index* to the devices list. If None, all devices are used in the context. If
    *platform_name* is not specified and *device_type* is, get a platform which has devices of that
    type. If *profiling* is True enable it.
    """
    if profiling:
        kwargs = {"properties": cl.command_queue_properties.PROFILING_ENABLE}
    else:
        kwargs = {}
    LOG.debug("Profiling enabled: %s", profiling)
    try:
        if platform_name:
            platform = get_platform(platform_name)
        elif device_type:
            platform = get_platform_by_device_type(device_type)
        else:
            platform = get_cuda_platform()
    except LookupError:
        LOG.error("Platform %s not found, using first one which can be found", platform_name)
        platform = get_platform("")
    LOG.debug("Using platform '%s'", platform.name)
    devices = platform.get_devices()
    if device_index is None:
        cfg.OPENCL.devices = devices
    else:
        cfg.OPENCL.devices = [devices[device_index]]

    cfg.OPENCL.ctx = cl.Context(cfg.OPENCL.devices)
    cfg.OPENCL.queues = get_command_queues(
        cfg.OPENCL.ctx, devices=cfg.OPENCL.devices, queue_kwargs=kwargs
    )
    cfg.OPENCL.queue = cfg.OPENCL.queues[0]


def get_program(src):
    """Create and build an OpenCL program from source string *src*."""
    if cfg.OPENCL.ctx is not None:
        return cl.Program(cfg.OPENCL.ctx, src).build()
    else:
        raise RuntimeError("OpenCL context has not been set yet")


def get_precision_header():
    """Return single or double precision vfloat definitions header."""
    return _SINGLE_HEADER if cfg.PRECISION.is_single() else _DOUBLE_HEADER


def get_source(file_names, precision_sensitive=True):
    """Get source by concatenating files from *file_names* list and apply
    single or double precision parametrization if *precision_sensitive*
    is True.
    """
    string = ""
    for file_name in file_names:
        string += pkg_resources.resource_string(__name__, "opencl/{}".format(file_name)).decode()

    if precision_sensitive:
        string = get_precision_header() + string

    return string


def get_metaobjects_source():
    """Get source string for metaobjects creation."""
    source = "#define MAX_OBJECTS {}".format(cfg.MAX_META_BODIES)
    source += get_source(
        ["polyobject.cl", "heapsort.cl", "polynoms_heapsort.cl", "rootfinding.cl", "metaobjects.cl"]
    )

    return source


def get_varconvolution_source(
    name,
    header="",
    inputs="",
    init="",
    compute_outer="",
    compute_inner="weight = 1.0;",
    after="",
    cplx=False,
    only_kernel=False,
):
    """Create a shift dependent convolution kernel function with *name*. *header* is an OpenCL code
    which is placed in the front of the source before the kernel function. *inputs* are additional
    kernel inputs (see opencl/varconvolution.in for the fixed ones), *init* is the kernel
    initialization code, *compute_outer* is called at every iteration of the outer (y) loop,
    *compute_inner* is called in the inner (x) loop. *after* is the code after both loops. If *cplx*
    is True, the complex version of the kernel is used.
    Pseudo-code of the OpenCL source for the noncomplex version will look like this::

        *header*

        kernel void *name* (read_only image2d_t input,
                            global vfloat *output,
                            const sampler_t sampler,
                            int2 window, *inputs*)
        {
            int idx = get_global_id (0);
            int idy = get_global_id (1);
            int width = get_global_size (0);
            int i, j, tx, ty, imx, imy;
            vfloat value, weight, result = 0.0;

            *init*

            for (j = 0; j < window.y; j++) {
                ty = window.y - j - 1;
                imy = idy + j - window.y / 2;
                *compute_outer*
                for (i = 0; i < window.x; i++) {
                    imx = idx + i - window.x / 2;
                    value = read_imagef (input, sampler, (int2)(imx, imy)).x;
                    tx = window.x - i - 1;
                    *compute_inner*
                    result += value * weight;
                }
            }

            *after*

            output[idy * width + idx] = result;
        }

    The complex version uses two inputs, *input_real* and *input_imag* which are also image2d_t
    instances.
    *compute_inner* must set the *weight* variable in order to apply the convolution kernel weight.
    """
    if inputs:
        inputs = "," + inputs
    kernel_src = get_source(["varconvolution.in"], precision_sensitive=False)
    kernel_src = kernel_src.split("%nl")[1 if cplx else 0]

    if only_kernel:
        top = ""
        header = ""
    else:
        # Precision definition and complex operations
        top = get_precision_header()
        if cplx:
            top += get_source(["vcomplex.cl"], precision_sensitive=False)

    return top + kernel_src.format(header, name, inputs, init, compute_outer, compute_inner, after)


def _get_varconvolve_2d_parametrized(
    name, func_name, func_src, normalized=True, additional_init="", only_kernel=False
):
    """Make a variable convolution kernel named varconvolve_*name*[_normalized], if *normalized* is
    True. *func_name* is the function name, *func_str* is the function code, if *only_kernel* is
    True only the kernel is returned. *additional_init* is added after the coordinate point
    computation and parameter read.
    Suitable for creating kernels where the function computing the convolution kernel takes f(x) and
    g(y) arguments instead of plain x, y coordinates.
    """
    inputs = "global vfloat2 *params"
    init = "vfloat2 p, param = params[idy * width + idx];"
    init += additional_init
    if normalized:
        init += "vfloat sum = 0.0;"
    compute_outer = "p.y = (vfloat) (ty - window.y / 2);"
    compute_inner = "p.x = (vfloat) (tx - window.x / 2);"
    compute_inner += "weight = {} (&p, &param);".format(func_name)
    if normalized:
        compute_inner += "sum += weight;"
        after = "result /= sum;"
    else:
        after = ""

    kernel_name = "varconvolve_{}".format(name)
    if normalized:
        kernel_name += "_normalized"

    return get_varconvolution_source(
        kernel_name,
        header=func_src,
        inputs=inputs,
        init=init,
        compute_outer=compute_outer,
        compute_inner=compute_inner,
        after=after,
        only_kernel=only_kernel,
    )


def get_varconvolve_gauss(normalized=True, window_fwnm=1000, only_kernel=False):
    """Create variable Gaussian convolution. The kernel sum is 1 if *normalized* is True, window is
    computed automatically for every x, y position in the original image based on the sigma at x, y
    and *window_fwnm* as 2 * sqrt(2 * log(*window_fwnm*)) * sigma. If *only_kernel* is True only the
    kernel is returned.
    """
    src = get_source(["varconvolution.in"], precision_sensitive=False).split("%nl")[2]
    # Make the kernel window size variable based on the
    fwnm_factor = 2 * np.sqrt(2 * np.log(window_fwnm))
    LOG.debug("Creating Gaussian convolution with window size FW(1/{})M".format(window_fwnm))
    additional_init = "window.x = (int) ({} * param.x + 0.5);".format(fwnm_factor)
    additional_init += "window.y = (int) ({} * param.y + 0.5);".format(fwnm_factor)
    additional_init += "window.x += 1 - window.x % 2;"
    additional_init += "window.y += 1 - window.y % 2;"

    return _get_varconvolve_2d_parametrized(
        "gauss",
        "get_gauss",
        src,
        normalized=normalized,
        additional_init=additional_init,
        only_kernel=only_kernel,
    )


def get_varconvolve_disk(normalized=True, smooth=True, only_kernel=False):
    """Create variable circlular kernel convolution, kernel sum is 1 if *normalized* is True, if
    *smooth* is True smooth out sharp edges of the disk. If *only_kernel* is True only the kernel
    is returned.
    """
    name = "disk_smooth" if smooth else "disk"
    func_name = "get_{}".format(name)
    src = get_source(["varconvolution.in"], precision_sensitive=False)
    header = src.split("%nl")[4 if smooth else 3]
    additional_init = "window.x = (int) (2 * param.x + 0.5);"
    additional_init += "window.x += 1 - window.x % 2;"
    additional_init += "window.y = (int) (2 * param.y + 0.5);"
    additional_init += "window.y += 1 - window.y % 2;"
    if smooth:
        additional_init += "window.x += 2;"
        additional_init += "window.y += 2;"

    return _get_varconvolve_2d_parametrized(
        name,
        func_name,
        header,
        normalized=normalized,
        additional_init=additional_init,
        only_kernel=only_kernel,
    )


def get_varconvolve_propagator(only_kernel=False):
    """Create the variable propagator convolution. If *only_kernel* is True only the kernel is
    returned.
    """
    inputs = "const vfloat lam,"
    inputs += "read_only image2d_t distances,"
    inputs += "const vfloat2 ps,"
    inputs += "const vfloat2 sigma"
    init = "vcomplex sum = (vcomplex)(0.0, 0.0); vfloat2 p;"
    compute_outer = "p.y = (vfloat) (j - window.y / 2) * ps.y;"
    compute_inner = "p.x = (vfloat) (i - window.x / 2) * ps.x;"
    compute_inner += "weight = get_propagator (&p, lam, "
    compute_inner += "read_imagef (distances, sampler, (int2)(imx, imy)).x, &sigma);"
    compute_inner += "sum += weight;"
    after = "result = result / sqrt (sum.x * sum.x + sum.y * sum.y);"
    src = get_source(["varconvolution.in"], precision_sensitive=False)
    header = src.split("%nl")[2]
    header += src.split("%nl")[5]

    return get_varconvolution_source(
        "varconvolve_propagator",
        header=header,
        inputs=inputs,
        init=init,
        compute_outer=compute_outer,
        compute_inner=compute_inner,
        after=after,
        cplx=True,
        only_kernel=only_kernel,
    )


def get_all_varconvolutions():
    """Create all variable convolutions."""
    src = get_source(["varconvolution.in"], precision_sensitive=False).split("%nl")
    header = "".join([func + "\n" for func in src[2:]])

    k_src = get_varconvolve_gauss(normalized=False, only_kernel=True)
    k_src += get_varconvolve_gauss(normalized=True, only_kernel=True)
    for norm, smooth in itertools.product([False, True], [False, True]):
        k_src += get_varconvolve_disk(normalized=norm, smooth=smooth, only_kernel=True)
    k_src += get_varconvolve_propagator(only_kernel=True)

    top = get_precision_header()
    top += get_source(["vcomplex.cl"], precision_sensitive=False)

    return top + header + k_src


def get_cache(buf):
    """
    Get a device memory object from cache *buf*, which can reside either
    on host or on device.
    """
    if isinstance(buf, cl.MemoryObject):
        result = buf
    else:
        result = cl.Buffer(
            cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf
        )

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


def get_platform(name):
    """Get the first OpenCL platform which contains *name* as its substring."""
    for platform in cl.get_platforms():
        if name in platform.name:
            return platform

    raise LookupError("Platform '{}' not found".format(name))


def get_platform_by_device_type(device_type):
    """Get platform with specific device type (CPU, GPU, ...)."""
    for platform in cl.get_platforms():
        device = platform.get_devices()[0]
        if device.type == device_type:
            return platform

    raise LookupError("There is no platform with device type '{}'", device_type)


def get_gpu_platform():
    """Get any platform with GPUs."""
    return get_platform_by_device_type(cl.device_type.GPU)


def get_cpu_platform():
    """Get any platform with CPUs."""
    return get_platform_by_device_type(cl.device_type.CPU)


def get_cuda_platform():
    """Get the NVIDIA CUDA platform if any."""
    return get_platform("NVIDIA CUDA")


def get_intel_platform():
    """Get the Intel platform if any."""
    return get_platform("Intel")


def get_command_queues(context, devices=None, queue_kwargs=None):
    """Create command queues for each of the *devices* within a specified *context*. If *devices* is
    None, they are obtained from *context*. *queue_kwargs* are passed to the CommandQueue
    constructor.
    """
    if devices is None:
        devices = context.devices
    if queue_kwargs is None:
        queue_kwargs = {}

    LOG.debug("Creating %d command queues", len(devices))
    queues = []
    for device in devices:
        queues.append(cl.CommandQueue(context, device, **queue_kwargs))

    return queues


def _make_vfloat_functions():
    """Make functions for creating OpenCL vfloat data types from host
    data types. Follow PyOpenCL make_floatn and make_doublen convention
    and use them for implementation.
    """

    def _wrapper(i):
        def make_vfloat(*args):
            if cfg.PRECISION.is_single():
                return getattr(cltypes, "make_float%d" % (i))(*args)
            else:
                return getattr(cltypes, "make_double%d" % (i))(*args)

        make_vfloat.__name__ = "make_vfloat%d" % (i)
        return make_vfloat

    for i in [2, 3, 4, 8, 16]:
        setattr(sys.modules[__name__], _wrapper(i).__name__, _wrapper(i))


_make_vfloat_functions()


def make_vcomplex(value):
    """Make complex value for OpenCL based on the set floating point
    precision.
    """
    return getattr(sys.modules[__name__], 'make_vfloat2')(value.real, value.imag)


def get_host(data, queue=None):
    """Get *data* as numpy.ndarray."""
    if not queue:
        queue = cfg.OPENCL.queue

    if isinstance(data, cl_array.Array):
        result = data.get()
    elif isinstance(data, np.ndarray):
        result = data
    elif isinstance(data, cl.Image):
        result = np.empty(data.shape[::-1], np.float32)
        cl.enqueue_copy(queue, result, data, origin=(0, 0), region=result.shape[::-1])
        if result.dtype != cfg.PRECISION.np_float:
            result = result.astype(cfg.PRECISION.np_float)
    else:
        raise TypeError("Unsupported data type {}".format(type(data)))

    return result


def get_array(data, queue=None):
    """Get pyopencl.array.Array from *data* which can be a numpy array, a pyopencl.array.Array or a
    pyopencl.Image. *queue* is an OpenCL command queue.
    """
    if not queue:
        queue = cfg.OPENCL.queue

    if isinstance(data, cl_array.Array):
        result = data
    elif isinstance(data, np.ndarray):
        if data.dtype.kind == "c":
            if data.dtype.itemsize != cfg.PRECISION.cl_cplx:
                data = data.astype(cfg.PRECISION.np_cplx)
            result = cl_array.to_device(queue, data.astype(cfg.PRECISION.np_cplx))
        else:
            if data.dtype.kind != "f" or data.dtype.itemsize != cfg.PRECISION.cl_float:
                data = data.astype(cfg.PRECISION.np_float)
            result = cl_array.to_device(queue, data.astype(cfg.PRECISION.np_float))
    elif isinstance(data, cl.Image):
        result = cl_array.empty(queue, data.shape[::-1], np.float32)
        cl.enqueue_copy(
            queue, result.data, data, offset=0, origin=(0, 0), region=result.shape[::-1]
        )
        if result.dtype.itemsize != cfg.PRECISION.cl_float:
            result = result.astype(cfg.PRECISION.np_float)
    else:
        raise TypeError("Unsupported data type {}".format(type(data)))

    return result


def are_images_supported():
    """Is the INTENSITY|FLOAT image format supported?"""
    fmt = cl.ImageFormat(cl.channel_order.INTENSITY, cl.channel_type.FLOAT)

    return fmt in cl.get_supported_image_formats(
        cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D
    )


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

    if fmt not in cl.get_supported_image_formats(queue.context, access, cl.mem_object_type.IMAGE2D):
        raise RuntimeError("INTENSITY|FLOAT image format not supported by this platform")

    if isinstance(data, cl.Image):
        result = data
    else:
        if isinstance(data, cl_array.Array) or isinstance(data, np.ndarray):
            if data.dtype.kind == "c":
                raise TypeError("Complex values are not supported")
            else:
                data = data.astype(np.float32)
        else:
            raise TypeError("Unsupported data type {}".format(type(data)))

        if isinstance(data, cl_array.Array):
            result = cl.Image(cfg.OPENCL.ctx, access, fmt, shape=data.shape[::-1])
            cl.enqueue_copy(queue, result, data.data, offset=0, origin=(0, 0), region=result.shape)
        elif isinstance(data, np.ndarray):
            result = cl.Image(
                cfg.OPENCL.ctx, access | mf.COPY_HOST_PTR, fmt, shape=data.shape[::-1], hostbuf=data
            )

    return result


def qmap(func, items, queues=None, args=(), kwargs=None):
    """Apply *func* to *items* on multiple command queues. The function *func* should block until
    the execution on a device is finished, otherwise the command queue which is assigned to it might
    return to the pool of usable resources too soon and stall execution. Consider using another
    mechanism if *func* is a kernel, i.e. the multi gpu execution might be realized without
    threading, which is used here.
    *func* is a callable with signature func(item, queue, *args, **kwargs) where item is an item to
    be processed and queue is the OpenCL command queue to be used. *queues* are the command queues
    to be used for computation, if not specified, all the default ones are used. *args* is a list
    and *kwargs* a dictionary, both passed to *func*.
    """
    queue_of_queues = queue.Queue()
    if queues is None:
        queues = cfg.OPENCL.queues
    for cl_queue in queues:
        queue_of_queues.put(cl_queue)
    if kwargs is None:
        kwargs = {}
    pool = ThreadPool(processes=len(queues))

    def process(item):
        queue = queue_of_queues.get()
        LOG.debug(
            "Mapping '{}' to item {} and queue {}".format(func.__name__, item, queues.index(queue))
        )
        result = func(item, queue, *args, **kwargs)
        queue_of_queues.task_done()
        queue_of_queues.put(queue)

        return result

    results = pool.map(process, items)
    pool.close()
    pool.join()

    return results


def get_event_duration(event, start=cl.profiling_info.START, stop=cl.profiling_info.END):
    """Get OpenCL event duration. *start* and *stop* define the OpenCL timer start and stop."""
    return (event.get_profiling_info(stop) - event.get_profiling_info(start)) * 1e-9 * q.s
