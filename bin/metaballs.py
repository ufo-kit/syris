import logging
import re
from matplotlib import pyplot as plt, cm
import numpy as np
import pyopencl as cl
from pyopencl.array import vec
import quantities as q
import syris
from syris import config as cfg
from syris.gpu import util as g_util
from syris.opticalelements.graphicalobjects import MetaBall
from syris.opticalelements.geometry import Trajectory
from libtiff import TIFF


LOG = logging.getLogger(__name__)


UNITS = q.mm
VECTOR_WIDTH = 1
SUPERSAMPLING = 1
MAX_OBJECTS = 30


def diff(ar):
    res = []
    for i in range(1, len(ar)):
        res.append(np.abs(ar[i] - ar[i - 1]))

    return res


def print_array(ar):
    if len(ar) == 0:
        return

    res = ""

    for i in range(len(ar)):
        res += "{:.8f}, ".format(float(ar[i]))

    print res


def create_metaball_random(n, pixel_size, radius_range):
    x = np.random.uniform(0, n * pixel_size)
    y = np.random.uniform(0, n * pixel_size)
    z = np.random.uniform(radius_range[0], radius_range[1]) * radius_range.units
    r = np.random.uniform(radius_range[0], radius_range[1]) * radius_range.units

    c_points = [(x, y, z)] * q.mm
    trajectory = Trajectory(c_points)
    metaball = MetaBall(trajectory, r)
    metaball.move(0 * q.s)

    return metaball, "({0}, {1}, {2}, {3}),\n".format(x, y, z.magnitude, r.magnitude)


def load_params(file_name):
    string = open(file_name, 'r').read()
    lines = string.split('\n')
    float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
    floats = 4 * [float_pattern]
    pattern = re.compile(r'\((?P<x>{})\, (?P<y>{})\, (?P<z>{})\, (?P<r>{})'.format(*floats))

    params = []
    for line in lines:
        match = pattern.match(line)
        if match:
            x, y = match.group('x'), match.group('y')
            z, r = match.group('z'), match.group('r')
            params.append((float(x), float(y), float(z), float(r)))

    return params


def create_metaballs(params):
    x, y, z, r = zip(*params)

    objects = ""
    metaballs = []
    for i in range(len(params)):
        c_points = [(x[i], y[i], z[i])] * q.mm
        trajectory = Trajectory(c_points)
        metaball = MetaBall(trajectory, r[i] * q.mm)
        metaball.move(0 * q.s)
        metaballs.append(metaball)
        objects += metaball.pack(UNITS)

    return metaballs, objects


def get_vfloat_mem_host(mem, size):
    res = np.empty(size, dtype=cfg.PRECISION.np_float)
    cl.enqueue_copy(cfg.OPENCL.queue, res, mem)

    return res


def create_metaballs_random(num_objects, write_parameters=False):
    objects_all = ""
    params_all = ""
    metaballs = []
    min_r = SUPERSAMPLING * 5 * pixel_size.rescale(UNITS).magnitude
    radius_range = (min_r,
                    SUPERSAMPLING * 50 *
                    pixel_size.rescale(UNITS).magnitude) * UNITS
    mid = (radius_range[0].rescale(UNITS).magnitude +
           radius_range[1].rescale(UNITS).magnitude) / 2
    coeff = 1.0 / mid
    eps = pixel_size.rescale(UNITS).magnitude
    print "mid, coeff, eps:", mid, coeff, eps
    print "objects:", num_objects
    for j in range(num_objects):
        metaball, params = create_metaball_random(n, pixel_size, radius_range)
        objects_all += metaball.pack(UNITS)
        params_all += params
        metaballs.append(metaball)

    if write_parameters:
        with open("/home/farago/data/params.txt", "w") as f:
            f.write(params_all + "\n")

    return metaballs, objects_all, coeff, mid


def get_z_range(metaballs):
    z_min = np.inf
    z_max = -np.inf

    for ball in metaballs:
        z_pos = ball.position[2]
        z_start = z_pos - 2 * ball.radius
        z_end = z_pos + 2 * ball.radius
        if z_start < z_min:
            z_min = z_start
        if z_end > z_max:
            z_max = z_end

    return z_min, z_max

def slow_metaballs(n, objects_mem, thickness_mem, num_objects, pixel_size):
    ev = prg.naive_thickness(cfg.OPENCL.queue,
                        (n, n),
                        None,
                        thickness_mem,
                        objects_mem,
                        np.uint32(num_objects),
                        g_util.make_vfloat2(z_min.rescale(UNITS).magnitude,
                                            z_max.rescale(UNITS).magnitude),
                        np.float32(pixel_size.rescale(UNITS)))
    cl.wait_for_events([ev])
    print "duration:", (ev.profile.end - ev.profile.start) * 1e-6 * q.ms

    res = np.empty((n, VECTOR_WIDTH * n), dtype=cfg.PRECISION.np_float)
    cl.enqueue_copy(cfg.OPENCL.queue, res, thickness_mem)

    return res



def fast_metaballs(n, mid, coeff, objects_mem, thickness_mem, num_objects, pixel_size):
    pobjects_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                             size=n ** 2 * MAX_OBJECTS * 4 * 7)
    left_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                         size=n ** 2 * 2 * MAX_OBJECTS)
    right_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                         size=n ** 2 * 2 * MAX_OBJECTS)

    ev = prg.thickness(cfg.OPENCL.queue,
                      (n, n),
                       None,
                       thickness_mem,
                       objects_mem,
                       pobjects_mem,
                       left_mem,
                       right_mem,
                       np.int32(num_objects),
                       vec.make_int2(0, 0),
                       vec.make_int4(0, 0, n, n),
                       g_util.make_vfloat2(pixel_size.rescale(UNITS).magnitude,
                                           pixel_size.rescale(UNITS).magnitude),
                       np.int32(True))
    cl.wait_for_events([ev])
    print "duration:", (ev.profile.end - ev.profile.start) * 1e-6 * q.ms

    res = np.empty((n, VECTOR_WIDTH * n), dtype=cfg.PRECISION.np_float)
    cl.enqueue_copy(cfg.OPENCL.queue, res, thickness_mem)

    return res


if __name__ == '__main__':
    syris.init()

    pixel_size = 1e-3 / SUPERSAMPLING * q.mm

    prg = g_util.get_program(g_util.get_metaobjects_source())
    n = SUPERSAMPLING * 512
    thickness_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                              size=n ** 2 * VECTOR_WIDTH * cfg.PRECISION.cl_float)

    mid, coeff, eps = 0.0275, 36.3636363636, 0.001
    params = load_params('/home/farago/data/params.txt')
    num_objects = len(params)
    balls, objects_all = create_metaballs(params)

    z_min, z_max = get_z_range(balls)
    print 'Z steps:', ((z_max - z_min) / pixel_size).simplified

    # Random metaballs creation
    # num_objects = np.random.randint(1, 100)
    # metaballs, objects_all, coeff, mid = create_metaballs_random(num_objects)
    # z_min, z_max = get_z_range(metaballs)
    # print coeff, mid

    objects_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY |
                            cl.mem_flags.COPY_HOST_PTR, hostbuf=objects_all)

    res = fast_metaballs(n, mid, coeff, objects_mem, thickness_mem, num_objects, pixel_size)
    res1 = slow_metaballs(n, objects_mem, thickness_mem, num_objects, pixel_size)
    objects_mem.release()

    TIFF.open("/home/farago/data/thickness/radio.tif", "w").\
        write_image(res[:, ::VECTOR_WIDTH].astype(np.float32))

    plt.figure()
    plt.imshow(res[:, ::VECTOR_WIDTH], origin="lower", cmap=cm.get_cmap("gray"),
               interpolation="nearest")
    plt.colorbar()

    plt.figure()
    plt.imshow(res1[:, ::VECTOR_WIDTH], origin="lower", cmap=cm.get_cmap("gray"),
               interpolation="nearest")
    plt.colorbar()
    plt.show()
