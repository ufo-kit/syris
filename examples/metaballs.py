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
from syris.bodies.isosurfaces import MetaBall, MetaBalls, project_metaballs_naive
from syris.geometry import Trajectory
from syris.util import make_tuple
from util import save_image


LOG = logging.getLogger(__name__)


UNITS = q.m
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
    # z = np.random.uniform(-1 * q.mm, 1 * q.mm) * radius_range.units
    z = np.random.uniform(- n / 2 * pixel_size + radius_range[1], n / 2 * pixel_size - radius_range[1])
    # z = np.random.uniform(radius_range[0], radius_range[1]) * radius_range.units
    r = np.random.uniform(radius_range[0], radius_range[1]) * radius_range.units

    c_points = [(x, y, z)] * q.mm
    trajectory = Trajectory(c_points)
    metaball = MetaBall(trajectory, r)
    metaball.move(0 * q.s)

    return metaball, "({0}, {1}, {2}, {3}),\n".format(x, y, z, r.magnitude)


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
        objects += metaball.pack()

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
        objects_all += metaball.pack()
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


def create_metaball_buffers(n, thickness):
    if thickness:
        res = np.empty((n, VECTOR_WIDTH * n), dtype=cfg.PRECISION.np_float)
        result_mem_size = n ** 2 * cfg.PRECISION.cl_float
        result_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                                size=result_mem_size)
    else:
        result_mem_size = n ** 2 * 2 * MAX_OBJECTS * cfg.PRECISION.cl_float
        res = np.empty(MAX_OBJECTS * 2 * n * VECTOR_WIDTH * n, dtype=cfg.PRECISION.np_float)
        res[:] = np.inf
        result_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                               cl.mem_flags.COPY_HOST_PTR, hostbuf=res)

    return result_mem, res


def intersections_to_slice(n, height, intersections_mem, z_start, pixel_size, program):
    slice_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE, size=n ** 2)
    slice = np.empty((n, n), dtype=np.uint8)

    ev = program.intersections_to_slice(cfg.OPENCL.queue,
                                        (n, n),
                                        None,
                                        slice_mem,
                                        intersections_mem,
                                        np.uint32(height),
                                        cfg.PRECISION.np_float(z_start.rescale(UNITS).magnitude),
                                        cfg.PRECISION.np_float(pixel_size.rescale(UNITS).magnitude))
    ev.wait()
    print "duration:", (ev.profile.end - ev.profile.start) * 1e-6 * q.ms

    cl.enqueue_copy(cfg.OPENCL.queue, slice, slice_mem)
    return slice


if __name__ == '__main__':
    syris.init()

    pixel_size = 1e-3 / SUPERSAMPLING * q.mm

    prg = g_util.get_program(g_util.get_metaobjects_source())
    n = SUPERSAMPLING * 512

    # params = load_params('/home/farago/data/params.txt')
    # # params = [(0.25, 0.25, 0.0, 0.05), (0.35, 0.35, 0.0, 0.1)]
    # num_objects = len(params)
    # balls, objects_all = create_metaballs(params)
    # z_min, z_max = get_z_range(balls)

    # Random metaballs creation
    # num_objects = np.random.randint(1, 100)
    # metaballs, objects_all, coeff, mid = create_metaballs_random(num_objects)
    positions = [(n / 4, n / 2, 0, n / 5),
                 (3 * n / 4, n / 2, 0, n / 5)] * pixel_size.rescale(q.mm).magnitude
    num_objects = len(positions)
    print positions
    metaballs, objects_all = create_metaballs(positions)
    z_min, z_max = get_z_range(metaballs)
    # print coeff, mid
    print 'z min, max:', z_min, z_max, n * pixel_size + z_min, UNITS
    print 'Z steps:', ((z_max - z_min) / pixel_size).simplified

    objects_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY |
                            cl.mem_flags.COPY_HOST_PTR, hostbuf=objects_all)

    traj = Trajectory([(0, 0, 0)] * q.m)
    comp = MetaBalls(traj, metaballs)
    # thickness = comp.project((n, n), pixel_size).get()
    thickness = project_metaballs_naive(metaballs, (n, n), make_tuple(pixel_size)).get()

    objects_mem.release()

    # for h in range(n):
    #     res = intersections_to_slice(n, h, res_mem, z_min, pixel_size, prg)
    #     save_image("/home/farago/data/thickness/radio_{:>05}.tif".format(h),
    #                res[:, ::VECTOR_WIDTH])

    plt.figure()
    plt.imshow(thickness[:, ::VECTOR_WIDTH], origin="lower", cmap=cm.get_cmap("gray"),
               interpolation="nearest")
    plt.colorbar()

    # plt.figure()
    # plt.imshow(res1[:, ::VECTOR_WIDTH], origin="lower", cmap=cm.get_cmap("gray"),
    #            interpolation="nearest")
    # plt.colorbar()
    plt.show()
