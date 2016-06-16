import logging
import re
from matplotlib import pyplot as plt
import numpy as np
import pyopencl as cl
import quantities as q
import syris
from syris import config as cfg
from syris.bodies.isosurfaces import MetaBall, MetaBalls, project_metaballs_naive
from syris.geometry import Trajectory
from syris.util import make_tuple, save_image
from util import get_default_parser, show


LOG = logging.getLogger(__name__)


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
        c_points = [(x[i], y[i], z[i])] * q.um
        trajectory = Trajectory(c_points)
        metaball = MetaBall(trajectory, r[i] * q.um)
        metaball.move(0 * q.s)
        metaballs.append(metaball)
        objects += metaball.pack()

    return metaballs, objects


def get_vfloat_mem_host(mem, size):
    res = np.empty(size, dtype=cfg.PRECISION.np_float)
    cl.enqueue_copy(cfg.OPENCL.queue, res, mem)

    return res


def create_metaballs_random(n, pixel_size, num, min_radius, max_radius):
    params = []

    for j in range(num):
        x = np.random.uniform(0, n)
        y = np.random.uniform(0, n)
        z = np.random.uniform(-2 * max_radius, 2 * max_radius)
        r = np.random.uniform(min_radius, max_radius)
        params.append([x, y, z, r])

    return create_metaballs(params)


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
        res = np.empty((n, n), dtype=cfg.PRECISION.np_float)
        result_mem_size = n ** 2 * cfg.PRECISION.cl_float
        result_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                               size=result_mem_size)
    else:
        result_mem_size = n ** 2 * 2 * cfg.MAX_META_BODIES * cfg.PRECISION.cl_float
        res = np.empty(cfg.MAX_META_BODIES * 2 * n * n,
                       dtype=cfg.PRECISION.np_float)
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
                                        cfg.PRECISION.np_float(z_start.rescale(q.um).magnitude),
                                        cfg.PRECISION.np_float(pixel_size.rescale(q.um).magnitude))
    ev.wait()
    print "duration:", (ev.profile.end - ev.profile.start) * 1e-6 * q.ms

    cl.enqueue_copy(cfg.OPENCL.queue, slice, slice_mem)
    return slice


def parse_args():
    parser = get_default_parser('Metaballs example')
    parser.add_argument('--n', type=int, default=512, help='Number of image pixels')
    parser.add_argument('--method', choices=['random', 'fixed', 'file'], default='random',
                        help='Use random number of metaballs, fixed metaballs or file with '
                        'packed metaball structure')
    parser.add_argument('--num', type=int, default=50, help='Number of random metaballs')
    parser.add_argument('--distance', type=float,
                        help='Distance of the two fixed metaballs in pixels')
    parser.add_argument('--min-radius', type=int, default=5,
                        help='Minimum radius of random metaballs in pixels')
    parser.add_argument('--max-radius', type=int, default=25,
                        help='Maximum radius of random metaballs in pixels')
    parser.add_argument('--input', type=str, help='Input file for packed metaballs')
    parser.add_argument('--output', type=str,
                        help='Output file for created metaballs as a packed structure')
    parser.add_argument('--output-thickness', type=str,
                        help='Output file for projected thickness')
    parser.add_argument('--algorithm', choices=['naive', 'fast'], default='fast',
                        help='Used algorithm for intersections computation')

    return parser.parse_args()


def main():
    args = parse_args()
    syris.init(device_index=0)
    shape = (args.n, args.n)
    pixel_size = 1 * q.um

    if args.method == 'random':
        # Random metaballs creation
        metaballs, objects_all = create_metaballs_random(args.n, pixel_size, args.num,
                                                         args.min_radius, args.max_radius)
    elif args.method == 'file':
        # 1e6 because packing converts to meters
        values = np.fromfile(args.input, dtype=np.float32) * 1e6
        metaballs, objects_all = create_metaballs(values.reshape(len(values) / 4, 4))
    else:
        distance = args.distance or args.n / 4
        positions = [(args.n / 2 - distance, args.n / 2, 0, args.n / 6),
                     (args.n / 2 + distance, args.n / 2, 0, args.n / 6)]
        metaballs, objects_all = create_metaballs(positions)

    if args.output:
        with open(args.output, mode='wb') as out_file:
            out_file.write(objects_all)

    z_min, z_max = get_z_range(metaballs)
    print 'z min, max:', z_min.rescale(q.um), z_max.rescale(q.um), args.n * pixel_size + z_min

    if args.algorithm == 'fast':
        traj = Trajectory([(0, 0, 0)] * q.m)
        comp = MetaBalls(traj, metaballs)
        thickness = comp.project(shape, pixel_size).get()
    else:
        print 'Z steps:', int(((z_max - z_min) / pixel_size).simplified.magnitude + 0.5)
        thickness = project_metaballs_naive(metaballs, shape, make_tuple(pixel_size)).get()

    if args.output_thickness:
        save_image(args.output_thickness, thickness)

    show(thickness)
    plt.show()


if __name__ == '__main__':
    main()
