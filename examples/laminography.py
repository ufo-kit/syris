import argparse
import itertools
import logging
import os
import time
import numpy as np
import quantities as q
import syris.gpu.util as gutil
from functools import partial
from multiprocessing import Lock, Pool
from concert.storage import write_libtiff
from syris.geometry import X_AX, Y_AX, Z_AX


LOCK = Lock()
LOG = logging.getLogger(__name__)


def make_projection(shape, ps, axis, mesh, center, lamino_angle, tomo_angle, ss=1):
    from syris.imageprocessing import bin_image
    if axis == 'z':
        lamino_angle = lamino_angle + 90 * q.deg
        tomo_angle = -tomo_angle
    axis = Y_AX if axis == 'y' else Z_AX
    mesh.clear_transformation()
    mesh.translate(center)
    mesh.rotate(lamino_angle, X_AX)
    mesh.rotate(tomo_angle, axis)

    orig_shape = shape
    shape = tuple([n * ss for n in orig_shape])
    ps = ps / ss

    projection = mesh.project(shape, ps)
    if ss > 1:
        projection = bin_image(projection, orig_shape, average=True)

    return projection.get()


def scan(shape, ps, axis, mesh, angles, prefix, lamino_angle=45 * q.deg, index=0, num_devices=1,
         shift_coeff=1e4, ss=1):
    """Make a scan of tomographic angles. *shift_coeff* is the coefficient multiplied by pixel size
    which shifts the triangles to get rid of faulty pixels.
    """
    psm = ps.simplified.magnitude
    log_fmt = '{}: {:>04}/{:>04} in {:6.2f} s, angle: {:>6.2f} deg, maxima: {}'

    # Move to the middle of the FOV
    point = (shape[1] * psm / 2, shape[0] * psm / 2, 0) * q.m
    if index == 0:
        LOG.info('Mesh shift: {}'.format(point.rescale(q.um)))
        LOG.info('Mesh shift in pixels: {}'.format((point / ps).simplified.magnitude))

    # Compute this device portion of tomographic angles
    enumerated = list(enumerate(angles))
    num_angles = len(enumerated)
    per_device = num_angles / num_devices
    stop = None if index == num_devices - 1 else (index + 1) * per_device
    mine = enumerated[index * per_device:stop]

    last = None
    # Projections which metric exceeds allowed limit
    checked_indices = []
    # Projections which exceed the allowed metric difference even after more iterations
    bad_indices = []
    for i, angle in mine:
        st = time.time()
        projs = [make_projection(shape, ps, axis, mesh, point, lamino_angle, angle, ss=ss)]
        max_vals = [projs[-1].max()]
        best = 0
        if last is not None and max_vals[0] > 2 * last or np.isnan(max_vals[0]):
            # Check for faulty pixels
            checked_indices.append(i)
            for shift in [-psm / shift_coeff, psm / shift_coeff]:
                shifted_point = point + (shift, 0, 0) * q.m
                projs.append(make_projection(shape, ps, axis, mesh, shifted_point,
                                             lamino_angle, angle, ss=ss))
                max_vals.append(projs[-1].max())
            best = np.argmin(max_vals)
            if best > 2 * last or np.isnan(best):
                bad_indices.append(i)
        duration = time.time() - st
        with LOCK:
            LOG.info(log_fmt.format(index, i + 1, num_angles, duration,
                                    float(angle.magnitude), max_vals))
        write_libtiff(prefix.format(i), projs[best])
        last = max_vals[best]

    with LOCK:
        LOG.info('Checked indices: {}'.format(checked_indices))
        LOG.info('Which map to files: {}'.format([prefix.format(i) for i in checked_indices]))
        LOG.info('Exceeding indices: {}'.format(bad_indices))
        LOG.info('Which map to files: {}'.format([prefix.format(i) for i in bad_indices]))

    return projs[best]


def make_ground_truth(args, shape, mesh):
    """Shape is (y, x), so the total number of slices is y."""
    import syris.config as cfg
    from syris.imageprocessing import bin_image

    if args.z_chunk % args.supersampling:
        raise ValueError('z_chunk must be dividable by supersampling')

    queue = cfg.OPENCL.queue
    # Move the mesh to the middle
    ps = args.pixel_size / args.supersampling
    psm = ps.simplified.magnitude
    orig_shape = shape
    shape = tuple([n * args.supersampling for n in shape])
    # Make sure the projections are computed with the same x- and y-offsets
    point = (shape[1] * psm / 2, shape[0] * psm / 2, shape[1] * psm / 2) * q.m
    LOG.info('Mesh shift: {}'.format(point.rescale(q.um)))
    LOG.info('Mesh shift in pixels: {}'.format((point / args.pixel_size).simplified.magnitude))
    mesh.translate(point)
    mesh.transform()
    mesh.sort()

    z_stack = np.empty((args.supersampling,) + orig_shape, dtype=cfg.PRECISION.np_float)

    for i in range(0, shape[0], args.z_chunk):
        end = min(i + args.z_chunk, shape[0])
        offset = gutil.make_vfloat3(0, i * ps.rescale(q.um), 0)
        slices = mesh.compute_slices((end - i,) + shape, ps, offset=offset).get()
        LOG.info('Computing slices {}-{}'.format(i, end))
        enumerated = list(enumerate(slices))[::args.supersampling]
        for j, sl in enumerated:
            # Z-dimension downsampling
            for k in range(args.supersampling):
                z_stack[k] = bin_image(slices[j + k], orig_shape, average=True, queue=queue).get()
            # Sum only the slices which are present (last run might not go to the end)
            sl = np.mean(z_stack[:slices.shape[0]], axis=0)
            index = (i + j) / args.supersampling
            write_libtiff(args.prefix.format(index), sl)

    return sl


def process(args, device_index):
    import syris
    from syris.geometry import Trajectory
    from syris.bodies.mesh import Mesh, read_blender_obj

    syris.init(device_index=device_index, logfile=args.logfile)
    path, ext = os.path.splitext(args.input)
    if ext == '.obj':
        tri = read_blender_obj(args.input)
    else:
        tri = np.load(args.input)
    tri = tri * q.um

    tr = Trajectory([(0, 0, 0)] * q.um)
    mesh = Mesh(tri, tr, center='bbox', iterations=2)

    fov = max([ends[1] - ends[0] for ends in mesh.extrema[:-1]]) * 1.1
    n = int(np.ceil((fov / args.pixel_size).simplified.magnitude))
    shape = (n, n)

    if args.make_gt:
        LOG.info('--- Args info ---')
        log_attributes(args)

        return make_ground_truth(args, shape, mesh)
    else:
        # 360 degrees -> twice the number of tomographic projections
        num_projs = int(np.pi * n) if args.num_projections is None else args.num_projections
        angles = np.linspace(0, 360, num_projs, endpoint=False) * q.deg
        if device_index == 0:
            LOG.info('n: {}, ps: {}, FOV: {}'.format(n, args.pixel_size, fov))
            LOG.info('Number of projections: {}'.format(num_projs))
            LOG.info('--- Mesh info ---')
            log_attributes(mesh)
            LOG.info('--- Args info ---')
            log_attributes(args)

        return scan(shape, args.pixel_size, args.rotation_axis, mesh, angles, args.prefix,
                    lamino_angle=args.lamino_angle, index=device_index,
                    num_devices=args.num_devices, ss=args.supersampling)


def parse_args():
    parser = argparse.ArgumentParser(description='Mesh example')
    parser.add_argument('input', type=str, help='Blender .obj input file name')
    parser.add_argument('--dset', type=str,
                        help='Data set name, if not specified guessed from input')
    parser.add_argument('--num-projections', type=int, help='Number of projections')
    parser.add_argument('--out-directory', type=str, default='dataset',
                        help="Output directory, result goes to 'out-directory/dset/projections'"
                        "or 'out-directory/dset/truth', depending on the --make-gt switch")
    parser.add_argument('--pixel-size', type=float, default=[750.], nargs='+',
                        help='Pixel size in nm')
    parser.add_argument('--lamino-angle', type=float, default=[5], nargs='+',
                        help='Laminographic angle in degrees')
    parser.add_argument('--rotation-axis', type=str, choices=['y', 'z'], default=['y'],
                        nargs='+', help='Rotation axis (y - up, z - beam direction)')
    parser.add_argument('--num-devices', type=int, default=1,
                        help='Number of compute devices to use')
    parser.add_argument('--supersampling', type=int, default=[1], nargs='+',
                        help='Supersampling computes with n-times more pixels than usual')
    # Ground truth related
    parser.add_argument('--z-chunk', type=int, default=100,
                        help='Number of ground truth slices to compute during one pass')
    parser.add_argument('--make-gt', action='store_true',
                        help='Create ground truth instead of projections')

    return parser.parse_args()


def main():
    args = parse_args()
    combinations = list(itertools.product(args.lamino_angle,
                                          args.pixel_size,
                                          args.rotation_axis,
                                          args.supersampling))
    if args.make_gt:
        image_directory = 'truth'
        file_prefix = 'slice'
    else:
        image_directory = 'projections'
        file_prefix = image_directory[:-1]

    file_prefix += '_{:>04}.tif'

    devices = range(args.num_devices)
    pool = Pool(processes=args.num_devices)

    for lamino_angle, pixel_size, rotation_axis, ss in combinations:
        # Prepare output
        if args.dset is None:
            dset = os.path.splitext(os.path.basename(args.input))[0]
        else:
            dset = args.dset
        dset += '_lamino_angle_{:>02}_deg'.format(int(lamino_angle))
        dset += '_axis_{}'.format(rotation_axis)
        dset += '_ps_{:>04}_nm'.format(int(pixel_size))
        dset += '_ss_{:>02}'.format(ss)

        args.prefix = os.path.join(args.out_directory, dset, image_directory, file_prefix)
        args.logfile = os.path.join(args.out_directory, dset, 'simulation.log')
        directory = os.path.dirname(args.prefix)
        if not os.path.exists(directory):
            os.makedirs(directory, mode=0o755)

        args.pixel_size = pixel_size * q.nm
        args.lamino_angle = lamino_angle * q.deg
        args.rotation_axis = rotation_axis
        args.supersampling = ss

        if args.num_devices == 1:
            # Easier exception message handling for debugging
            process(args, 0)
        else:
            exec_func = partial(process, args)
            pool.map(exec_func, devices)


def log_attributes(obj):
    """Log object *obj* attributes."""
    for attr in dir(obj):
        if not attr.startswith('_') and not callable(getattr(obj, attr)):
            LOG.info('{}: {}'.format(attr, getattr(obj, attr)))


if __name__ == '__main__':
    main()
