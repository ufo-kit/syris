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

"""Laminography data set generation with mesh geometry."""
import imageio
import itertools
import glob
import logging
import os
import time
import numpy as np
import quantities as q
from functools import partial
from multiprocessing import Lock, Pool
from syris.geometry import X_AX, Y_AX, Z_AX
from .util import get_default_parser


LOCK = Lock()
LOG = logging.getLogger(__name__)


def make_projection(shape, ps, axis, mesh, center, lamino_angle, tomo_angle, ss=1):
    from syris.imageprocessing import bin_image

    if axis == "z":
        lamino_angle = lamino_angle + 90 * q.deg
        tomo_angle = -tomo_angle
    axis = Y_AX if axis == "y" else Z_AX
    mesh.clear_transformation()
    mesh.translate(center)
    mesh.rotate(lamino_angle, X_AX)
    mesh.rotate(tomo_angle, axis)

    orig_shape = shape
    shape = tuple([n * ss for n in orig_shape])
    ps = ps / ss

    # t=None for trajectory override
    projection = mesh.project(shape, ps, t=None)
    if ss > 1:
        projection = bin_image(projection, orig_shape, average=True)

    return projection.get()


def read_mesh(filename, iterations=1, mesh_pixel_size=None):
    from syris.bodies.mesh import Mesh, read_blender_obj
    from syris.geometry import Trajectory

    path, ext = os.path.splitext(filename)
    if ext == ".obj":
        tri = read_blender_obj(filename)
    else:
        tri = np.load(filename)

    if mesh_pixel_size:
        tri = tri * mesh_pixel_size * q.nm
    else:
        tri = tri * q.um
    tr = Trajectory([(0, 0, 0)] * q.um)

    return Mesh(tri, tr, center=None, iterations=iterations)


def scan(
    shape,
    ps,
    axis,
    mesh_filename,
    angles,
    prefix,
    lamino_angle=45 * q.deg,
    index=0,
    num_devices=1,
    shift_coeff=1e4,
    ss=1,
    mesh_pixel_size=None,
    num_meshes=1,
    supersampling_projection=1,
):
    """Make a scan of tomographic angles. *shift_coeff* is the coefficient multiplied by pixel size
    which shifts the triangles to get rid of faulty pixels.
    """
    psm = ps.simplified.magnitude
    log_fmt = "{}: {:>04}/{:>04} in {:6.2f} s, angle: {:>6.2f} deg, maxima: {}"
    if os.path.isfile(mesh_filename):
        mesh_filenames = [mesh_filename]
    else:
        mesh_filenames = sorted(glob.glob(mesh_filename))

    # Move to the middle of the FOV
    point = (shape[1] * psm / 2, 0, 0) * q.m
    if index == 0:
        LOG.info("Mesh shift: {}".format(point.rescale(q.um)))
        LOG.info("Mesh shift in pixels: {}".format((point / ps).simplified.magnitude))

    # Compute this device portion of tomographic angles
    enumerated = list(enumerate(angles))
    num_angles = len(enumerated)
    per_device = num_angles // num_devices
    stop = None if index == num_devices - 1 else (index + 1) * per_device
    mine = enumerated[index * per_device : stop]

    last = None
    # Projections which metric exceeds allowed limit
    checked_indices = []
    # Projections which exceed the allowed metric difference even after more iterations
    bad_indices = []
    i_mesh = None
    for i, angle in mine:
        if i * num_meshes // num_angles != i_mesh:
            i_mesh = i * num_meshes // num_angles
            mesh = read_mesh(
                mesh_filenames[i_mesh],
                iterations=supersampling_projection,
                mesh_pixel_size=mesh_pixel_size,
            )
            with LOCK:
                LOG.info("i: %d, reading mesh %d", i, i_mesh)
        st = time.time()
        projs = [make_projection(shape, ps, axis, mesh, point, lamino_angle, angle, ss=ss)]
        max_vals = [projs[-1].max()]
        best = 0
        if last is not None and max_vals[0] > 2 * last or np.isnan(max_vals[0]):
            # Check for faulty pixels
            checked_indices.append(i)
            for shift in [-psm / shift_coeff, psm / shift_coeff]:
                shifted_point = point + (shift, 0, 0) * q.m
                projs.append(
                    make_projection(
                        shape, ps, axis, mesh, shifted_point, lamino_angle, angle, ss=ss
                    )
                )
                max_vals.append(projs[-1].max())
            best = np.argmin(max_vals)
            if max_vals[best] > 2 * last or np.isnan(max_vals[best]):
                bad_indices.append(i)
        duration = time.time() - st
        with LOCK:
            LOG.info(
                log_fmt.format(index, i + 1, num_angles, duration, float(angle.magnitude), max_vals)
            )
        imageio.imwrite(prefix.format(i), projs[best])
        last = max_vals[best]

    with LOCK:
        LOG.info("Checked indices: {}".format(checked_indices))
        LOG.info("Which map to files: {}".format([prefix.format(i) for i in checked_indices]))
        LOG.info("Exceeding indices: {}".format(bad_indices))
        LOG.info("Which map to files: {}".format([prefix.format(i) for i in bad_indices]))

    return projs[best]


def make_ground_truth(args, shape, mesh):
    """Shape is (y, x), so the total number of slices is y."""
    import syris.config as cfg
    from syris.imageprocessing import bin_image

    if args.z_chunk % args.supersampling:
        raise ValueError("z_chunk must be dividable by supersampling")

    queue = cfg.OPENCL.queue
    # Move the mesh to the middle
    ps = args.pixel_size / args.supersampling
    psm = ps.simplified.magnitude
    orig_shape = shape
    shape = tuple([n * args.supersampling for n in shape])
    # Make sure the projections are computed with the same x- and y-offsets
    point = (shape[1] * psm / 2, shape[0] * psm / 2, shape[1] * psm / 2) * q.m
    LOG.info("Mesh shift: {}".format(point.rescale(q.um)))
    LOG.info("Mesh shift in pixels: {}".format((point / args.pixel_size).simplified.magnitude))
    mesh.translate(point)
    mesh.transform()
    mesh.sort()

    z_stack = np.empty((args.supersampling,) + orig_shape, dtype=cfg.PRECISION.np_float)

    for i in range(0, shape[0], args.z_chunk):
        end = min(i + args.z_chunk, shape[0])
        offset = (0, i * ps.rescale(q.um).magnitude, 0) * q.um
        slices = mesh.compute_slices((end - i,) + shape, ps, offset=offset).get()
        LOG.info("Computing slices {}-{}".format(i, end))
        enumerated = list(enumerate(slices))[:: args.supersampling]
        for j, sl in enumerated:
            # Z-dimension downsampling
            for k in range(args.supersampling):
                z_stack[k] = bin_image(slices[j + k], orig_shape, average=True, queue=queue).get()
            # Sum only the slices which are present (last run might not go to the end)
            sl = np.mean(z_stack[: slices.shape[0]], axis=0)
            index = (i + j) // args.supersampling
            imageio.imwrite(args.prefix.format(index), sl)

    return sl


def process(args, device_index):
    import syris

    syris.init(
        device_index=device_index, logfile=args.logfile, double_precision=args.double_precision
    )
    mesh = read_mesh(
        args.input if os.path.isfile(args.input) else sorted(glob.glob(args.input))[0],
        iterations=args.supersampling_projection
    )

    if args.n:
        n = args.n
        fov = n * args.pixel_size
    else:
        fov = max([ends[1] - ends[0] for ends in mesh.extrema[:-1]]) * 1.1
        n = int(np.ceil((fov / args.pixel_size).simplified.magnitude))
    shape = (n, n)

    if args.make_gt:
        LOG.info("--- Args info ---")
        log_attributes(args)

        return make_ground_truth(args, shape, mesh)
    else:
        num_projs = int(np.pi * n) if args.num_projections is None else args.num_projections
        angles = np.linspace(0, args.rotation_angle, num_projs, endpoint=False) * q.deg
        if device_index == 0:
            LOG.info("n: {}, ps: {}, FOV: {}".format(n, args.pixel_size, fov))
            LOG.info("Total rotation angle: {} deg".format(args.rotation_angle))
            LOG.info("Number of projections: {}".format(num_projs))
            LOG.info("--- Mesh info ---")
            log_attributes(mesh)
            LOG.info("--- Args info ---")
            log_attributes(args)

        return scan(
            shape,
            args.pixel_size,
            args.rotation_axis,
            args.input,
            angles,
            args.prefix,
            lamino_angle=args.lamino_angle,
            index=device_index,
            num_devices=args.num_devices,
            ss=args.supersampling,
            num_meshes=args.num_meshes,
            supersampling_projection=args.supersampling_projection,
            mesh_pixel_size=args.mesh_pixel_size,
        )


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument(
        "input",
        type=str,
        help="Input file name or file names prefix"
    )
    parser.add_argument("--n", type=int, help="Number of pixels")
    parser.add_argument(
        "--supersampling-projection",
        type=int,
        default=1,
        help="Supersampling for mesh projections computation",
    )
    parser.add_argument(
        "--dset", type=str, help="Data set name, if not specified guessed from input"
    )
    parser.add_argument("--num-projections", type=int, help="Number of projections")
    parser.add_argument(
        "--out-directory",
        type=str,
        default="dataset",
        help="Output directory, result goes to 'out-directory/dset/projections'"
        "or 'out-directory/dset/truth', depending on the --make-gt switch",
    )
    parser.add_argument(
        "--pixel-size", type=float, default=[750.0], nargs="+", help="Pixel size in nm"
    )
    parser.add_argument(
        "--mesh-pixel-size", type=float, help="Physical mesh pixel size in nm"
    )
    parser.add_argument(
        "--rotation-angle", type=float, default=180, help="Total rotation angle in degrees"
    )
    parser.add_argument(
        "--lamino-angle", type=float, default=[5], nargs="+", help="Laminographic angle in degrees"
    )
    parser.add_argument(
        "--rotation-axis",
        type=str,
        choices=["y", "z"],
        default=["y"],
        nargs="+",
        help="Rotation axis (y - up, z - beam direction)",
    )
    parser.add_argument(
        "--num-devices", type=int, default=1, help="Number of compute devices to use"
    )
    parser.add_argument(
        "--num-meshes", type=int, default=1, help="Number of meshes / rotation angle"
    )
    parser.add_argument(
        "--supersampling",
        type=int,
        default=[1],
        nargs="+",
        help="Supersampling computes with n-times more pixels than usual",
    )
    parser.add_argument("--double-precision", action="store_true", help="Use double precision")
    # Ground truth related
    parser.add_argument(
        "--z-chunk",
        type=int,
        default=100,
        help="Number of ground truth slices to compute during one pass",
    )
    parser.add_argument(
        "--make-gt", action="store_true", help="Create ground truth instead of projections"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isfile(args.input) and args.num_meshes > 1:
        raise ValueError("--num-meshes > 1 can be used only if more meshes are specified")

    combinations = list(
        itertools.product(
            args.lamino_angle, args.pixel_size, args.rotation_axis, args.supersampling
        )
    )
    if args.make_gt:
        image_directory = "truth"
        file_prefix = "slice"
    else:
        image_directory = "projections"
        file_prefix = image_directory[:-1]

    file_prefix += "_{:>04}.tif"

    devices = list(range(args.num_devices))
    pool = Pool(processes=args.num_devices)

    for lamino_angle, pixel_size, rotation_axis, ss in combinations:
        # Prepare output
        if args.dset is None:
            dset = os.path.splitext(os.path.basename(args.input))[0]
        else:
            dset = args.dset
        if len(combinations) > 1:
            dset += "_lamino_angle_{:>02}_deg".format(int(lamino_angle))
            dset += "_axis_{}".format(rotation_axis)
            dset += "_ps_{:>04}_nm".format(int(pixel_size))
            dset += "_ss_{:>02}".format(ss)

        args.prefix = os.path.join(args.out_directory, dset, image_directory, file_prefix)
        args.logfile = os.path.join(args.out_directory, dset, "simulation.log")
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
        if not attr.startswith("_") and not callable(getattr(obj, attr)):
            LOG.info("{}: {}".format(attr, getattr(obj, attr)))


if __name__ == "__main__":
    main()
