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
import logging
import os
import time
import numpy as np
import quantities as q
from functools import partial
from multiprocessing import Lock, Pool
from syris.geometry import X_AX, Y_AX, Z_AX
from syris.devices.cameras import Camera
from .util import get_default_parser
import pyvista as pv

LOCK = Lock()
LOG = logging.getLogger(__name__)


def make_projection(
    mesh, camera, parallel=True, ss=1, orig_shape=None, subsampling=0
):
    from syris.imageprocessing import bin_image

    # t=None for trajectory override
    projection = mesh.project(
        camera=camera, parallel=parallel, iterations=subsampling
    )
    if ss > 1:
        projection = bin_image(projection, orig_shape, average=True)

    return projection.get()


def scan(camera, mesh, args, index=0, shift_coeff=1e4, orig_shape=None):
    """Make a scan of tomographic angles. *shift_coeff* is the coefficient multiplied by pixel size
    which shifts the triangles to get rid of faulty pixels.
    """
    psm = camera.pixel_size.simplified.magnitude
    log_fmt = "{}: {:>04}/{:>04} in {:6.2f} s, angle: {:>6.2f} deg, maxima: {}"

    # Move to the middle of the FOV
    point = (camera.shape[1] * psm / 2, camera.shape[0] * psm / 2, 0) * q.m
    if index == 0:
        LOG.info("Mesh shift: {}".format(point.rescale(q.um)))
        LOG.info(
            "Mesh shift in pixels: {}".format(
                (point / camera.pixel_size).simplified.magnitude
            )
        )

    last = None
    checked_indices = []
    bad_indices = []

    mesh.clear_transformation()
    mesh.translate(point)
    camera.clear_transformation()
    camera.translate(point)

    # New geometry setup logic
    # We use detector-to-object distance (dod)
    dod = args.sdd - args.sod
    start_point = np.array([0, 0, -dod]) * q.um
    camera.translate(start_point)
    camera.look_at(point)

    if args.rotation_axis == "z":
        lamino_angle = args.lamino_angle + 90 * q.deg
    else:
        lamino_angle = args.lamino_angle

    lamino_axis = X_AX
    tomo_axis = Y_AX if args.rotation_axis == "y" else Z_AX
    num_projs = (
        int(np.pi * args.n)
        if args.num_projections is None
        else args.num_projections
    )
    tomo_angle = args.rotation_angle / (num_projs - 1) * q.deg

    # Tilt the camera to give a lamigraphy angle
    camera.rotate_around(lamino_angle, lamino_axis, pivot=point)
    camera.look_at(point)

    for i in range(num_projs):
        st = time.time()
        projs = []
        max_vals = []

        proj = make_projection(
            mesh,
            camera,
            ss=args.supersampling,
            parallel=(not args.conebeam),
            orig_shape=orig_shape,
            subsampling=args.subsampling,
        )
        projs.append(proj)
        max_vals.append(proj.max())
        best = 0

        # 2. Check for faulty pixels (if not the first frame)
        if (
            last is not None
            and (max_vals[0] > 2 * last or np.isnan(max_vals[0]))
        ) and not args.skip_pixel_testing:
            checked_indices.append(i)
            shift_val = psm / shift_coeff
            shift_vec = camera.u * shift_val

            for shift_dir in [-1, 1]:
                temp_shift = shift_dir * shift_vec
                camera.translate(temp_shift)

                shifted_proj = make_projection(
                    mesh,
                    camera,
                    ss=args.supersampling,
                    parallel=(not args.conebeam),
                    orig_shape=orig_shape,
                )
                projs.append(shifted_proj)
                max_vals.append(shifted_proj.max())

                camera.translate(-temp_shift)

            best = np.argmin(max_vals)
            if max_vals[best] > 2 * last or np.isnan(max_vals[best]):
                bad_indices.append(i)

        duration = time.time() - st
        current_tomo_angle = (i * tomo_angle).rescale(q.deg).magnitude
        with LOCK:
            LOG.info(
                log_fmt.format(
                    index,
                    i + 1,
                    num_projs,
                    duration,
                    current_tomo_angle,
                    max_vals,
                )
            )

        imageio.imwrite(args.prefix.format(i), projs[best])

        last = max_vals[best]
        camera.rotate_around(tomo_angle, tomo_axis, pivot=point)

    # Log the summary
    with LOCK:
        LOG.info("Checked indices: {}".format(checked_indices))
        LOG.info(
            "Which map to files: {}".format(
                [args.prefix.format(i) for i in checked_indices]
            )
        )
        LOG.info("Exceeding indices: {}".format(bad_indices))
        LOG.info(
            "Which map to files: {}".format(
                [args.prefix.format(i) for i in bad_indices]
            )
        )


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
    LOG.info(
        "Mesh shift in pixels: {}".format(
            (point / args.pixel_size).simplified.magnitude
        )
    )
    mesh.translate(point)
    mesh.transform()
    mesh.sort()

    z_stack = np.empty(
        (args.supersampling,) + orig_shape, dtype=cfg.PRECISION.np_float
    )

    for i in range(0, shape[0], args.z_chunk):
        end = min(i + args.z_chunk, shape[0])
        offset = (0, i * ps.rescale(q.um).magnitude, 0) * q.um
        slices = mesh.compute_slices(
            (end - i,) + shape, ps, offset=offset
        ).get()
        LOG.info("Computing slices {}-{}".format(i, end))
        enumerated = list(enumerate(slices))[:: args.supersampling]
        for j, sl in enumerated:
            # Z-dimension downsampling
            for k in range(args.supersampling):
                z_stack[k] = bin_image(
                    slices[j + k], orig_shape, average=True, queue=queue
                ).get()
            # Sum only the slices which are present (last run might not go to the end)
            sl = np.mean(z_stack[: slices.shape[0]], axis=0)
            index = (i + j) // args.supersampling
            imageio.imwrite(args.prefix.format(index), sl)

    return sl


def process(args, device_index):
    import syris
    from syris.geometry import Trajectory
    from syris.bodies.mesh import Mesh

    syris.init(
        device_index=device_index,
        logfile=args.logfile,
        double_precision=args.double_precision,
        compute_backend="cuda",
    )

    tr = Trajectory([(0, 0, 0)] * q.um)

    if args.input:
        input = args.input
        LOG.info(f"Loading mesh from file: {input}")
    else:
        input = pv.examples.download_dragon()
        LOG.info(
            "No input file provided, loading default PyVista mesh (dragon)..."
        )

    mesh = Mesh.from_file(input, tr, center=None, unit=q.um, use_normals=True)

    if args.n:
        n = args.n
        fov = n * args.pixel_size
    else:
        fov = max([ends[1] - ends[0] for ends in mesh.extrema[:-1]]) * 1.1
        n = int(np.ceil((fov / args.pixel_size).simplified.magnitude))

    orig_shape = (n, n)

    ss = args.supersampling
    ps_ss = args.pixel_size / ss
    shape_ss = tuple([val * ss for val in orig_shape])

    camera = Camera(
        pixel_size=ps_ss,  # Use the supersampled pixel size
        shape=shape_ss,  # Use the supersampled shape
        trajectory=tr,
        source_detector_distance=args.sdd * q.um,
    )

    if args.make_gt:
        LOG.info("--- Args info ---")
        log_attributes(args)

        return make_ground_truth(args, orig_shape, mesh)
    else:
        if device_index == 0:
            LOG.info("n: {}, ps: {}, FOV: {}".format(n, args.pixel_size, fov))
            LOG.info(
                "Total rotation angle: {} deg".format(args.rotation_angle)
            )
            LOG.info("--- Mesh info ---")
            log_attributes(mesh)
            LOG.info("--- Args info ---")
            log_attributes(args)

        return scan(camera, mesh, args, orig_shape=orig_shape)


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument(
        "--input", type=str, default=None, help="Input filename of a mesh."
    )
    parser.add_argument("--n", type=int, help="Number of pixels")
    parser.add_argument(
        "--subsampling",
        type=int,
        default=0,
        help="Pixel-wise adaptive subsampling. 0 for single sample, n for n levels of adaptive subsampling.",
    )
    parser.add_argument(
        "--dset",
        type=str,
        help="Data set name, if not specified guessed from input",
    )
    parser.add_argument(
        "--num-projections", type=int, help="Number of projections"
    )
    parser.add_argument(
        "--out-directory",
        type=str,
        default="dataset",
        help="Output directory, result goes to 'out-directory/dset/projections'"
        "or 'out-directory/dset/truth', depending on the --make-gt switch",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=[750.0],
        nargs="+",
        help="Pixel size in nm",
    )
    parser.add_argument(
        "--rotation-angle",
        type=float,
        default=180,
        help="Total rotation angle in degrees",
    )
    parser.add_argument(
        "--lamino-angle",
        type=float,
        default=[30],
        nargs="+",
        help="Laminographic angle in degrees",
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
        "--conebeam", action="store_true", help="Use conebeam geometry"
    )
    parser.add_argument("--sod", type=float, help="Source to object distance")
    parser.add_argument(
        "--sdd", type=float, help="Source to detector distance"
    )
    parser.add_argument(
        "--num-devices",
        type=int,
        default=1,
        help="Number of compute devices to use",
    )
    parser.add_argument(
        "--supersampling",
        type=int,
        default=[1],
        nargs="+",
        help="Supersampling computes with n-times more pixels than usual",
    )
    parser.add_argument(
        "--double-precision", action="store_true", help="Use double precision"
    )
    # Ground truth related
    parser.add_argument(
        "--z-chunk",
        type=int,
        default=100,
        help="Number of ground truth slices to compute during one pass",
    )
    parser.add_argument(
        "--make-gt",
        action="store_true",
        help="Create ground truth instead of projections",
    )
    parser.add_argument(
        "--skip-pixel-testing",
        action="store_true",
        help="Skip the faulty pixel check (re-projection) logic.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    combinations = list(
        itertools.product(
            args.lamino_angle,
            args.pixel_size,
            args.rotation_axis,
            args.supersampling,
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
            if args.input:
                dset = os.path.splitext(os.path.basename(args.input))[0]
            else:
                dset = "dragon_default"
        else:
            dset = args.dset
        if len(combinations) > 1:
            dset += "_lamino_angle_{:>02}_deg".format(int(lamino_angle))
            dset += "_axis_{}".format(rotation_axis)
            dset += "_ps_{:>04}_nm".format(int(pixel_size))
            dset += "_ss_{:>02}".format(ss)

        args.prefix = os.path.join(
            args.out_directory, dset, image_directory, file_prefix
        )
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
