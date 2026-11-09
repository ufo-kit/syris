# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Laminography data sets with matched OpenCL and CUDA projection settings."""

import copy
import itertools
import logging
import multiprocessing
import os
import time
from functools import partial

import imageio.v2 as imageio
import numpy as np
import quantities as q

import syris.config as cfg
from syris.geometry import X_AX, Y_AX, Z_AX

from .mesh_projection import (
    Projector,
    add_mesh_options,
    cone_distances,
    detector_center,
    initialize,
    load_mesh,
    log_projection_setup,
    positive_int,
    rotation_span,
    validate_mesh_options,
)
from .util import get_default_parser

LOG = logging.getLogger(__name__)


def projection_angles(args, n):
    count = int(np.pi * n) if args.num_projections is None else args.num_projections
    return np.linspace(0, args.rotation_angle, count, endpoint=False) * q.deg


def scan(mesh, projector, args, index=0):
    """Rotate the mesh in a fixed detector frame, identically on both backends."""
    angles = projection_angles(args, projector.shape[0])
    indices = np.array_split(np.arange(len(angles)), args.num_devices)[index]
    if not len(indices):
        return None
    point = detector_center(projector.shape, projector.pixel_size * projector.ss)
    lamino_angle = args.lamino_angle
    if args.rotation_axis == "z":
        lamino_angle = lamino_angle + 90 * q.deg
    axis = Y_AX if args.rotation_axis == "y" else Z_AX

    def project(index, shift=0 * q.m):
        mesh.clear_transformation()
        center = point.copy()
        center[0] += shift
        mesh.translate(center)
        mesh.rotate(lamino_angle, X_AX)
        angle = angles[index] if args.rotation_axis == "y" else -angles[index]
        mesh.rotate(angle, axis)
        return projector.project()

    for _ in range(args.warmup):
        project(indices[0])
    LOG.info(
        "Worker %s: warm-up complete (%s projections); %s measured views",
        index,
        args.warmup,
        len(indices),
    )

    last = None
    checked_indices = []
    bad_indices = []
    durations = []
    start = time.perf_counter()
    for i in indices:
        tick = time.perf_counter()
        proj = project(i)
        maximum = proj.max()
        suspicious = not np.isfinite(maximum) or (last is not None and maximum > 2 * last)
        if args.test_pixels and suspicious:
            checked_indices.append(int(i))
            candidates = [proj]
            shift = projector.pixel_size * projector.ss / 1e4
            candidates.extend(project(i, sign * shift) for sign in (-1, 1))
            maxima = [p.max() for p in candidates]
            scores = [m if np.isfinite(m) else np.inf for m in maxima]
            best = int(np.argmin(scores))
            proj, maximum = candidates[best], maxima[best]
            if not np.isfinite(maximum) or (last is not None and maximum > 2 * last):
                bad_indices.append(int(i))
        duration = time.perf_counter() - tick
        durations.append(duration)
        LOG.info(
            "%s: %04d/%04d in %.6f s; angle %.6f deg; maximum %.8g m",
            index,
            i + 1,
            len(angles),
            duration,
            angles[i].magnitude,
            maximum,
        )
        imageio.imwrite(args.prefix.format(i), proj)
        last = maximum if np.isfinite(maximum) else None

    LOG.info(
        "Worker %s projection time (pose, build, trace, transfer, averaging): %.6f s; "
        "median: %.6f s",
        index,
        sum(durations),
        np.median(durations),
    )
    LOG.info(
        "Worker %s wall time including file output: %.6f s", index, time.perf_counter() - start
    )
    LOG.info("Retried views: %s; still suspect: %s", checked_indices, bad_indices)
    return proj


def make_ground_truth(args, shape, mesh):
    """Write occupancy slices using the shared OpenCL slicer, with CPU averaging."""
    if args.z_chunk % args.supersampling:
        raise ValueError("z_chunk must be divisible by supersampling")
    ss = args.supersampling
    ps = args.pixel_size / ss
    fine_shape = tuple(n * ss for n in shape)
    point = detector_center(shape, args.pixel_size)
    point[2] = shape[1] * args.pixel_size / 2
    mesh.clear_transformation()
    mesh.translate(point)
    mesh.transform()
    mesh.sort()
    for i in range(0, fine_shape[0], args.z_chunk):
        end = min(i + args.z_chunk, fine_shape[0])
        offset = np.array([0, i * float(ps.rescale(q.m).magnitude), 0]) * q.m
        slices = mesh.compute_slices((end - i,) + fine_shape, ps, offset=offset).get()
        # The slicer's stack axis is mesh y; each image has (z, x) axes.
        slices = (
            slices.astype(cfg.PRECISION.np_float)
            .reshape(
                (end - i) // ss,
                ss,
                shape[0],
                ss,
                shape[1],
                ss,
            )
            .mean(axis=(1, 3, 5))
        )
        LOG.info("Computing slices %s-%s", i // ss, end // ss)
        for j, sl in enumerate(slices):
            imageio.imwrite(args.prefix.format(i // ss + j), sl)
    return sl


def process(args, index=0):
    initialize(args, args.device_index + index)
    mesh = load_mesh(args)
    span = rotation_span(mesh)
    if args.conebeam:
        args.sod, args.sdd = cone_distances(args, span)
        span *= float((args.sdd / args.sod).simplified.magnitude)
    n = args.n
    if n is None:
        n = int(np.ceil(float((span * args.margin / args.pixel_size).simplified.magnitude)))
    shape = (n, n)
    if args.make_gt:
        return make_ground_truth(args, shape, mesh)
    log_projection_setup(args, shape, args.pixel_size)
    LOG.info(
        "Angles: %s views over [0, %s) degrees; axis %s; laminography angle %s",
        len(projection_angles(args, n)),
        args.rotation_angle,
        args.rotation_axis,
        args.lamino_angle,
    )
    return scan(mesh, Projector(mesh, shape, args.pixel_size, args), args, index=index)


def parse_args(argv=None, backend="opencl"):
    parser = get_default_parser(__doc__)
    add_mesh_options(parser)
    parser.add_argument("input_path", nargs="?", help="Mesh file (alias for --input)")
    parser.add_argument("--input", help="Mesh file; omitted: built-in cube")
    parser.add_argument(
        "--n", type=positive_int, help="Detector width and height; omitted: fit mesh"
    )
    parser.add_argument("--margin", type=float, default=1.2, help="Automatic FOV margin factor")
    parser.add_argument("--dset", help="Data set name; omitted: mesh filename or cube_default")
    parser.add_argument("--num-projections", type=positive_int)
    parser.add_argument("--out-directory", default="dataset")
    parser.add_argument(
        "--pixel-size", type=float, default=[750.0], nargs="+", help="Pixel size in nm"
    )
    parser.add_argument(
        "--rotation-angle", type=float, default=180, help="Angular range in degrees"
    )
    parser.add_argument(
        "--lamino-angle", type=float, default=[5], nargs="+", help="Tilt in degrees"
    )
    parser.add_argument("--rotation-axis", choices=["y", "z"], default=["y"], nargs="+")
    parser.add_argument("--num-devices", type=positive_int, default=1)
    parser.add_argument(
        "--supersampling",
        type=positive_int,
        default=[1],
        nargs="+",
        help="Uniform samples per pixel axis; average the resulting s by s samples",
    )
    parser.add_argument("--z-chunk", type=positive_int, default=100)
    parser.add_argument(
        "--make-gt", action="store_true", help="Write occupancy slices using OpenCL"
    )
    testing = parser.add_mutually_exclusive_group()
    testing.add_argument(
        "--test-pixels",
        action="store_true",
        help="Opt in to shifted retries of suspect projections",
    )
    testing.add_argument(
        "--skip-pixel-testing",
        action="store_false",
        dest="test_pixels",
        help="Disable shifted retries (the default)",
    )
    parser.set_defaults(test_pixels=False)
    args = parser.parse_args(argv)
    validate_mesh_options(parser, args, backend)
    if args.input is not None and args.input_path is not None:
        parser.error("Specify the mesh either positionally or with --input")
    args.input = args.input if args.input is not None else args.input_path
    if args.margin <= 0 or any(ps <= 0 for ps in args.pixel_size):
        parser.error("--margin and --pixel-size must be positive")
    if args.make_gt and args.num_devices != 1:
        parser.error("--make-gt uses one OpenCL device; set --num-devices 1")
    if args.make_gt and any(args.z_chunk % ss for ss in args.supersampling):
        parser.error("--z-chunk must be divisible by every --supersampling factor")
    return args


def main(argv=None, backend="opencl"):
    args = parse_args(argv, backend=backend)
    combinations = itertools.product(
        args.lamino_angle,
        args.pixel_size,
        args.rotation_axis,
        args.supersampling,
    )
    multiple = (
        len(args.lamino_angle)
        * len(args.pixel_size)
        * len(args.rotation_axis)
        * len(args.supersampling)
    ) > 1
    image_directory, file_prefix = (
        ("truth", "slice") if args.make_gt else ("projections", "projection")
    )
    result = None
    for angle, pixel_size, axis, ss in combinations:
        run = copy.copy(args)
        dset = args.dset
        if dset is None:
            dset = (
                os.path.splitext(os.path.basename(args.input))[0] if args.input else "cube_default"
            )
        if multiple:
            dset += f"_lamino_angle_{angle:g}_deg_axis_{axis}_ps_{pixel_size:g}_nm_ss_{ss:02}"
        directory = os.path.join(args.out_directory, dset, image_directory)
        os.makedirs(directory, exist_ok=True)
        run.prefix = os.path.join(directory, file_prefix + "_{:04}.tif")
        run.logfile = os.path.join(args.out_directory, dset, "simulation.log")
        run.pixel_size = pixel_size * q.nm
        run.lamino_angle = angle * q.deg
        run.rotation_axis = axis
        run.supersampling = ss
        if args.num_devices == 1:
            result = process(run)
        else:
            # Spawn avoids inheriting GPU contexts; each worker owns disjoint file indices.
            with multiprocessing.get_context("spawn").Pool(args.num_devices) as pool:
                pool.map(partial(process, run), range(args.num_devices))
    return result


if __name__ == "__main__":
    main()
