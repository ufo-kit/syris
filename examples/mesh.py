# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Rotate and project a mesh with OpenCL; mesh_cuda uses the same workflow."""

import logging
import time

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import tqdm

from syris.geometry import X_AX, Y_AX

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
from .util import get_default_parser, show

LOG = logging.getLogger(__name__)


def main(argv=None, backend="opencl"):
    args = parse_args(argv, backend=backend)
    initialize(args, args.device_index)
    mesh = load_mesh(args)
    shape = (args.n_y, args.n_x)
    span = rotation_span(mesh)
    magnification = 1
    if args.conebeam:
        args.sod, args.sdd = cone_distances(args, span)
        magnification = float((args.sdd / args.sod).simplified.magnitude)
    pixel_size = args.pixel_size
    if pixel_size is None:
        pixel_size = span * args.margin * magnification / min(shape)
    center = detector_center(shape, pixel_size)
    if args.translate is not None:
        center[:2] = args.translate
    projector = Projector(mesh, shape, pixel_size, args)
    log_projection_setup(args, shape, pixel_size)

    def set_pose(index):
        mesh.clear_transformation()
        mesh.translate(center)
        mesh.rotate(args.x_rotate, X_AX)
        mesh.rotate(index * args.y_rotate, Y_AX)

    for _ in range(args.warmup):
        set_pose(0)
        projector.project()
    LOG.info("Warm-up complete: %s untimed projection(s)", args.warmup)
    durations = []
    start = time.perf_counter()
    for i in tqdm.tqdm(range(args.num_y_rotations)):
        tick = time.perf_counter()
        set_pose(i)
        proj = projector.project()
        durations.append(time.perf_counter() - tick)
        if args.projection_filename is not None:
            imageio.imwrite(f"{args.projection_filename}_{i:05}.tif", proj)
    LOG.info(
        "Projection time (pose, build, trace, transfer, averaging): %.6f s; median: %.6f s",
        sum(durations),
        np.median(durations),
    )
    LOG.info("Wall time including file output: %.6f s", time.perf_counter() - start)

    if args.compute_slice:
        # The slicer is OpenCL-only, including when invoked through mesh_cuda.
        LOG.info("Slice generation uses OpenCL in both examples")
        mesh.transform()
        mesh.sort()
        offset = (
            np.array(
                [
                    0,
                    float(center[1].rescale(q.m).magnitude),
                    -shape[0] * float(pixel_size.rescale(q.m).magnitude) / 2,
                ]
            )
            * q.m
        )
        sl = mesh.compute_slices((1,) + shape, pixel_size, offset=offset).get()[0]
        if args.slice_filename is not None:
            imageio.imwrite(args.slice_filename, sl)
        if not args.no_show:
            show(sl, title="Slice")
    if not args.no_show:
        show(proj, title="Projection thickness [m]")
        plt.show()
    return proj


def parse_args(argv=None, backend="opencl"):
    """Both entry points accept the same geometry and sampling arguments."""
    parser = get_default_parser(__doc__)
    add_mesh_options(parser)
    parser.add_argument("--input", help="Mesh file; omitted: built-in cube")
    parser.add_argument("--n", type=positive_int, default=256, help="Default detector width/height")
    parser.add_argument("--n-y", type=positive_int, help="Detector rows (overrides --n)")
    parser.add_argument("--n-x", type=positive_int, help="Detector columns (overrides --n)")
    parser.add_argument("--pixel-size", type=float, help="Pixel size; omitted: fit all rotations")
    parser.add_argument("--pixel-size-units", default="um")
    parser.add_argument("--margin", type=float, default=1.2, help="Automatic FOV margin factor")
    parser.add_argument(
        "--supersampling",
        type=positive_int,
        default=1,
        help="Uniform samples per pixel axis; average the resulting s by s samples",
    )
    parser.add_argument("--translate", type=float, nargs=2, help="Mesh center (x, y) in um")
    parser.add_argument("--x-rotate", type=float, default=0, help="Initial x rotation in degrees")
    parser.add_argument("--y-rotate", type=float, default=0, help="Y rotation step in degrees")
    parser.add_argument("--num-y-rotations", type=positive_int, default=1)
    parser.add_argument("--projection-filename", help="TIFF prefix; appends _00000.tif, etc.")
    parser.add_argument("--compute-slice", action="store_true", help="Also compute an OpenCL slice")
    parser.add_argument("--slice-filename", help="Slice output filename")
    parser.add_argument("--no-show", action="store_true", help="Do not open plotting windows")
    args = parser.parse_args(argv)
    validate_mesh_options(parser, args, backend)
    if args.margin <= 0 or (args.pixel_size is not None and args.pixel_size <= 0):
        parser.error("--margin and --pixel-size must be positive")
    args.n_y = args.n if args.n_y is None else args.n_y
    args.n_x = args.n if args.n_x is None else args.n_x
    if args.pixel_size is not None:
        args.pixel_size *= q.Quantity(1, args.pixel_size_units)
    args.x_rotate *= q.deg
    args.y_rotate *= q.deg
    if args.translate is not None:
        args.translate = np.array(args.translate) * q.um
    return args


if __name__ == "__main__":
    main()
