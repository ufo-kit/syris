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

"""Mesh projection and slice."""
import imageio
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
import tqdm
from syris.bodies.mesh import Mesh, MeshReader, PyvistaReader
from syris.devices.cameras import Camera
from .util import get_default_parser, show
import pyvista as pv


LOG = logging.getLogger(__name__)


def main():
    """Main function."""
    plotter = pv.Plotter()

    args = parse_args()
    syris.init(loglevel=logging.INFO, double_precision=args.double_precision)

    units = q.Quantity(1, args.units)
    tr = geom.Trajectory([(0, 0, 0)] * units)

    reader = MeshReader(PyvistaReader("/home/lt0649/Dev/syris/examples/monkey.obj", units.units))
    vertices, normals, bounds = reader.scene

    mesh = Mesh(vertices, tr, center=args.center, iterations=args.supersampling, bounds=bounds, normals=normals)
    LOG.info("Number of triangles: {}".format(mesh.num_triangles))

    mesh.visualize(plotter)

    shape = (args.n, args.n)
    if args.pixel_size is None:
        if args.input is None:
            fov = 4.0 * units
        else:
            # Maximum sample size in x and y direction
            max_diff = np.max(mesh.extrema[:-1, 1] - mesh.extrema[:-1, 0])
            fov = max_diff
        fov *= args.margin
        args.pixel_size = fov / args.n
    else:
        fov = args.n * args.pixel_size

    if args.translate is None:
        translate = (fov.simplified.magnitude / 2.0, fov.simplified.magnitude / 2.0, 0) * q.m
    else:
        translate = (
            args.translate[0].simplified.magnitude,
            args.translate[1].simplified.magnitude,
            0,
        ) * q.m
    LOG.info("Translation: {}".format(translate.rescale(q.um)))

    mesh.translate(translate)
    # mesh.rotate(args.x_rotate, geom.X_AX)

    camera = Camera(
        11 * q.um, 0.1, 530.0, 23.0, 12, (2048, 2048), focal_length=18 * q.um, coordinate_system=mesh.child_cs
    )
    coords = [0, 0, 3] * q.um
    camera.translate(coords)
    camera.visualize(plotter, cmap="plasma")

    fmt = "n: {}, pixel size: {}, FOV: {}"
    LOG.info(fmt.format(args.n, args.pixel_size.rescale(q.um), fov.rescale(q.um)))
    st = time.time()
    for i in tqdm.tqdm(range(args.num_y_rotations)):
        proj = mesh.project(shape, args.pixel_size, camera=camera, parallel=False)
        if args.projection_filename is not None:
            imageio.imwrite(args.projection_filename + f"_{i:>05}.tif", proj)
        mesh.rotate(args.y_rotate, geom.Z_AX)


    LOG.info("Duration: {} s".format(time.time() - st))
    offset = (0, translate[1].simplified, -(fov / 2.0).simplified) * q.m

    if args.compute_slice:
        sl = mesh.compute_slices((1,) + shape, args.pixel_size, offset=offset).get()[0]
        if args.slice_filename is not None:
            imageio.imwrite(args.slice_filename, sl)
        show(sl, title="Slice at y = {}".format(args.n / 2))

    show(proj, title="Projection")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = get_default_parser(__doc__)

    parser.add_argument("--input", type=str, help="Input .obj file")
    parser.add_argument("--units", type=str, default="m", help="Mesh physical units")
    parser.add_argument("--n", type=int, default=256, help="Number of pixels")
    parser.add_argument(
        "--supersampling", type=int, default=1, help="Supersampling for mesh computation"
    )
    parser.add_argument("--pixel-size", type=float, help="Pixel size in um")
    parser.add_argument("--center", type=str, help="Mesh centering on creation")
    parser.add_argument("--translate", type=float, nargs=2, help="Translation as (x, y) in um")
    parser.add_argument("--x-rotate", type=float, default=0.0, help="Rotation around x axis [deg]")
    parser.add_argument("--y-rotate", type=float, default=0.0, help="Rotation around y axis [deg]")
    parser.add_argument(
        "--num-y-rotations",
        type=int,
        default=1,
        help="How many times rotate around y axis (tomography simulation)"
    )
    parser.add_argument(
        "--margin", type=float, default=1.0, help="Margin in factor of the full FOV"
    )
    parser.add_argument(
        "--projection-filename",
        type=str,
        help="Save projection to this filename prefix (.tif is appended)"
    )
    parser.add_argument("--compute-slice", action="store_true", help="Compute also one slice")
    parser.add_argument("--slice-filename", type=str, help="Save slice to this filename")
    parser.add_argument("--double-precision", action="store_true", help="Use double precision")

    args = parser.parse_args()
    if args.pixel_size is not None:
        args.pixel_size = args.pixel_size * q.um
    args.x_rotate = args.x_rotate * q.deg
    args.y_rotate = args.y_rotate * q.deg
    if args.translate is not None:
        args.translate = args.translate * q.um

    return args


if __name__ == "__main__":
    main()
