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

"""Rotate around and project a mesh in parallel or conebeams setups"""

import imageio
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
from syris.devices.cameras import Camera
import tqdm
from syris.bodies.mesh import Mesh
from .util import get_default_parser, show

LOG = logging.getLogger(__name__)


def main():
    """Main function."""
    args = parse_args()

    pixel_units = q.Quantity(1, args.pixel_size_units)
    mesh_units = q.Quantity(1, args.mesh_units)

    syris.init(
        loglevel=logging.INFO,
        double_precision=args.double_precision,
        compute_backend="cuda",
    )

    tr = geom.Trajectory([(0, 0, 0)] * mesh_units)
    mesh = Mesh.from_file(
        args.input, tr, center=args.center, unit=mesh_units, use_normals=True
    )
    LOG.info(f"Number of triangles: {mesh.num_triangles}")

    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds

    width = xmax - xmin
    height = ymax - ymin
    max_span = np.sqrt(width**2 + height**2) * mesh_units

    if args.conebeam:
        LOG.info("Setting up for CONE-BEAM projection...")

        if args.sod is None:
            args.sod = max_span * 2
            LOG.info(f"SOD not provided, defaulting to 2 * object span: {args.sod:.2f}")

        if args.sdd is None:
            args.sdd = args.sod + max_span * 1.5
            LOG.info(
                f"SDD not provided, defaulting to SOD + 1.5 * object span: {args.sdd:.2f}"
            )

        magnification = args.sdd / args.sod
        LOG.info(f"Magnification: {magnification:.2f}x")

        if args.pixel_size is None:
            fov = (max_span * magnification) * args.margin
            LOG.info(f"Required FOV on detector: {fov:.2f} {mesh_units}")
            args.pixel_size = fov / args.n

        camera_z_position = args.sdd - args.sod
        sdd_for_camera = args.sdd
    else:
        LOG.info("Setting up for PARALLEL-BEAM projection...")
        magnification = 1.0

        if args.pixel_size is None:
            fov = max_span * args.margin
            LOG.info(f"Required FOV on detector: {fov:.2f} {mesh_units}")
            args.pixel_size = fov / args.n

        camera_z_position = max_span
        sdd_for_camera = 0 * mesh_units

    fov = args.n * args.pixel_size

    camera_trajectory = geom.Trajectory([(0, 0, 0)] * mesh_units)
    camera = Camera(
        pixel_size=args.pixel_size,
        shape=args.n,
        trajectory=camera_trajectory,
        source_detector_distance=sdd_for_camera,
    )

    camera.translate([0, 0, -camera_z_position.magnitude] * mesh_units)

    print("\n--- Simulation Setup ---")
    LOG.info(f"Projection Mode: {'Cone-Beam' if args.conebeam else 'Parallel-Beam'}")
    LOG.info(f"Image Resolution: {args.n}x{args.n} pixels")
    LOG.info(f"Final Pixel Size: {args.pixel_size.rescale(pixel_units):.4f}")
    LOG.info(f"Field of View (FOV): {fov.rescale(mesh_units):.4f}")
    if args.conebeam:
        LOG.info(f"SOD: {args.sod}, SDD: {args.sdd}")
    LOG.info(f"Mesh Bounding Box (min): {mesh.extrema[:, 0].rescale(mesh_units)}")
    LOG.info(f"Mesh Bounding Box (max): {mesh.extrema[:, 1].rescale(mesh_units)}")
    LOG.info(f"Initial Camera Position: {camera.position.rescale(mesh_units)}\n")

    origin = -camera.position

    st = time.time()
    for i in tqdm.tqdm(range(args.num_y_rotations)):
        proj = mesh.project(
            camera=camera, parallel=not args.conebeam, iterations=args.supersampling
        ).astype(np.float32)

        if args.projection_filename is not None:
            imageio.imwrite(args.projection_filename + f"_{i:>05}.tif", proj)

        camera.rotate(args.y_rotate, geom.Y_AX, shift=origin)

    LOG.info(f"Duration: {time.time() - st:.2f} s")

    show(proj, title="Projection")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = get_default_parser(__doc__)

    # --- Mesh Arguments ---
    parser.add_argument("--input", type=str, required=True, help="Input .obj file")
    parser.add_argument(
        "--mesh-units",
        type=str,
        default="um",
        help="Physical units of the mesh file (e.g., 'm', 'cm', 'um')",
    )
    parser.add_argument(
        "--center", type=str, default=None, help="Mesh centering on creation"
    )

    # --- Detector Arguments ---
    parser.add_argument(
        "--n", type=int, default=2048, help="Number of pixels in each dimension"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="[Optional] Size of a single pixel. If not provided, it's calculated automatically.",
    )
    parser.add_argument(
        "--pixel-size-units",
        type=str,
        default="um",
        help="Units for the pixel size if specified (e.g., 'm', 'cm', 'um')",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.2,
        help="Margin factor for automatic FOV calculation (e.g., 1.2 for a 20%% margin)",
    )
    parser.add_argument(
        "--supersampling",
        type=int,
        default=1,
        help="Supersampling for mesh computation",
    )
    parser.add_argument(
        "--x-rotate", type=float, default=0.0, help="Rotation around x axis [deg]"
    )
    parser.add_argument(
        "--y-rotate", type=float, default=0.0, help="Rotation around y axis [deg]"
    )
    parser.add_argument(
        "--num-y-rotations",
        type=int,
        default=1,
        help="How many times to rotate around y axis",
    )
    parser.add_argument(
        "--projection-filename",
        type=str,
        help="Save projection to this filename prefix",
    )
    parser.add_argument(
        "--double-precision", action="store_true", help="Use double precision"
    )
    parser.add_argument("--conebeam", action="store_true", help="Use conebeam geometry")
    parser.add_argument("--sod", type=float, help="Source to object distance")
    parser.add_argument("--sdd", type=float, help="Source to detector distance")

    args = parser.parse_args()

    if args.pixel_size is not None:
        args.pixel_size = args.pixel_size * q.Quantity(1, args.pixel_size_units)

    args.x_rotate = args.x_rotate * q.deg
    args.y_rotate = args.y_rotate * q.deg

    return args


if __name__ == "__main__":
    main()
