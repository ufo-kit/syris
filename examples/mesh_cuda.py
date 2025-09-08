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
from syris.geometry import Trajectory
from syris.devices.cameras import Camera
import tqdm
from syris.bodies.mesh import Mesh
from .util import get_default_parser, show

LOG = logging.getLogger(__name__)

def main():
    """Main function."""
    args = parse_args()
    syris.init(loglevel=logging.INFO, double_precision=args.double_precision, compute_backend="cuda")

    pixel_units = q.Quantity(1, args.pixel_size_units)
    mesh_units = q.Quantity(1, args.mesh_units)

    tr = geom.Trajectory([(0, 0, 0)] * mesh_units)
    mesh = Mesh.from_file(args.input, tr, center=args.center, iterations=args.supersampling, unit=mesh_units)
    LOG.info("Number of triangles: {}".format(mesh.num_triangles))

    if args.pixel_size is None:
        LOG.info("Pixel size not provided, calculating automatically...")
        max_mesh_span = np.max(mesh.extrema[:-1, 1] - mesh.extrema[:-1, 0]).rescale(mesh_units)
        fov = max_mesh_span * args.margin
        args.pixel_size = fov / args.n
    else:
        fov = args.n * args.pixel_size


    # --- CAMERA SETUP ---
    camera_trajectory = geom.Trajectory([(0, 0, 0)] * mesh_units)
    camera = Camera(pixel_size=args.pixel_size, shape=args.n, trajectory=camera_trajectory)
    camera.translate([fov, fov, -10] * mesh_units)

    print("\n--- Simulation Setup ---")
    LOG.info(f"Image Resolution: {args.n}x{args.n} pixels")
    LOG.info(f"Final Pixel Size: {args.pixel_size.rescale(pixel_units):.4f}")
    LOG.info(f"Field of View (FOV): {fov.rescale(mesh_units):.4f}")
    LOG.info(f"Mesh Bounding Box (min): {mesh.extrema[:, 0].rescale(mesh_units)}")
    LOG.info(f"Mesh Bounding Box (max): {mesh.extrema[:, 1].rescale(mesh_units)}")
    LOG.info(f"Camera Position: {camera.position.rescale(mesh_units)}\n")

    st = time.time()
    for i in tqdm.tqdm(range(args.num_y_rotations)):
        proj = mesh.project(camera=camera, parallel=True)
        if args.projection_filename is not None:
            imageio.imwrite(args.projection_filename + f"_{i:>05}.tif", proj)
        mesh.rotate(args.y_rotate, geom.Y_AX)

    LOG.info("Duration: {} s".format(time.time() - st))

    show(proj, title="Projection")
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = get_default_parser(__doc__)

    # --- Mesh Arguments ---
    parser.add_argument("--input", type=str, required=True, help="Input .obj file")
    parser.add_argument("--mesh-units", type=str, default="um", help="Physical units of the mesh file (e.g., 'm', 'cm', 'um')")
    parser.add_argument("--center", type=str, default="bbox", help="Mesh centering on creation")

    # --- Detector Arguments ---
    parser.add_argument("--n", type=int, default=2048, help="Number of pixels in each dimension")
    parser.add_argument("--pixel-size", type=float, default=None, help="[Optional] Size of a single pixel. If not provided, it's calculated automatically.")
    parser.add_argument("--pixel-size-units", type=str, default="um", help="Units for the pixel size if specified (e.g., 'm', 'cm', 'um')")
    parser.add_argument("--margin", type=float, default=1.2, help="Margin factor for automatic FOV calculation (e.g., 1.2 for a 20%% margin)")
    parser.add_argument("--supersampling", type=int, default=1, help="Supersampling for mesh computation")
    parser.add_argument("--x-rotate", type=float, default=0.0, help="Rotation around x axis [deg]")
    parser.add_argument("--y-rotate", type=float, default=0.0, help="Rotation around y axis [deg]")
    parser.add_argument("--num-y-rotations", type=int, default=1, help="How many times to rotate around y axis")
    parser.add_argument("--projection-filename", type=str, help="Save projection to this filename prefix")
    parser.add_argument("--double-precision", action="store_true", help="Use double precision")

    args = parser.parse_args()

    if args.pixel_size is not None:
        args.pixel_size = args.pixel_size * q.Quantity(1, args.pixel_size_units)
        
    args.x_rotate = args.x_rotate * q.deg
    args.y_rotate = args.y_rotate * q.deg

    return args


if __name__ == "__main__":
    main()
