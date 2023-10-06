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

"""Simple 4D tomography example. Two cubes rotate around the tomographic rotation axis and at the
same time move along y-axis. The total vertical displacement between rotation start and end is the
cube edge. This leads to an "incomplete" data set with increasingly more missing data in the
sinogram space from top to bottom. There is exactly one complete sinogram, the middle one.
"""
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.base import CompositeBody
from syris.bodies.mesh import Mesh, make_cube
from .trajectory import make_circle
from .util import get_default_parser, show


def make_cube_body(n, ps, cube_edge, phase_shift=None):
    fov = n * ps
    triangles = make_cube().magnitude * cube_edge / 2
    # Rotation around the vertical axis
    points = make_circle(axis="y", overall_angle=np.pi * q.rad, phase_shift=phase_shift).magnitude
    points = points * fov / 4 + [n // 2, 0, 0] * ps
    trajectory = geom.Trajectory(points, pixel_size=ps, velocity=ps / q.s)
    # *orientation* aligns the object with the trajectory derivative
    mesh = Mesh(triangles, trajectory, orientation=geom.Z_AX)

    return mesh


def main():
    syris.init()
    args = parse_args()
    n = 256
    shape = (n, n)
    ps = 1 * q.um
    num_projections = None
    cube_edge = n / 4 * ps
    fmt = os.path.join(args.output, "projection_{:04}.tif")

    x = np.linspace(n // 4 + n // 8, 3 * n // 4 + n // 8, num=10)
    y = z = np.zeros(x.shape)
    cube_0 = make_cube_body(n, ps, cube_edge)
    cube_1 = make_cube_body(n, ps, cube_edge, phase_shift=np.pi * q.rad)
    # Vertical motion component has such velocity that the cubes are displaced by their edge length
    # 1 pixel for making sure we have one "complete" sinogram
    velocity = (cube_edge - ps) / cube_0.trajectory.time
    traj_y = geom.Trajectory(list(zip(y, x, z)) * ps, pixel_size=ps, velocity=velocity)
    composite = CompositeBody(traj_y, bodies=[cube_0, cube_1])
    # Total time is the rotation time because we want one tomographic data set
    total_time = cube_0.trajectory.time

    dt = composite.get_next_time(0 * q.s, ps)
    if num_projections is None:
        num_projections = int(np.ceil((total_time / dt).simplified.magnitude))

    print("              num_projs:", num_projections)
    print("          rotation time:", cube_0.trajectory.time)
    print("   vertical motion time:", traj_y.time)
    print("        simulation time:", total_time)

    if args.create_animation:
        projections = np.zeros((num_projections, n, n))

    for i in range(num_projections):
        t = total_time / num_projections * i
        composite.move(t, clear=True)
        projection = composite.project(shape, ps).get().astype("float32")
        imageio.imwrite(fmt.format(i), projection)
        if args.create_animation:
            projections[i] = projection

    show(projection)
    plt.show()
    if args.create_animation:
        # Animate Projections
        writergif = animation.PillowWriter(fps=20)
        projection_animation = animate_volume(projections, title="Projection: ", axis=0)
        f = args.output + "projections.gif"
        projection_animation.save(f, writer=writergif)
        plt.close("all")

        # Animate Sinogram
        sinogram_animation = animate_volume(projections, title="Sinogram: ", axis=1)
        f = args.output + "sinogram.gif"
        sinogram_animation.save(f, writer=writergif)


def animate_volume(volume, title="Projection: ", axis=0):
    fig = plt.figure()
    ax = plt.gca()
    ax.set_title("")
    im_1 = plt.imshow(np.take(volume, 0, axis=axis),
                      vmin=np.min(volume), vmax=np.max(volume), cmap="Greys")
    plt.colorbar()

    def update(i):
        im_1.set_data(np.take(volume, i, axis=axis))
        ax.set_title(title + str(i))
        return im_1,

    return animation.FuncAnimation(fig, update, blit=True, repeat=True, frames=volume.shape[axis])


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument("--output", type=str, help="Output directory for projections.")
    parser.add_argument("--create_animation", action='store_true',
                        help="Creates an animations of the projections and sinograms.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
