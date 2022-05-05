"""Simple 4D tomography example. Two cubes rotate around the tomographic rotation axis and at the
same time move along y-axis. The total vertical displacement between rotation start and end is the
cube edge. This leads to an "incomplete" data set with increasingly more missing data in the
sinogram space from top to bottom. There is exactly one complete sinogram, the middle one.
"""
import imageio
import os
import matplotlib.pyplot as plt
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

    for i in range(num_projections):
        t = total_time / num_projections * i
        composite.move(t, clear=True)
        projection = composite.project(shape, ps).get()
        imageio.imwrite(fmt.format(i), projection)

    show(projection)
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument("--output", type=str, help="Output directory for projections.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
