"""
Composite body example
======================

Here we show how to use a composite body in order to move groups of objects around. This is possible
in two ways, either manually by translating and rotating the composite body, or automatically by
using a trajectory.

Manual
------

This example shows manual rotation of a grid of spheres with different radii around one of the
spheres. CompositeObject is used to simplify the transformations workflow.

Trajectory
----------

This example has the same result as the previous one but achieved by trajectories.


Subtrajectories
---------------

This example shows a circular global motion followed by the whole composite body and its sub-bodies,
which are cuboids following their own local linear trajectories. The sub-bodies further move along
their own trajectories.
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.base import CompositeBody
from syris.bodies.isosurfaces import MetaBall
from syris.bodies.mesh import make_cube, Mesh
from syris.geometry import Trajectory
from .util import get_default_parser, show
from .trajectory import make_circle


def _make_metaballs(args):
    metaballs = []
    radius = args.n / 16.0
    shift = 3 * radius
    for i in range(-1, 2):
        dx = i * 3 * radius
        for j in range(-1, 2):
            index = (i + 1) * 3 + (j + 1)
            # Scale the metaballs to be [0.5, 1.0] radius large.
            coeff = 0.5 + 0.5 / 8 * index
            current_radius = coeff * radius
            dy = j * 3 * radius
            traj = Trajectory([(shift + dx, shift + dy, 0)] * args.ps, pixel_size=args.ps)
            metaballs.append(MetaBall(traj, current_radius * args.ps))

    return metaballs


def make_manual_sequence(args):
    metaballs = _make_metaballs(args)
    traj = Trajectory([(args.n / 2.0, args.n / 2.0, 0)] * args.ps, pixel_size=args.ps)
    composite = CompositeBody(traj, bodies=metaballs)
    # Move the sub-bodies relative to the composite body and also move the composite body to the
    # center of the image.
    composite.move(0 * q.s)

    im = None
    d_angle = 10 * q.deg
    fmt = "Projection at rotation {:>9}"
    # Rotate around 360 deg
    for i in range(int((360 * q.deg / d_angle).magnitude) + 1):
        p = composite.project(args.shape, args.ps).get()
        if im is None:
            im = show(p, title=fmt.format(i * d_angle))
        else:
            im.axes.set_title(fmt.format(i * d_angle))
            im.set_data(p)
        plt.draw()
        # Rotation takes care of the relative rotation of the sub-bodies around the composite body.
        composite.rotate(d_angle, geom.Z_AX)

    plt.show()


def make_trajectory_sequence(args):
    # Make a small circle (1 / 10 of the pixel size), so that the composite body only rotates and
    # does not translate. Put this circle in the middle of the image.
    circle = args.n / 2 * args.ps + make_circle(n=1024).magnitude * args.ps / 10
    traj = Trajectory(circle, velocity=args.ps / q.s, pixel_size=args.ps)
    metaballs = _make_metaballs(args)
    composite = CompositeBody(traj, bodies=metaballs)

    im = None
    d_angle = 10 * q.deg
    fmt = "Projection at rotation {:>9}"
    # Rotate around 360 deg
    for i, t in enumerate(np.linspace(0, traj.time.simplified.magnitude, 37) * q.s):
        # Reset transformation matrices
        composite.clear_transformation()
        # Move to the desired position, i.e. around the circle and then each metaball moves relative
        # to the composite body.
        composite.move(t)
        p = composite.project(args.shape, args.ps).get()
        if im is None:
            im = show(p, title=fmt.format(i * d_angle))
        else:
            im.axes.set_title(fmt.format(i * d_angle))
            im.set_data(p)
        plt.draw()

    plt.show()


def make_complex_trajectory_sequence(args):
    edge = 20
    x = np.linspace(0, args.n / 2 - args.n / 4 - edge - 5, num=10)
    y = z = np.zeros(x.shape)
    # Move along x axis
    traj_x = Trajectory(list(zip(x, y, z)) * args.ps, velocity=args.ps / q.s, pixel_size=args.ps)
    # Move along y axis
    traj_y = Trajectory(list(zip(y, x, z)) * args.ps, velocity=args.ps / q.s, pixel_size=args.ps)
    # Move along both x and y axes
    traj_xy = Trajectory(list(zip(x, x, z)) * args.ps, velocity=args.ps / q.s, pixel_size=args.ps)
    # Circular trajectory of the composite body rotates around the image center and with radius
    # n / 4 pixels.
    circle = args.n / 2 * args.ps + make_circle().magnitude * args.n / 4 * args.ps
    traj_circle = Trajectory(circle, velocity=args.ps / q.s, pixel_size=args.ps)
    # Make the trajectory of the circle the same duration as the simple linear one.
    traj_circle = Trajectory(circle, velocity=traj_circle.length / traj_xy.length * args.ps / q.s)
    # three cubes in the same height and depth, shifted only along the x axis.
    traj_stationary = Trajectory([(0, 0, 0)] * args.ps)
    traj_stationary_1 = Trajectory([(-2 * edge, 0, 0)] * args.ps)
    traj_stationary_2 = Trajectory([(2 * edge, 0, 0)] * args.ps)

    cube = make_cube() / q.m * edge * args.ps
    # The cubes are elongated along y axis.
    cube[::2, :] /= 3

    mesh = Mesh(cube, traj_x, orientation=geom.Y_AX)
    mesh_2 = Mesh(cube, traj_y, orientation=geom.Y_AX)
    mesh_3 = Mesh(cube, traj_xy, orientation=geom.Y_AX)
    mesh_stationary = Mesh(cube, traj_stationary, orientation=geom.Y_AX)
    mesh_stationary_1 = Mesh(cube, traj_stationary_1, orientation=geom.Y_AX)
    mesh_stationary_2 = Mesh(cube, traj_stationary_2, orientation=geom.Y_AX)
    bodies = [mesh, mesh_2, mesh_3, mesh_stationary, mesh_stationary_1, mesh_stationary_2]
    composite = CompositeBody(traj_circle, bodies=bodies, orientation=geom.Y_AX)
    composite.bind_trajectory(args.ps)

    total_time = composite.time
    if args.t is None:
        times = np.linspace(0, 1, 100)
    else:
        if args.t < 0 or args.t > 1:
            raise ValueError("--t must be in the range [0, 1]")
        times = [args.t]

    im = None
    for index, i in enumerate(times):
        t = i * total_time
        composite.clear_transformation()
        composite.move(t)
        p = composite.project(args.shape, args.ps).get()
        if im is None:
            im = show(p, title="Projection")
        else:
            im.set_data(p)
            plt.draw()

    plt.show()


def main():
    parser = get_default_parser(__doc__)
    subparsers = parser.add_subparsers(help="sub-command help", dest="sub-commands", required=True)
    manual = subparsers.add_parser("manual", help="Manual positioning via simple transformations")
    trajectory = subparsers.add_parser("trajectory", help="Automatic positioning via trajectories")
    subtrajectories = subparsers.add_parser(
        "subtrajectories", help="Automatic positioning with " "local sub-body trajectories"
    )
    manual.set_defaults(_func=make_manual_sequence)
    trajectory.set_defaults(_func=make_trajectory_sequence)

    subtrajectories.add_argument(
        "--t",
        type=float,
        help="Time at which to compute the projection normalized to "
        "[0, 1], if not specified, complete sequence is shown",
    )
    subtrajectories.set_defaults(_func=make_complex_trajectory_sequence)

    args = parser.parse_args()
    syris.init(device_index=0)

    # Set constants
    args.n = 512
    args.shape = (args.n, args.n)
    args.ps = 1 * q.um

    args._func(args)


if __name__ == "__main__":
    main()
