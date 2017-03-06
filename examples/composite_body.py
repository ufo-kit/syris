"""CompositeBody example shows how to combine multiple objects to follow a global trajectory. A
sphere travels along the positive x axis and a cube along a line y = x. A composite body encompasses
both of the bodies and moves along y. Thus, when we move and project the composite body we will see
the motion of the objects combined with the motion of the composite body.
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.bodies.base import CompositeBody
from syris.bodies.isosurfaces import MetaBall
from syris.bodies.mesh import make_cube, Mesh
from syris.geometry import Trajectory
from util import get_default_parser, show


def main():
    args = parse_args()
    syris.init(device_index=0)
    n = 512
    shape = (n, n)
    ps = 1 * q.um

    x = np.linspace(0, n, num=10)
    y = z = np.zeros(x.shape)
    traj_x = Trajectory(zip(x, y, z) * ps, velocity=ps / q.s)
    traj_y = Trajectory(zip(y, x, z) * ps, velocity=ps / q.s)
    traj_xy = Trajectory(zip(n - x, x, z) * ps, velocity=ps / q.s)
    mb = MetaBall(traj_x, n * ps / 16)
    cube = make_cube() / q.m * 16 * ps
    mesh = Mesh(cube, traj_xy)
    composite = CompositeBody(traj_y, bodies=[mb, mesh])
    composite.bind_trajectory(ps)
    t = args.t * n * q.s

    composite.move(t)
    p = composite.project(shape, ps).get()
    show(p, title='Projection')

    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--t', type=float, default=0.5,
                        help='Time at which to compute the projection normalized to [0, 1]')
    args = parser.parse_args()
    if args.t < 0 or args.t > 1:
        raise ValueError('--t must be in the range [0, 1]')

    return args


if __name__ == '__main__':
    main()
