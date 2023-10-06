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

"""Example of a trajectory which simulates tomographic rotation."""
import matplotlib.pyplot as plt
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh, make_cube
from .trajectory import make_circle
from .util import show


def main():
    syris.init()
    n = 256
    shape = (n, n)
    ps = 1 * q.um
    fov = n * ps
    triangles = make_cube().magnitude * n / 8.0 * ps
    # Rotation around the vertical axis
    points = make_circle(axis="y").magnitude * fov / 30000 + fov / 2
    trajectory = geom.Trajectory(points, pixel_size=ps, velocity=10 * q.um / q.s)
    # *orientation* aligns the object with the trajectory derivative
    mesh = Mesh(triangles, trajectory, orientation=geom.Z_AX)
    # Compute projection at the angle Pi/4
    projection = mesh.project(shape, ps, t=trajectory.time / 8).get()

    show(projection)
    plt.show()


if __name__ == "__main__":
    main()
