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
