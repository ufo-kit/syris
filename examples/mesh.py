"""A rotated cube."""
import matplotlib.pyplot as plt
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh, make_cube


def main():
    syris.init()
    cube = make_cube()
    tr = geom.Trajectory([(0, 0, 0)] * q.m)
    mesh = Mesh(cube, tr)

    n = 256
    shape = (n, n)
    fov = 4. * q.m
    ps = fov / n

    center = (fov.magnitude / 2., fov.magnitude / 2., 0) * q.m
    mesh.translate(center)
    mesh.rotate(45 * q.deg, geom.Y_AX)
    mesh.rotate(35 * q.deg, geom.X_AX)

    proj = mesh.project(shape, ps).get()
    offset = syris.gpu.util.make_vfloat3(0, center[1].rescale(q.um), -(fov / 2.).rescale(q.um))
    sl = mesh.compute_slices((1,) + shape, ps, offset=offset).get()[0]

    plt.figure()
    plt.imshow(proj)
    plt.title('Projection')

    plt.figure()
    plt.imshow(sl)
    plt.title('Slice at y = {}'.format(n / 2))
    plt.show()


if __name__ == '__main__':
    main()
