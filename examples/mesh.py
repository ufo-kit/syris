"""Mesh projection and slice."""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh, make_cube, read_blender_obj


LOG = logging.getLogger(__name__)


def main():
    """Main function."""
    args = parse_args()
    syris.init()
    triangles = make_cube() if args.input is None else read_blender_obj(args.input) * q.m
    tr = geom.Trajectory([(0, 0, 0)] * q.m)
    mesh = Mesh(triangles, tr)

    shape = (args.n, args.n)
    if args.pixel_size is None:
        if args.input is None:
            fov = 4. * q.m
        else:
            # Maximum sample size in x and y direction
            max_diff = np.max(mesh.extrema[:-1, 1] - mesh.extrema[:-1, 0])
            fov = max_diff
        fov *= args.margin
        args.pixel_size = fov / args.n
    else:
        fov = args.n * args.pixel_size

    center = (fov.simplified.magnitude / 2., fov.simplified.magnitude / 2., 0) * q.m
    mesh.translate(center)
    mesh.rotate(args.y_rotate, geom.Y_AX)
    mesh.rotate(args.x_rotate, geom.X_AX)
    fmt = 'n: {}, pixel size: {}, FOV: {}'
    LOG.info(fmt.format(args.n, args.pixel_size.simplified, fov.simplified))

    proj = mesh.project(shape, args.pixel_size, t=None).get()
    offset = syris.gpu.util.make_vfloat3(0, center[1].simplified, -(fov / 2.).simplified)
    sl = mesh.compute_slices((1,) + shape, args.pixel_size, offset=offset).get()[0]

    plt.figure()
    plt.imshow(proj)
    plt.title('Projection')
    plt.colorbar()

    plt.figure()
    plt.imshow(sl)
    plt.title('Slice at y = {}'.format(args.n / 2))
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input', type=str, help='Input .obj file')
    parser.add_argument('--n', type=int, default=256, help='Number of pixels')
    parser.add_argument('--pixel-size', type=float, help='Pixel size in um')
    parser.add_argument('--x-rotate', type=float, default=0., help='Rotation around x axis [deg]')
    parser.add_argument('--y-rotate', type=float, default=0., help='Rotation around y axis [deg]')
    parser.add_argument('--margin', type=float, default=1., help='Margin in factor of the full FOV')

    args = parser.parse_args()
    if args.pixel_size is not None:
        args.pixel_size = args.pixel_size * q.um
    args.x_rotate = args.x_rotate * q.deg
    args.y_rotate = args.y_rotate * q.deg

    return args


if __name__ == '__main__':
    main()
