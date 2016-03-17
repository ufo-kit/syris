"""Mesh projection and slice."""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.geometry as geom
from syris.bodies.mesh import Mesh, make_cube, read_blender_obj
from syris.util import save_image
from util import show


LOG = logging.getLogger(__name__)


def main():
    """Main function."""
    args = parse_args()
    syris.init(loglevel=logging.INFO)
    units = q.Quantity(1, args.units)
    triangles = make_cube().magnitude if args.input is None else read_blender_obj(args.input)
    triangles = triangles * units
    tr = geom.Trajectory([(0, 0, 0)] * units)
    mesh = Mesh(triangles, tr, center=args.center)
    LOG.info('Number of triangles: {}'.format(mesh.num_triangles))

    shape = (args.n, args.n)
    if args.pixel_size is None:
        if args.input is None:
            fov = 4. * units
        else:
            # Maximum sample size in x and y direction
            max_diff = np.max(mesh.extrema[:-1, 1] - mesh.extrema[:-1, 0])
            fov = max_diff
        fov *= args.margin
        args.pixel_size = fov / args.n
    else:
        fov = args.n * args.pixel_size

    if args.translate is None:
        translate = (fov.simplified.magnitude / 2., fov.simplified.magnitude / 2., 0) * q.m
    else:
        translate = (args.translate[0].simplified.magnitude,
                     args.translate[1].simplified.magnitude, 0) * q.m
    LOG.info('Translation: {}'.format(translate.rescale(q.um)))

    mesh.translate(translate)
    mesh.rotate(args.y_rotate, geom.Y_AX)
    mesh.rotate(args.x_rotate, geom.X_AX)
    fmt = 'n: {}, pixel size: {}, FOV: {}'
    LOG.info(fmt.format(args.n, args.pixel_size.rescale(q.um), fov.rescale(q.um)))

    proj = mesh.project(shape, args.pixel_size, t=None).get()
    offset = syris.gpu.util.make_vfloat3(0, translate[1].simplified, -(fov / 2.).simplified)

    if args.projection_filename is not None:
        save_image(args.projection_filename, proj)

    if args.compute_slice:
        sl = mesh.compute_slices((1,) + shape, args.pixel_size, offset=offset).get()[0]
        if args.slice_filename is not None:
            save_image(args.slice_filename, sl)
        show(sl, title='Slice at y = {}'.format(args.n / 2))

    show(proj, title='Projection')
    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--input', type=str, help='Input .obj file')
    parser.add_argument('--units', type=str, default='um', help='Mesh physical units')
    parser.add_argument('--n', type=int, default=256, help='Number of pixels')
    parser.add_argument('--pixel-size', type=float, help='Pixel size in um')
    parser.add_argument('--center', type=str, help='Mesh centering on creation')
    parser.add_argument('--translate', type=float, nargs=2, help='Translation as (x, y) in um')
    parser.add_argument('--x-rotate', type=float, default=0., help='Rotation around x axis [deg]')
    parser.add_argument('--y-rotate', type=float, default=0., help='Rotation around y axis [deg]')
    parser.add_argument('--margin', type=float, default=1., help='Margin in factor of the full FOV')
    parser.add_argument('--projection-filename', type=str, help='Save projection to this filename')
    parser.add_argument('--compute-slice', action='store_true', help='Compute also one slice')
    parser.add_argument('--slice-filename', type=str, help='Save slice to this filename')

    args = parser.parse_args()
    if args.pixel_size is not None:
        args.pixel_size = args.pixel_size * q.um
    args.x_rotate = args.x_rotate * q.deg
    args.y_rotate = args.y_rotate * q.deg
    if args.translate is not None:
        args.translate = args.translate * q.um

    return args


if __name__ == '__main__':
    main()
