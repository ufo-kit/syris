"""Example showing variable convolution."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.config as cfg
import syris.gpu.util as gutil
import syris.imageprocessing as ip
from syris.bodies.simple import make_grid
from syris.util import save_image
from util import get_default_parser, show


def main():
    args = parse_args()
    syris.init(device_index=0)
    m = 20

    if args.input == 'grid':
        image = make_grid(args.n, m * q.m).thickness.get()
    elif args.input == 'lena':
        from scipy.misc import lena
        image = lena().astype(cfg.PRECISION.np_float)
        if args.n != image.shape[0]:
            image = gutil.get_host(ip.rescale(image, (args.n, args.n)))

    n = image.shape[0]
    crop_n = n - 2 * m - 2
    y, x = np.mgrid[-n / 2:n / 2, -n / 2:n / 2]
    # Compute a such that the disk diameter is exactly the period when distance from the middle is n
    # / 2
    a = m / (2 * (crop_n / 2.) ** 2)
    radii = (a * np.sqrt(x ** 2 + y ** 2) ** 2 + 1e-3).astype(cfg.PRECISION.np_float)
    x_param = radii
    y_param = radii

    result = ip.varconvolve_disk(image, (y_param, x_param)).get()
    result = ip.crop(result, (m - 1, m - 1, crop_n, crop_n)).get()
    radii = ip.crop(radii, (m - 1, m - 1, crop_n, crop_n)).get()
    image = ip.crop(image, (m - 1, m - 1, crop_n, crop_n)).get()

    if args.output:
        save_image(args.output, result)

    show(image, title='Original Image')
    show(2 * radii, title='Blurring Disk Diameters')
    show(result, title='Blurred Image')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)

    parser.add_argument('--input', default='grid', choices=['grid', 'lena'],
                        help='Input image')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument('--n', type=int, default=512, help='Number of pixels in one dimension')

    return parser.parse_args()


if __name__ == '__main__':
    main()
