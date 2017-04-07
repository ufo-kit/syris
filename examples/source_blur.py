"""Source blur example."""
import matplotlib.pyplot as plt
import quantities as q
import syris
import syris.imageprocessing as ip
from syris.geometry import Trajectory
from syris.bodies.simple import make_sphere
from syris.devices.sources import make_topotomo
from syris.physics import propagate
from util import get_default_parser, get_material, show


def main():
    args = parse_args()
    syris.init()
    n = 1024
    d = args.propagation_distance * q.m
    shape = (n, n)
    ps = 1 * q.um
    energy = 20 * q.keV
    tr = Trajectory([(n / 2, n / 2, 0)] * ps, pixel_size=ps)
    sample = make_sphere(n, n / 30 * ps, pixel_size=ps, material=get_material('air_5_30_kev.mat'))

    bm = make_topotomo(pixel_size=ps, trajectory=tr)
    print 'Source size FWHM (height x width): {}'.format(bm.size.rescale(q.um))

    u = bm.transfer(shape, ps, energy, t=0 * q.s)
    u = sample.transfer(shape, ps, energy)
    u = propagate([sample], shape, [energy], d, ps)
    intensity = (abs(u) ** 2).real.get()
    incoh = bm.apply_blur(intensity, d, ps).get()

    region = (n / 4, n / 4, n / 2, n / 2)
    intensity = ip.crop(intensity, region).get()
    incoh = ip.crop(incoh, region).get()

    show(intensity, title='Coherent')
    show(incoh, title='Applied source blur')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--propagation-distance', type=float, default=2,
                        help='Propagation distance [m]')

    return parser.parse_args()


if __name__ == '__main__':
    main()
