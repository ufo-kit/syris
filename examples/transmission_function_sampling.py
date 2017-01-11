"""Aliasing of the transmission function of a wedge which projection is computed as f(x, y) = x.
Delta is chosen in such a way that it causes phase shift between two adjacent pixels 2Pi in case of
no supersampling. Thus, The transmission function of the wedge along x has phase 0, 2Pi, 4Pi, ...,
which is lost due to insufficient pixel spacing in case of no supersampling.

The used material is a pure phase material, i.e. beta = 0. The results are the real part of the T(x,
y), which is cos(-2 Pi / lambda x delta).
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.bodies.simple import StaticBody
from syris.materials import Material
from syris.physics import energy_to_wavelength
from util import get_default_parser, show


def compute_transmission_function(n, ps, supersampling, energy, material):
    n *= supersampling
    shape = (n, n)
    ps /= supersampling
    wedge = np.tile(np.arange(n), [n, 1]) * ps
    wedge = StaticBody(wedge, ps, material=material)

    return wedge.transfer(shape, ps, energy).get()


def main():
    args = parse_args()
    syris.init()
    n = 32
    ps = 1 * q.um
    energies = np.arange(5, 30) * q.keV
    energy = 10 * q.keV
    lam = energy_to_wavelength(energy)
    # Delta causes phase shift between two adjacent pixels by 2 Pi
    delta = (lam / ps).simplified.magnitude
    ri = np.ones_like(energies.magnitude, dtype=np.complex) * delta + 0j
    material = Material('dummy', ri, energies)
    fmt = 'Computing with n: {:>4}, pixel size: {}'

    wedge = np.tile(np.arange(n), [n, 1]) * ps
    # Non-supersampled object shape causes phase shifts 0, 2Pi, 4Pi, ..., thus the phase is constant
    # as an extreme result of aliasing
    print fmt.format(n, ps)
    u = compute_transmission_function(n, ps, 1, energy, material)
    # Supersampling helps resolve the transmission function
    print fmt.format(n * args.supersampling, ps / args.supersampling)
    u_s = compute_transmission_function(n, ps, args.supersampling, energy, material)

    show(wedge.magnitude, title='Projected Object [um]')
    show(u.real, title='Re[T(x, y)]')
    show(u_s.real, title='Re[T(x, y)] Supersampled')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--supersampling', type=int, choices=[2, 4, 8, 16, 32], default=32,
                        help='Supersampling used to prevent transmission function aliasing')

    return parser.parse_args()


if __name__ == '__main__':
    main()
