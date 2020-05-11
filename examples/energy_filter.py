"""Energy filter based on Gaussian profile."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.math as smath
from syris.util import get_gauss
from syris.geometry import Trajectory
from syris.devices.sources import make_topotomo
from syris.devices.filters import GaussianFilter
from syris.physics import propagate
from .util import get_default_parser, show


def get_spectrum(source, energies, pixel_size):
    return np.array([source.get_flux(e, 0 * q.rad, pixel_size) for e in energies])


def main():
    args = parse_args()
    syris.init()
    n = 1024
    shape = (n, n)
    ps = 1 * q.um
    tr = Trajectory([(n / 2, n / 2, 0)] * ps, pixel_size=ps)
    energy_center = args.energy_center * q.keV
    fwhm = args.energy_resolution * energy_center
    sigma = smath.fwnm_to_sigma(fwhm, n=2)
    # Make sure we resolve the curve nicely
    energies = np.arange(max(1 * q.keV, energy_center - 2 * fwhm),
                         energy_center + 2 * fwhm,
                         fwhm / 25) * q.keV
    dE = energies[1] - energies[0]
    print('Energy from, to, step, number:', energies[0], energies[-1], dE, len(energies))

    bm = make_topotomo(dE=dE, pixel_size=ps, trajectory=tr)
    spectrum_energies = np.arange(1, 50, 1) * q.keV
    native_spectrum = get_spectrum(bm, spectrum_energies, ps)

    fltr = GaussianFilter(energies, energy_center, sigma)
    gauss = get_gauss(energies.magnitude, energy_center.magnitude, sigma.magnitude)
    filtered_spectrum = get_spectrum(bm, energies, ps) * gauss

    intensity = propagate([bm, fltr], shape, energies, 0 * q.m, ps).get()

    show(intensity, title='Intensity for energy range {} - {}'.format(energies[0], energies[-1]))

    plt.figure()
    plt.plot(spectrum_energies.magnitude, native_spectrum)
    plt.title('Source Spectrum')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Intensity')

    plt.figure()
    plt.plot(energies.magnitude, gauss)
    plt.title('Gaussian Filter')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Transmitted intensity')

    plt.figure()
    plt.plot(energies.magnitude, filtered_spectrum)
    plt.title('Filtered Spectrum')
    plt.xlabel('Energy [keV]')
    plt.ylabel('Intensity')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--energy-center', type=float, default=10,
                        help='Energy at which the filter is centered [keV]')
    parser.add_argument('--energy-resolution', type=float, default=1e-2,
                        help='Energy resolution (FWHM of the Gaussian filter) [dE/E]')

    return parser.parse_args()


if __name__ == '__main__':
    main()

