"""Show different propagators."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
from numpy.fft import fftfreq
import syris
from syris.physics import energy_to_wavelength, compute_propagator
from util import show


def compute_fourier_propagator(n, lam, z, ps, fresnel=True):
    try:
        ps[0]
    except:
        ps = (ps.magnitude, ps.magnitude) * ps.units

    lam = lam.rescale(q.m).magnitude
    z = z.rescale(q.m).magnitude
    ps = ps.rescale(q.m).magnitude

    freqs = fftfreq(n)
    f = np.tile(freqs, [n, 1])
    g = np.copy(f.transpose())
    f /= ps[1]
    g /= ps[0]

    if fresnel:
        result = np.exp(1j * 2 * np.pi / lam * z) * \
            np.exp(-1j * np.pi * lam * z * (f ** 2 + g ** 2))
    else:
        result = np.exp(1j * 2 * np.pi / lam * z * np.sqrt(1 - (f * lam) ** 2 - (g * lam) ** 2))

    return result


def main():
    """main"""
    syris.init(double_precision=True)
    n = 512
    ps = 0.5 * q.um
    energy = 10 * q.keV
    lam = energy_to_wavelength(energy)
    # Compute the sampling limit for given n, ps and lam
    ca = (lam / 2 / ps).simplified.magnitude
    tga = np.tan(np.arccos(ca))
    distance = (tga * n * ps / 2).simplified
    print 'Propagation distance:', distance

    propagator = compute_propagator(n, distance, lam, ps, apply_phase_factor=True,
                                    mollified=False).get()
    full_propagator = compute_propagator(n, distance, lam, ps, fresnel=False, mollified=False).get()
    np_propagator = compute_fourier_propagator(n, lam, distance, ps)
    np_full_propagator = compute_fourier_propagator(n, lam, distance, ps, fresnel=False)
    diff = propagator - np_propagator
    full_diff = full_propagator - np_full_propagator

    show(np.fft.fftshift(propagator.real), 'Syris Fresnel Propagator (Real Part)')
    show(np.fft.fftshift(np_propagator.real), 'Numpy Fresnel propagator (Real Part)')
    show(np.fft.fftshift(diff.real), 'Fresnel Syris - Fresnel Numpy (Real Part)')

    show(np.fft.fftshift(full_propagator.real), 'Syris Full Propagator (Real Part)')
    show(np.fft.fftshift(np_full_propagator.real), 'Numpy Full propagator (Real Part)')
    show(np.fft.fftshift(full_diff.real), 'Full Syris - Full Numpy (Real Part)')
    plt.show()


if __name__ == '__main__':
    main()
