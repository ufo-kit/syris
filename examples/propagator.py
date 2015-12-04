"""Show light propagator."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
from numpy.fft import fftfreq
import syris
from syris.physics import energy_to_wavelength, compute_propagator


def get_fresnel_kernel_f(n, lam, z, ps):
    """Get fresnel kernel in the Fourier space with image size *n*, pixel size
    *ps*, wavelength *lam*."""

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

    return np.exp(-1j * np.pi * lam * z * (f ** 2 + g ** 2))


def show(data, title):
    """Show *data* with *title*."""
    plt.figure()
    plt.imshow(data)
    plt.title(title)
    plt.colorbar()


def main():
    """main"""
    syris.init()
    n = 2048
    ps = 1 * q.um
    energy = 10 * q.keV
    lam = energy_to_wavelength(energy)
    distance = 100 * q.cm

    propagator = compute_propagator(n, distance, lam, ps, mollified=False).get()
    np_propagator = get_fresnel_kernel_f(n, lam, distance, ps)
    diff = propagator - np_propagator

    print diff.real.min(), diff.real.max()
    print diff.imag.min(), diff.imag.max()

    show(np.fft.fftshift(propagator.real), 'Syris Propagator')
    show(np.fft.fftshift(np_propagator.real), 'Numpy propagator')
    show(diff.real, 'Difference Real Part')
    show(diff.imag, 'Difference Imaginary Part')
    plt.show()


if __name__ == '__main__':
    main()
