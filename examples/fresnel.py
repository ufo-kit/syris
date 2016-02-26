"""Comparison of analytical and numerical Fresnel diffraction pattern. The object is a square
aperture from Introduction to Fourier Optics by J. W. Goodmann, 2nd edition.
"""
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import scipy.special
import syris
import syris.config as cfg
from syris.imageprocessing import fft_2, ifft_2, crop
from syris.physics import compute_propagator, compute_propagation_sampling, energy_to_wavelength


LOG = logging.getLogger(__name__)


def propagate_numerically(n, w, ps, d, lam):
    """Propagate square aperture numerically."""
    u_0 = np.zeros((n, n), dtype=cfg.PRECISION.np_cplx)
    wp = int(np.round(w / ps).simplified.magnitude)
    region = slice(n / 2 - wp, n / 2 + wp, 1)
    u_0[region, region] = 1 + 0j

    fp = compute_propagator(n, d, lam, ps, mollified=False)
    u_0 = fft_2(u_0)
    u_0 *= fp
    ifft_2(u_0)
    res = abs(u_0) ** 2

    return crop_to_aperture(res, w, ps)


def propagate_analytically(n, w, ps, d, lam):
    """Propagate square aperture analytically."""
    x = np.arange(-n / 2 + 0.5, n / 2 + 0.5) * ps
    a_1 = -(np.sqrt(2 / (lam * d)) * (w + x)).simplified.magnitude
    a_2 = (np.sqrt(2 / (lam * d)) * (w - x)).simplified.magnitude

    ssa_1, csa_1 = scipy.special.fresnel(a_1)
    ssa_2, csa_2 = scipy.special.fresnel(a_2)
    x_image = ((csa_2 - csa_1) ** 2 + (ssa_2 - ssa_1) ** 2)
    y_image, x_image = np.meshgrid(x_image, x_image)
    image = 0.25 * x_image * y_image

    return crop_to_aperture(image.astype(cfg.PRECISION.np_float), w, ps)


def crop_to_aperture(image, w, ps):
    """Crop *image* to 2x aperture width."""
    n = image.shape[0]
    wp = int(np.round(w / ps).simplified.magnitude)
    # Crop to 4w
    region = (n / 2 - 2 * wp, n / 2 - 2 * wp, 4 * wp, 4 * wp)

    return crop(image, region).get()


def main():
    """Main function."""
    syris.init(loglevel=logging.INFO)
    args = parse_args()
    lam = energy_to_wavelength(args.energy * q.keV).simplified
    w = args.aperture / 2 * q.um
    # Width of the aperture is 2w, make the aperture half the image size
    fov = 4 * w
    ss = args.supersampling
    d = (w ** 2 / args.fn / lam).simplified
    ns, ps = compute_propagation_sampling(lam, d, fov, fresnel=True)
    # Convolution outlier
    ns *= 2
    # Power of two for FFT
    n = int(2 ** np.ceil(np.log2(ns)))
    # Supersampling of the pixel size requires supersampling^2 more data points because we enlarge
    # the FOV by changing the diffraction angle via changing the pixel size and this enlarged FOV is
    # then sampled by supersampling-smaller pixel size
    n *= ss ** 2
    ps /= ss
    fmt = 'n sampling: {}, n: {}, ps: {}, FOV: {}, propagation distance: {}'
    LOG.info(fmt.format(ns, n, np.round(ps.rescale(q.nm), 2), fov, np.round(d.rescale(q.cm), 2)))

    res = propagate_numerically(n, w, ps, d, lam)
    res_a = propagate_analytically(n, w, ps, d, lam)

    x_data = np.linspace(-2 * w.magnitude, 2 * w.magnitude, res.shape[0])
    aperture = np.zeros(res.shape[1])
    aperture[res.shape[1] / 4:3 * res.shape[1] / 4] = 1
    plt.figure()
    plt.plot(x_data, aperture, label='Aperture')
    plt.plot(x_data, res[n / 4 / ss], label='Numerical')
    plt.plot(x_data, res_a[n / 4 / ss], label='Analytical')
    plt.title('Analytical vs. Numerical Diffraction Pattern')
    plt.xlabel(r'$\mu m$')
    plt.ylabel(r'$I$')
    plt.legend(loc='best')
    plt.grid()
    LOG.info('MSE: {}'.format(np.mean((res - res_a) ** 2)))

    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fn', type=float, default=4.0, help='Fresnel number')
    parser.add_argument('--energy', type=float, default=1.0, help='Energy [keV]')
    parser.add_argument('--aperture', type=float, default=100.0, help='Aperture width [um]')
    parser.add_argument('--supersampling', type=int, default=4, help='Supersampling')

    return parser.parse_args()


if __name__ == '__main__':
    main()
