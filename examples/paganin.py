"""Show forward phase contrast simulation and backward phase retrieval using the Paganin method[1].

[1] Paganin, David, et al. "Simultaneous phase and amplitude extraction from a single defocused
image of a homogeneous object." Journal of microscopy 206.1 (2002): 33-40.
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.config as cfg
from numpy.fft import fftfreq
from syris.bodies.simple import make_sphere
from syris.gpu.util import get_array
from syris.imageprocessing import decimate, fft_2, ifft_2
from syris.physics import propagate, compute_propagator, energy_to_wavelength
from util import get_default_parser, get_material, show


def get_propagator_psf(n, d, ps, energy):
    lam = energy_to_wavelength(energy)
    propagator = compute_propagator(n, d, lam, ps).get()

    return np.fft.fftshift(np.fft.ifft2(propagator))


def compute_tie_kernel(n, pixel_size, distance, material, energy):
    pixel_size = pixel_size.rescale(q.m).magnitude
    distance = distance.rescale(q.m).magnitude
    f_0 = fftfreq(n) / pixel_size
    f = f_0 * 2 * np.pi
    f_0, g_0 = np.meshgrid(f_0, f_0)
    f, g = np.meshgrid(f, f)
    ri = material.get_refractive_index(energy)
    delta = ri.real
    beta = ri.imag
    mju = material.get_attenuation_coefficient(energy).rescale(1 / q.m).magnitude
    fmt = '                            mju: {}'
    print fmt.format(mju)
    fmt = '                          delta: {}'
    print fmt.format(delta)
    fmt = '                           beta: {}'
    print fmt.format(beta)

    return mju / (distance * ri.real * (f ** 2 + g ** 2) + mju)
    # Alternative forms
    # lam = energy_to_wavelength(energy).rescale(q.m).magnitude
    # tmp = 4 * np.pi * lam * beta
    # alpha = beta / delta
    # return alpha / (np.pi * lam * distance * (f_0 ** 2 + g_0 ** 2) + alpha)
    # return tmp / (lam ** 2 * distance * delta * (f ** 2 + g ** 2) + tmp)


def main():
    args = parse_args()
    syris.init()
    # Propagate to 20 cm
    d = 5 * q.cm
    # Compute grid
    n_camera = 256
    n = n_camera * args.supersampling
    shape = (n, n)
    material = get_material('pmma_5_30_kev.mat')
    energy = 15 * q.keV
    ps = 1 * q.um
    ps_hd = ps / args.supersampling
    radius = n / 4. * ps_hd

    fmt = '                     Wavelength: {}'
    print fmt.format(energy_to_wavelength(energy))
    fmt = 'Pixel size used for propagation: {}'
    print fmt.format(ps_hd.rescale(q.um))
    print '                  Field of view: {}'.format(n * ps_hd.rescale(q.um))
    fmt = '                Sphere diameter: {}'
    print fmt.format(2 * radius)

    sample = make_sphere(n, radius, pixel_size=ps_hd, material=material)
    projection = sample.project((n, n), ps_hd).get() * 1e6
    projection = decimate(projection, (n_camera, n_camera), average=True).get()
    # Propagation with a monochromatic plane incident wave
    hd = propagate([sample], shape, [energy], d, ps_hd).get()
    ld = decimate(hd, (n_camera, n_camera), average=True).get()

    kernel = compute_tie_kernel(n_camera, ps, d, material, energy)
    mju = material.get_attenuation_coefficient(energy).rescale(1 / q.m).magnitude
    f_ld = fft_2(ld)
    f_ld *= get_array(kernel.astype(cfg.PRECISION.np_float))
    retrieved = ifft_2(f_ld).get().real
    retrieved = - 1 / mju * np.log(retrieved) * 1e6

    show(hd, title='High resolution')
    show(ld, title='Low resolution (decector)')
    show(retrieved, title='Retrieved [um]')
    show(projection, title='Projection [um]')
    show(projection - retrieved, title='Projection - retrieved')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--supersampling', type=int, default=8,
                        help='Supersampling used to prevent propagation artefacts')

    return parser.parse_args()


if __name__ == '__main__':
    main()

