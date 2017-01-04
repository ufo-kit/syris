"""Edge enhancement caused by free-space propagation. Control the accuracy by the --supersampling
option. If this value is too low (1), the propagators are not resolved correctly and the resulting
images contain artefacts. Increase this value to e.g. 4 and you will see how the propagators and the
results change.
"""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.geometry import Trajectory
from syris.bodies.simple import make_sphere
from syris.devices.cameras import Camera
from syris.devices.detectors import Detector
from syris.devices.filters import Scintillator
from syris.devices.lenses import Lens
from syris.devices.sources import make_topotomo
from syris.math import fwnm_to_sigma
from syris.physics import propagate, compute_propagator, energy_to_wavelength
from util import get_default_parser, get_material, show


def get_propagator_psf(n, d, ps, energy):
    lam = energy_to_wavelength(energy)
    propagator = compute_propagator(n, d, lam, ps).get()

    return np.fft.fftshift(np.fft.ifft2(propagator))


def main():
    args = parse_args()
    syris.init()
    # Propagate to 20 cm
    d = 20 * q.cm
    # Compute grid
    n_camera = 256
    n = n_camera * args.supersampling
    shape = (n, n)
    material = get_material('pmma_5_30_kev.mat')
    energies = material.energies
    dE = energies[1] - energies[0]
    # Lens with magnification 5 and numerical aperture 0.25
    lens = Lens(5, na=0.25)
    # Considered visible light wavelengths
    vis_wavelengths = np.arange(500, 700) * q.nm
    # Simple camera quantum efficiencies
    cam_qe = 0.1 * np.ones(len(vis_wavelengths))
    camera = Camera(10 * q.um, 0.1, 500, 23, 32, (256, 256), exp_time=args.exposure * q.ms,
                    fps=1 / q.s, quantum_efficiencies=cam_qe, wavelengths=vis_wavelengths,
                    dtype=np.float32)
    # Scintillator emits visible light into a region given by a Gaussian distribution
    x = camera.wavelengths.rescale(q.nm).magnitude
    sigma = fwnm_to_sigma(50)
    emission = np.exp(-(x - 600) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    # Scintillator 50 um thick, light yield 14 and refractive index 1.84
    luag = get_material('luag.mat')
    scintillator = Scintillator(50 * q.um,
                                luag,
                                14 * np.ones(len(energies)) / q.keV,
                                energies,
                                emission / q.nm,
                                camera.wavelengths,
                                1.84)
    detector = Detector(scintillator, lens, camera)
    # Pixel size used for propagation
    ps = detector.pixel_size / args.supersampling

    fmt = 'Pixel size used for propagation: {}'
    print fmt.format(ps.rescale(q.um))
    fmt = '  Effective detector pixel size: {}'
    print fmt.format(detector.pixel_size.rescale(q.um))
    print '                  Field of view: {}'.format(n * ps.rescale(q.um))

    # Bending magnet source
    trajectory = Trajectory([(n / 2, n / 2, 0)] * ps)
    source = make_topotomo(dE=dE, trajectory=trajectory, pixel_size=ps)

    sample = make_sphere(n, n / 4. * ps, pixel_size=ps, material=material)
    # Propagation with a monochromatic plane incident wave
    coherent = propagate([source, sample], shape, [15 * q.keV], d, ps, t=0 * q.s,
                         detector=detector).get()
    coherent *= camera.exp_time.simplified.magnitude
    # Decimate to fit the effective pixel size of the detector system
    coherent_ld = camera.get_image(coherent, shot_noise=False, amplifier_noise=False)

    # Propagation which takes into account polychromaticity
    poly = propagate([source, sample], shape, range(10, 30) * q.keV, d, ps, t=0 * q.s,
                     detector=detector).get()
    poly *= camera.exp_time.simplified.magnitude
    poly_ld = camera.get_image(poly, shot_noise=args.noise, amplifier_noise=args.noise)

    # Compute and show some of the used propagators
    propagator_10 = get_propagator_psf(n, d, ps, 10 * q.keV)
    propagator_30 = get_propagator_psf(n, d, ps, 30 * q.keV)

    show(coherent, title='Coherent Supersampled')
    show(coherent_ld, title='Coherent Detector')
    show(propagator_10.real, title='Propagator PSF for 10 keV (real part)')
    show(propagator_30.real, title='Propagator PSF for 30 keV (real part)')
    show(poly, title='Polychromatic Supersampled')
    show(poly_ld, title='Polychromatic Detector')
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--supersampling', type=int, default=8,
                        help='Supersampling used to prevent propagation artefacts')
    parser.add_argument('--noise', action='store_true',
                        help='Apply noise on the polychromatic image')
    parser.add_argument('--exposure', type=float, default=1,
                        help='Camera exposure time [ms], defines the amount of ' +
                        'noise in the polychromatic image')

    return parser.parse_args()


if __name__ == '__main__':
    main()
