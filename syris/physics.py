"""Physics on the light path."""

import logging
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import quantities.constants.quantum as cq
from pyfft.cl import Plan
from syris.gpu import util as g_util
from syris.imageprocessing import get_gauss_2d, fft_2, ifft_2
from syris.math import fwnm_to_sigma
from syris import config as cfg


LOG = logging.getLogger(__name__)


def transfer(thickness, refractive_index, wavelength, queue=None, out=None):
    """Transfer *thickness* (can be either a numpy or pyopencl array) with *refractive_index* and
    given *wavelength*. Use command *queue* for computation and *out* pyopencl array.
    """
    if queue is None:
        queue = cfg.OPENCL.queue

    if isinstance(thickness, cl_array.Array):
        thickness_mem = thickness
    else:
        prep = thickness.simplified.magnitude.astype(cfg.PRECISION.np_float)
        thickness_mem = cl_array.to_device(queue, prep)

    if out is None:
        out = cl_array.Array(queue, thickness_mem.shape, cfg.PRECISION.np_cplx)

    cfg.OPENCL.programs['physics'].transfer(queue,
                                            thickness_mem.shape[::-1],
                                            None,
                                            out.data,
                                            thickness_mem.data,
                                            cfg.PRECISION.np_cplx(refractive_index),
                                            cfg.PRECISION.np_float(
                                                wavelength.simplified.magnitude))

    return out


def compute_propagator(size, distance, lam, pixel_size, region=None, apply_phase_factor=False,
                       mollified=True, queue=None):
    """Create a propagator with (*size*, *size*) dimensions for propagation *distance*, wavelength
    *lam* and *pixel_size*. *region* is the diameter of the the wavefront area which is capable of
    interference. If *apply_phase_factor* is True, apply the phase factor defined by Fresnel
    approximation. If *mollified* is True the aliased frequencies are suppressed. If command *queue*
    is specified, execute the kernel on it.
    """
    if size % 2:
        raise ValueError('Only even sizes are supported')
    if queue is None:
        queue = cfg.OPENCL.queue

    # Check the sampling
    r_cutoff = compute_aliasing_limit(size, lam, pixel_size, distance, fov=region, fourier=False)
    min_n = 4
    if r_cutoff < min_n:
        LOG.error('Propagator too narrow, propagation distance too small or pixel size too large')
    f_cutoff = compute_aliasing_limit(size, lam, pixel_size, distance, fov=region, fourier=True)
    if f_cutoff < min_n:
        LOG.error('Propagator too wide, propagation distance too large or pixel size too small')

    out = cl_array.Array(queue, (size, size), cfg.PRECISION.np_cplx)
    if apply_phase_factor:
        phase_factor = np.exp(2 * np.pi * distance.simplified / lam.simplified * 1j)
    else:
        phase_factor = 0 + 0j

    cfg.OPENCL.programs['physics'].propagator(queue,
                                              (size / 2 + 1, size / 2 + 1),
                                              None,
                                              out.data,
                                              cfg.PRECISION.np_float(distance.simplified),
                                              cfg.PRECISION.np_float(lam.simplified),
                                              cfg.PRECISION.np_float(pixel_size.simplified),
                                              g_util.make_vcomplex(phase_factor))

    if mollified:
        fwtm = compute_aliasing_limit(size, lam, pixel_size, distance,
                                      fov=size * pixel_size, fourier=True)
        if region is not None:
            fwtm_region = compute_aliasing_limit(size, lam, pixel_size, distance, region,
                                                 fourier=True)
            fwtm = min(fwtm_region, fwtm)

        sigma = fwnm_to_sigma(fwtm, n=10)
        mollifier = get_gauss_2d(size, sigma, fourier=False, queue=queue)
        out = out * mollifier

    return out


def propagate(samples, energies, distance, pixel_size, region=None, apply_phase_factor=False,
              mollified=True, queue=None, out=None, plan=None):
    """Propagate *samples* which are :class:`syris.opticalelements.OpticalElement`
    instances at *energies* to *distance*. Use *pixel_size*, limit coherence to *region*,
    *apply_phase_factor* is as by the Fresnel approximation phase factor, *queue* an OpenCL command
    queue, *out* a PyOpenCL Array and *plan* and FFT plan.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    shape = samples[0].project().shape
    if plan is None:
        plan = Plan(shape, queue=queue)
    u = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_cplx)
    intensity = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
    intensity.fill(0)

    for energy in energies:
        u.fill(1)
        lam = energy_to_wavelength(energy)
        propagator = compute_propagator(u.shape[0], distance, lam, pixel_size, region=region,
                                        apply_phase_factor=apply_phase_factor,
                                        mollified=mollified, queue=queue)
        for sample in samples:
            u *= sample.transfer(energy, queue=queue, out=out)
        fft_2(u.data, plan, wait_for_finish=True)
        u *= propagator
        ifft_2(u.data, plan, wait_for_finish=True)
        intensity += abs(u) ** 2

    return intensity


def energy_to_wavelength(energy):
    """Convert *energy* [eV-like] to wavelength [m]."""
    res = cq.h * q.velocity.c / energy

    return res.rescale(q.m)


def wavelength_to_energy(wavelength):
    """Convert wavelength [m-like] to energy [eV]."""
    res = cq.h * q.velocity.c / wavelength

    return res.rescale(q.eV)


def ref_index_to_attenuation_coeff(ref_index, lam):
    """Convert refractive index to the linear attenuation coefficient
    given by :math:`\\mu = \\frac{4 \\pi \\beta}{\\lambda}` based on given
    *ref_index* and wavelength *lam*.
    """
    return 4 * np.pi * ref_index.imag / lam.simplified


def compute_collection(num_aperture, opt_ref_index):
    """Get the collection efficiency of the scintillator combined with
    a lens. The efficiency is given by :math:`\eta = \\frac{1}{2}
    \\left( \\frac{N\!A}{n} \\right)^2`, where :math:`N\!A` is the numerical
    aperture *num_aperture* of the lens, :math:`n` is the optical refractive
    index *opt_ref_index* given by the :class:`.Scintillator`.
    """
    return 0.5 * (num_aperture / opt_ref_index) ** 2


def visible_light_attenuation_coeff(scintillator, camera, num_aperture,
                                    lens_trans_eff):
    """Get the visible light attenuation coefficient given by the optical
    system composed of a *scintillator*, objective lens with transmission
    efficiency *lens_trans_eff* and a *camera*. If we assume the lens
    transmission efficiency :math:`l_{eff}` and optics collection efficiency
    :math:`\eta` to be constant for all visible light wavelengths, define
    :math:`Q_{scint}\\left( \lambda_{vis}\\right)` to be the quantum
    efficiency of the scintillator for a visible light wavelength
    :math:`\lambda_{vis}`, :math:`Q_{cam} \\left( \lambda_{vis}\\right)`
    to be the quantum efficiency of the camera, the attenuation coefficient
    is given by

    .. math::
        l_{eff} \cdot \eta \int Q_{scint}\\left( \lambda_{vis}\\right)
        Q_{cam}\\left( \lambda_{vis}\\right) \, \mathrm{d}\lambda_{vis}.
    """
    quantum_eff = np.sum(scintillator.quantum_effs * camera.quantum_effs)

    return lens_trans_eff * quantum_eff * compute_collection(num_aperture,
                                                             scintillator.opt_ref_index)


def compute_diffraction_angle(diameter, propagation_distance):
    """Compute the diffraction angle for a region where a wavefield within the *diameter* can
    interfere on a *propagation_distance*.
    """
    distance = propagation_distance.simplified.magnitude
    diameter = diameter.simplified.magnitude

    return np.arctan(diameter / (2 * distance))


def compute_aliasing_limit(n, wavelength, pixel_size, propagation_distance, fov=None, fourier=True):
    """Get the non-aliased fraction of data points when propagating a wavefield to a region :math:`n
    \\times pixel\\_size` to *propagation_distance* using *wavelength*, *pixel_size* and field of
    view *fov* (if not specified computed as *n* * *pixel_size*). If *fourier* is True then the
    limit is computed for the Fourier space.
    """
    if fov is None:
        fov = n * pixel_size
    cos_al_max = wavelength.simplified.magnitude / (2 * pixel_size.simplified.magnitude)
    diffraction_angle = compute_diffraction_angle(fov, propagation_distance)
    ratio = cos_al_max / np.cos(np.pi / 2 - diffraction_angle)
    if fourier:
        ratio = 1 / ratio

    return int(np.floor(min(n, max(1, n * ratio))))


def compute_propagation_sampling(wavelength, distance, fov, fresnel=True):
    """Compute the required number of pixels and pixel size in order to satisfy the sampling theorem
    when propagating a wavefield with *wavelength* to *distance* and we want to propagate field of
    view *fov*. If *fresnel* is true, the same distance computation approximation is done as when
    computing a Fresnel propagator (2nd order Taylor series expansion for the square root).
    """
    if fresnel:
        r = distance + (fov / 2) ** 2 / (2 * distance)
    else:
        r = np.sqrt((fov / 2) ** 2 + distance ** 2)
    # Nyquist f_max = 1 / 2 pixels
    # cos_alpha = lam / (2 * ps) = fov / (2 * r)
    ps = (wavelength * r / fov).rescale(q.um)
    n = int(np.ceil((fov / ps).simplified.magnitude))

    return n, ps
