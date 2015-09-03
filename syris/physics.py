"""Physics on the light path."""

import numpy as np
import pyopencl as cl
import quantities as q
import quantities.constants.quantum as cq
from pyopencl.array import Array
from syris.gpu import util as g_util
from syris.imageprocessing import get_gauss_2d
from syris.math import fwnm_to_sigma
from syris.util import make_tuple
from syris import config as cfg


def transfer(thickness, refractive_index, wavelength, shape=None, queue=None, out_memory=None):
    """Transfer *thickness* (can be either a numpy array or OpenCL Buffer)
    with *refractive_index* and given *wavelength*. *shape* is the image shape
    as (width, height) in case *thickness* is an OpenCL Buffer. *ctx* is
    OpenCL context and queue a CommandQueue.
    """
    if queue is None:
        queue = cfg.OPENCL.queue

    if isinstance(thickness, cl.Buffer):
        thickness_mem = thickness
    else:
        thickness_mem = cl.Buffer(queue.context, cl.mem_flags.READ_ONLY |
                                  cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=thickness.simplified.magnitude.astype(
                                  cfg.PRECISION.np_float))
        shape = thickness.shape

    if out_memory is None:
        out = Array(queue, shape, cfg.PRECISION.np_cplx)
        out_memory = out.data

    cfg.OPENCL.programs['physics'].transfer(queue,
                                            shape[::-1],
                                            None,
                                            out_memory,
                                            thickness_mem,
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

    out = Array(queue, (size, size), cfg.PRECISION.np_cplx)
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
        fwtm = compute_aliasing_limit(size, lam, pixel_size, size * pixel_size, distance)
        if region is not None:
            fwtm_region = compute_aliasing_limit(size, lam, pixel_size, region, distance)
            fwtm = min(fwtm_region, fwtm)

        sigma = fwnm_to_sigma(fwtm * size * pixel_size, n=10)
        mollifier = get_gauss_2d(size, sigma, pixel_size, fourier=False, queue=queue)
        out = out * mollifier

    return out


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


def compute_aliasing_limit(n, lam, pixel_size, diameter, propagation_distance):
    """Get the non-aliased fraction of data points when propagating a wavefield to a region :math:`n
    \\times pixel\\_size` to *propagation_distance* using wavelength *lam* and *pixel_size*.
    """
    cos_al_max = lam.simplified.magnitude / (2 * pixel_size.simplified.magnitude)
    diffraction_angle = compute_diffraction_angle(diameter, propagation_distance)

    return min(1, np.cos(np.pi / 2 - diffraction_angle) / cos_al_max)
