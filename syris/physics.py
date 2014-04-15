"""Physics on the light path."""

import numpy as np
import pyopencl as cl
import quantities as q
import quantities.constants.quantum as cq
from syris.gpu import util as g_util
from syris import config as cfg


def transfer(thickness, refractive_index, wavelength, shape=None, ctx=None,
             queue=None, out_memory=None):
    """Transfer *thickness* (can be either a numpy array or OpenCL Buffer)
    with *refractive_index* and given *wavelength*. *shape* is the image shape
    as (width, height) in case *thickness* is an OpenCL Buffer. *ctx* is
    OpenCL context and queue a CommandQueue.
    """
    if ctx is None:
        ctx = cfg.OPENCL.ctx
    if queue is None:
        queue = cfg.OPENCL.queue

    if isinstance(thickness, cl.Buffer):
        thickness_mem = thickness
    else:
        thickness_mem = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                  hostbuf=thickness.simplified.magnitude.astype(
                                          cfg.PRECISION.np_float))
        shape = thickness.shape[::-1]

    if out_memory is None:
        out_memory = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                               size=shape[0] * shape[1] * cfg.PRECISION.cl_cplx)

    cfg.OPENCL.programs['physics'].transfer(queue,
                                            shape,
                                            None,
                                            out_memory,
                                            thickness_mem,
                                            cfg.PRECISION.np_cplx(refractive_index),
                                            cfg.PRECISION.np_float(
                                                wavelength.simplified.magnitude))

    return out_memory



def compute_propagator(size, distance, lam, pixel_size, apply_phase_factor=False,
                   copy_to_host=False, ctx=None, queue=None):
    """Create a propagator with (*size*, *size*) dimensions for propagation
    *distance*, wavelength *lam*, *pixel_size* and if *apply_phase_factor*
    is True, apply the phase factor defined by Fresne approximation. If
    *copy_to_host* is True, copy the propagator to host. If command *queue*
    is specified, execute the kernel on it. *ctx* is OpenCL context and
    *queue* is a CommandQueue.
    """
    if ctx is None:
        ctx = cfg.OPENCL.ctx
    if queue is None:
        queue = cfg.OPENCL.queue

    mem = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=size ** 2 * cfg.PRECISION.cl_cplx)
    if apply_phase_factor:
        phase_factor = np.exp(2 * np.pi * distance.simplified / lam.simplified * 1j)
    else:
        phase_factor = 0 + 0j

    cfg.OPENCL.programs['physics'].propagator(queue,
                                 (size, size),
                                  None,
                                  mem,
                                  cfg.PRECISION.np_float(distance.simplified),
                                  cfg.PRECISION.np_float(lam.simplified),
                                  cfg.PRECISION.np_float(pixel_size.simplified),
                                  g_util.make_vcomplex(phase_factor))

    if copy_to_host:
        res = np.empty((size, size), dtype=cfg.PRECISION.np_cplx)
        cl.enqueue_copy(queue, res, mem)
    else:
        res = mem

    return res


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
    if lam.simplified.units == q.m:
        lam = lam.simplified
    else:
        lam = energy_to_wavelength(lam)

    return 4 * np.pi * ref_index.imag / lam.simplified


def optics_collection_eff(num_aperture, opt_ref_index):
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

    return lens_trans_eff * quantum_eff * \
        optics_collection_eff(num_aperture, scintillator.opt_ref_index)
