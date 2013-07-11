"""Physics on the light path."""

import numpy as np
import quantities as q
import quantities.constants.quantum as cq
from syris.gpu import util as g_util

CL_PRG = g_util.get_program(g_util.get_source(["vcomplex.cl", "physics.cl"]))


def propagator(size, distance, lam, copy_to_host=False, queue=None):
    """Create a propagator with (*size*, *size*) dimensions for propagation
    *distance*, wavelength *lam* and if *copy_to_host* is True, copy the
    propagator to host. If command *queue* is specified, execute the kernel
    on it.
    """
    pass


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
