# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""Physics on the light path."""

import logging
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import quantities.constants.quantum as cq
from pyopencl import clmath
from syris.gpu import util as g_util
from syris.imageprocessing import get_gauss_2d, fft_2, ifft_2, get_butterworth
from syris.math import fwnm_to_sigma
from syris import config as cfg
from syris.util import make_tuple


LOG = logging.getLogger(__name__)


def transfer(
    thickness,
    refractive_index,
    wavelength,
    exponent=False,
    queue=None,
    out=None,
    check=True,
    block=False,
):
    """Transfer *thickness* (can be either a numpy or pyopencl array) with *refractive_index* and
    given *wavelength*. If *exponent* is True, compute the exponent of the function without applying
    the wavenumber. Use command *queue* for computation and *out* pyopencl array. If *block* is
    True, wait for the kernel to finish. If *check* is True, the function is checked for aliasing
    artefacts. Returned *out* array is different from the input one because of the pyopencl.clmath
    behavior.
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

    if exponent or check:
        wavenumber = cfg.PRECISION.np_float(2 * np.pi / wavelength.simplified.magnitude)
        ev = cfg.OPENCL.programs["physics"].transmission_add(
            queue,
            thickness_mem.shape[::-1],
            None,
            out.data,
            thickness_mem.data,
            cfg.PRECISION.np_cplx(refractive_index),
            wavenumber,
            np.int32(1),
        )
        if check and not is_wavefield_sampling_ok(out, queue=queue):
            LOG.warning("Insufficient transmission function sampling")
        if not exponent:
            # Apply the exponent
            out = clmath.exp(out, queue=queue)
    else:
        ev = cfg.OPENCL.programs["physics"].transfer(
            queue,
            thickness_mem.shape[::-1],
            None,
            out.data,
            thickness_mem.data,
            cfg.PRECISION.np_cplx(refractive_index),
            cfg.PRECISION.np_float(wavelength.simplified.magnitude),
        )
    if block:
        ev.wait()

    return out


def compute_propagator(
    size,
    distance,
    lam,
    pixel_size,
    fresnel=True,
    region=None,
    apply_phase_factor=False,
    mollified=True,
    mollifier="gauss",
    fourier=True,
    queue=None,
    block=False,
):
    """Create a propagator with (*size*, *size*) dimensions for propagation *distance*, wavelength
    *lam* and *pixel_size*. If *fresnel* is True, use the Fresnel approximation, if it is False, use
    the full propagator (don't approximate the square root). *region* is the diameter of the the
    wavefront area which is capable of interference. If *apply_phase_factor* is True, apply the
    phase factor defined by Fresnel approximation. If *mollified* is True the aliased frequencies
    are suppressed. If *fourier* is True, compute the propagator in the frequency space, otherwise
    compute the Fourier transform of a real-space propagator, which gives better results for very
    narrow real-space propagators (edge-enhancement regime) with no or low supersampling.
    *mollifier* can be "gauss" or "butterworth". If command *queue* is specified, execute the kernel
    on it. If *block* is True, wait for the kernel to finish.
    """
    if mollifier not in ("gauss", "butterworth"):
        raise ValueError("Mollifier can be either 'gauss' or 'butterworth'")
    if size % 2:
        raise ValueError("Only even sizes are supported")
    if queue is None:
        queue = cfg.OPENCL.queue
    pixel_size = make_tuple(pixel_size)
    if mollified and mollifier == "butterworth" and pixel_size[0] != pixel_size[1]:
        raise RuntimeError("Butterworth requires identical pixel sizes")

    def check_cutoff(ps):
        # Check the sampling
        r_cutoff = compute_aliasing_limit(size, lam, ps, distance, fov=region, fourier=False)
        min_n = 4
        if r_cutoff < min_n:
            LOG.warning(
                "Propagator too narrow, propagation distance too small or pixel size too large"
            )
        f_cutoff = compute_aliasing_limit(size, lam, ps, distance, fov=region, fourier=True)
        if f_cutoff < min_n:
            LOG.warning(
                "Propagator too wide, propagation distance too large or pixel size too small"
            )

    check_cutoff(pixel_size[1])
    check_cutoff(pixel_size[0])

    out = cl_array.Array(queue, (size, size), cfg.PRECISION.np_cplx)
    if apply_phase_factor:
        phase_factor = np.exp(2 * np.pi * distance.simplified / lam.simplified * 1j)
    else:
        phase_factor = 0 + 0j

    ev = cfg.OPENCL.programs["physics"].propagator(
        queue,
        (size // 2 + 1, size // 2 + 1),
        None,
        out.data,
        cfg.PRECISION.np_float(distance.simplified),
        cfg.PRECISION.np_float(lam.simplified),
        g_util.make_vfloat2(*pixel_size[::-1].simplified),
        g_util.make_vcomplex(phase_factor),
        np.int32(fresnel),
        np.int32(fourier),
    )
    if block:
        ev.wait()

    if mollified:

        def compute_sigma_component(ps):
            cutoff = compute_aliasing_limit(size, lam, ps, distance, fourier=fourier)
            if region is not None:
                cutoff_region = compute_aliasing_limit(
                    size,
                    lam,
                    ps,
                    distance,
                    fov=region,
                    fourier=fourier
                )
                cutoff = min(cutoff_region, cutoff)

            if mollifier == "gauss":
                cutoff = fwnm_to_sigma(cutoff, n=10)

            return cutoff

        cutoff = (
            compute_sigma_component(pixel_size[0]),
            compute_sigma_component(pixel_size[1])
        )
        if mollifier == "gauss":
            mollifier = get_gauss_2d(size, cutoff, fourier=False, queue=queue, block=block)
        else:
            mollifier = get_butterworth(size, cutoff[1] // 2, order=5, queue=queue, block=block)
        out = out * mollifier

        if not fourier:
            # Energy conservation, do this numerically to make sure that after mollification it's
            # still on point
            out /= cl_array.sum(out)

    return out


def is_wavefield_sampling_ok(wavefield_exponent, queue=None, out=None):
    """Check the sampling of the *wavefield_exponent*. Use OpenCL *queue* and *out* array. Return
    True if the sampling is OK, False otherwise.
    """
    shape = wavefield_exponent.shape
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl_array.zeros(queue, shape, np.int8)
    y, x = shape[0] // 2, shape[1] // 2

    cfg.OPENCL.programs["physics"].check_transmission_function(
        queue,
        (x, y),
        None,
        wavefield_exponent.imag.data,
        out.data,
        np.int32(shape[1]),
        np.int32(shape[0]),
    )
    return not out.any()


def transfer_many(
    objects,
    shape,
    pixel_size,
    energy,
    exponent=False,
    offset=None,
    queue=None,
    out=None,
    t=None,
    check=True,
    block=False,
):
    """Compute transmission from more *objects*. If *exponent* is True, compute only the exponent,
    if it is False, evaluate the exponent. Use *shape* (y, x), *pixel_size*, *energy*, *offset* as
    (y, x), OpenCL command *queue*, *out* array, time *t*, check the sampling if *check* is True and
    wait for OpenCL kernels if *block* is True. Returned *out* array is different from the input one
    because of the pyopencl.clmath behavior.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    if out is None:
        out = cl_array.zeros(queue, shape, cfg.PRECISION.np_cplx)
    u_sample = cl_array.Array(queue, shape, cfg.PRECISION.np_cplx)

    for i, sample in enumerate(objects):
        try:
            out += sample.transfer(
                shape,
                pixel_size,
                energy,
                exponent=True,
                offset=offset,
                t=t,
                queue=queue,
                out=u_sample,
                check=False,
                block=block,
            )
        except NotImplementedError:
            LOG.debug("%s does not support real space transfer", sample)

    if check and not is_wavefield_sampling_ok(out, queue=queue):
        LOG.warning("Insufficient transmission function sampling")

    # Apply the exponent
    if not exponent:
        out = clmath.exp(out, queue=queue)

    return out


def propagate(
    samples,
    shape,
    energies,
    distance,
    pixel_size,
    region=None,
    apply_phase_factor=False,
    mollified=True,
    detector=None,
    offset=None,
    queue=None,
    out=None,
    t=None,
    check=True,
    block=False,
):
    """Propagate *samples* with *shape* as (y, x) which are
    :class:`syris.opticalelements.OpticalElement` instances at *energies* to *distance*. Use
    *pixel_size*, limit coherence to *region*, *apply_phase_factor* is as by the Fresnel
    approximation phase factor, *offset* is the sample offset. *queue* an OpenCL command queue,
    *out* a PyOpenCL Array. If *block* is True, wait for the kernels to finish. If *check* is True,
    check the transmission function sampling.
    """
    if queue is None:
        queue = cfg.OPENCL.queue
    u = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_cplx)
    intensity = cl_array.zeros(queue, shape, cfg.PRECISION.np_float)

    for energy in energies:
        u.fill(0)
        u = transfer_many(
            samples,
            shape,
            pixel_size,
            energy,
            offset=offset,
            queue=queue,
            out=u,
            t=t,
            check=check,
            block=block,
        )
        if distance != 0 * q.m:
            lam = energy_to_wavelength(energy)
            propagator = compute_propagator(
                u.shape[0],
                distance,
                lam,
                pixel_size,
                region=region,
                apply_phase_factor=apply_phase_factor,
                mollified=mollified,
                queue=queue,
                block=block,
            )
            fft_2(u, queue=queue, block=block)
            for sample in samples:
                try:
                    u *= sample.transfer_fourier(
                        shape, pixel_size, energy, t=t, queue=queue, out=None, block=block
                    )
                except NotImplementedError:
                    LOG.debug("%s does not support fourier space transfer", sample)
            u *= propagator
            ifft_2(u, queue=queue, block=block)
        if detector:
            intensity += detector.convert(abs(u) ** 2, energy)
        else:
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
    """Get the collection efficiency of the scintillator combined with a lens. The efficiency is
    given by :math:`\eta = \\frac{1 - \\sqrt{1 - \\left( \\frac{N\!A}{n} \\right)^2}}{2}`, where
    :math:`N\!A` is the numerical aperture *num_aperture* of the lens, :math:`n` is the optical
    refractive index *opt_ref_index* given by the :class:`.Scintillator`.  """
    return (1 - np.sqrt(1 - (num_aperture / opt_ref_index) ** 2)) / 2


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

    r = np.sqrt(propagation_distance ** 2 + (fov / 2) ** 2).simplified
    kx = (fov / (2 * r)).simplified.magnitude
    lam_spatial = (wavelength / (kx * pixel_size)).simplified.magnitude
    ratio = lam_spatial / 2
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


def compute_propagation_distance_limit(n, wavelength, pixel_size):
    """Compute the propagation distance which just fits the sampling theorem for *n* pixels,
    *wavelength* and *pixel_size*.
    """
    alpha = np.arccos((wavelength / (2 * pixel_size)).simplified.magnitude)

    return (np.tan(alpha) * n * pixel_size / 2).simplified
