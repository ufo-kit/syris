"""
X-ray sources at synchrotrons. They provide X-ray photons used for imaging.
Synchrotron radiation sources provide high photon flux of photons with
different energies, which form a spectrum characteristic for a given source
type.
"""
import numpy as np
import quantities as q
import quantities.constants.electron as qe
import pyopencl.array as cl_array
from quantities.quantity import Quantity
from quantities.constants import fine_structure_constant
from scipy import integrate, special
import syris.config as cfg
from syris.geometry import Trajectory
from syris.opticalelements import OpticalElement
from syris.util import make_tuple


class BendingMagnet(OpticalElement):

    """Bending magnet X-ray source."""

    _SR_CONST = 3 * fine_structure_constant.simplified / (4 * np.pi ** 2)

    def __init__(self, electron_energy, el_current, magnetic_field, sample_distance, dE, size,
                 pixel_size, trajectory, profile_approx=True):
        """The parameters are *electron_energy*, electric *el_current*, *magnetic_field*, place it
        into *sample_distance* (distance between the source and a sample), take into account energy
        spacing *dE* which sets the amount of photons obtained for an energy to be:

        .. math::

            \Phi = \int_{E - dE / 2}^{E + dE / 2} \Phi(E) dE

        Set its *size* (y, x) specified as FWHM and approximate it by a Gaussian. *pixel_size* is
        the effective pixel size. *trajectory* is the trajectory defining the source position in
        space and time. If *profile_approx* is True, the profile at a given vertical observation
        angle will not be integrated over the relevant energies but will be calculated for the mean
        energy and multiplied by *dE*.
        """

        super(BendingMagnet, self).__init__()
        self.electron_energy = electron_energy.simplified
        self.el_current = el_current.simplified
        self.magnetic_field = magnetic_field
        self.sample_distance = sample_distance.simplified
        self.dE = dE
        self.size = size.simplified
        self.pixel_size = make_tuple(pixel_size, num_dims=2)
        self.trajectory = trajectory
        self.profile_approx = profile_approx

    @property
    def gama(self):
        """:math:`\\frac{E}{m_ec^2}`"""
        return self.electron_energy / (qe.electron_mass * q.c ** 2)

    @property
    def critical_energy(self):
        """Critical energy of the source is defined as
            .. math::

                \epsilon_c [keV] = 0.665 E^2 [GeV] B[T]
        """
        return 0.665 * self.electron_energy.rescale(q.GeV).magnitude ** 2 * \
            self.magnetic_field.rescale(q.T).magnitude * q.keV

    def _get_full_profile(self, energy, angles, pixel_size):
        """Get the vertical profile based on energies integration.  If there are two energies e_0
        and e_1, the profile at a specific angle will be calculated by integrating the energy range
        from e_0 - d_e, e_0 + d_e, where d_e = e_1 - e_0. *angles* are the vertical angles for which
        the flux is computed, *pixel_size* is the vertical pixel size.
        """
        def _get_flux_wrapper(photon_energy, vertical_angle):
            """Get rid of quantities, because scipy.romberg
            cannot work with them.
            """
            flux = self.get_flux(photon_energy * q.keV, vertical_angle * q.rad,
                                 pixel_size).magnitude
            # Conversion is 1e3 * dE / E, 1e3 for the 0.1 % BW. Integration takes care of the dE so
            # we need to correct by 1e3 / E
            return flux * 1e3 / photon_energy

        def _get_flux_at_angle(angle, energy, d_energy):
            e_0 = energy - d_energy / 2.0
            e_1 = energy + d_energy / 2.0
            return integrate.romberg(_get_flux_wrapper, e_0, e_1, args=(angle,))

        get_profiles = np.vectorize(_get_flux_at_angle)

        energy = energy.rescale(q.keV).magnitude
        d_energy = self.dE.rescale(q.keV).magnitude

        return get_profiles(angles.rescale(q.rad).magnitude, energy, d_energy) / q.s

    def get_next_time(self, t_0, distance):
        """Get the next time when the source will have moved more than *distance*."""
        return self.trajectory.get_next_time(t_0)

    def _transfer(self, shape, pixel_size, energy, offset, t=0 * q.s, queue=None, out=None,
                  block=False):
        """Compute the flat field wavefield."""
        if queue is None:
            queue = cfg.OPENCL.queue

        ps = make_tuple(pixel_size)
        y = self.trajectory.get_point(t)[1]
        fov = np.arange(0, shape[0]) * ps[0] - y + offset[0]
        angles = np.arctan((fov / self.sample_distance).simplified)
        profile = self._create_vertical_profile(energy, angles, ps[0]).rescale(1 / q.s).magnitude

        profile = cl_array.to_device(queue, profile.astype(cfg.PRECISION.np_float))
        if out is None:
            out = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_cplx)

        ev = cfg.OPENCL.programs['physics'].make_flat(queue,
                                                      shape[::-1],
                                                      None,
                                                      out.data,
                                                      profile.data)
        if block:
            ev.wait()

        return out

    def _create_vertical_profile(self, energy, angles, pixel_size):
        if self.profile_approx:
            # Much faster but less precise.
            # dE / E = 1e-3 = 0.1 % BW, we need to convert it to the actual bandwidth of the
            # energies we use, so the result is 1e3 * dE / E
            bw_conv = 1e3 * (self.dE / energy).simplified.magnitude
            result = self.get_flux(energy, angles, pixel_size) * bw_conv
        else:
            # Full energy integration.
            result = self._get_full_profile(energy, angles, pixel_size)

        return result

    def get_flux(self, photon_energy, vertical_angle, pixel_size):
        """Get the photon flux coming from the source consisting of photons
        with *photon_energy* and get it at the vertical observation angle
        *vertical_angle*.
        """
        gama = Quantity(self.electron_energy / (qe.electron_mass * q.c ** 2)).simplified
        gama_psi = gama * vertical_angle.rescale(q.rad)
        norm_energy = photon_energy.rescale(self.critical_energy.units) / self.critical_energy
        xi = Quantity(0.5 * norm_energy.magnitude * (1.0 + gama_psi ** 2) ** (3.0 / 2)).magnitude
        angle_step = np.arctan(pixel_size.simplified / self.sample_distance.simplified)

        # 1e-3 for 0.1 % BW
        return Quantity(BendingMagnet._SR_CONST * gama ** 2 *
                        self.el_current / q.elementary_charge *
                        norm_energy ** 2 *
                        (1.0 + gama_psi ** 2) ** 2 *
                        (special.kv(2.0 / 3, xi) ** 2 + gama_psi ** 2 /
                         (1.0 + gama_psi ** 2) * special.kv(1.0 / 3, xi) ** 2) *
                        angle_step.rescale(q.rad) ** 2 * 1e-3).simplified


def make_topotomo(dE=None, trajectory=None, pixel_size=None, ring_current=200 * q.mA):
    """Make the TopoTomo bending magnet source located at ANKA, KIT. Use *dE* for energy spacing (1
    keV if not specified), *trajectory* for simulating beam fluctuations. If it is None a (1024,
    1024) window is used with the beam center in the middle and no fluctuations.  *pixel_size*
    specifies the pixel spacing between the window points, if not specified 1 um is used.
    *ring_current* is the storage ring electric current.
    """
    if not pixel_size:
        pixel_size = 1 * q.um
    if not trajectory:
        trajectory = Trajectory([(512, 512, 0)] * pixel_size)
    if not dE:
        dE = 1 * q.keV

    return BendingMagnet(2.5 * q.GeV, ring_current, 1.5 * q.T, 30 * q.m, dE,
                         (142, 503) * q.um, pixel_size, trajectory)
