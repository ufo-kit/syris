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
from syris.opticalelements import OpticalElement
from syris.util import make_tuple


class BendingMagnet(OpticalElement):

    """Bending magnet X-ray source."""

    _SR_CONST = 3 * fine_structure_constant.simplified / (4 * np.pi ** 2)

    def __init__(self, electron_energy, el_current, magnetic_field, sample_distance, energies, size,
                 pixel_size, height, profile_approx=True, trajectory=None):
        """Create a BendingMagnet source with electron beam *electron_energy*, electric
        *el_current*, *magnetic_field*, place it into *sample_distance* (distance between the source
        and a sample), take into account all *energies* and set its *size* (y, x) specified as FWHM
        and approximate it by a Gaussian. *pixel_size* is the effective pixel size. *trajectory* is
        the trajectory defining the source position in space and time. The source (0, 0, 0) is in
        the middle of the produced image. *height* is the number of points for creating vertical
        profiles. If *profile_approx* is True, the profile at a given vertical observation angle
        will not be integrated over the relevant energies but will be calculated for the mean energy
        and multiplied by :math:`\Delta E`.
        """
        super(BendingMagnet, self).__init__()
        self.electron_energy = electron_energy.simplified
        self.el_current = el_current.simplified
        self.magnetic_field = magnetic_field
        self.sample_distance = sample_distance.simplified
        self.energies = energies
        self.size = size.simplified
        self.pixel_size = make_tuple(pixel_size, num_dims=2)
        self.trajectory = trajectory
        self._d_energy = 0 if len(self.energies) == 1 else self.energies[1] - self.energies[0]
        self._angle_step = np.arctan(pixel_size.simplified / self.sample_distance.simplified)

        # Compute the vertical span based on how much the source deviates in y-direction
        shift_down = shift_up = 0
        if trajectory:
            def get_shift(func):
                extrema = np.abs(func(trajectory.points[1, :]))
                return int(np.ceil((extrema / pixel_size).simplified.magnitude))

            shift_down = get_shift(min)
            shift_up = get_shift(max)

        half = height / 2
        aperture = np.arange(-half - shift_down, half + shift_up) * pixel_size
        self._angles = np.arctan((aperture / sample_distance).simplified.magnitude) * q.rad
        # Index to the angles where the angle is zero
        self._mean_angle_index = shift_down + height / 2

        self.profile_approx = profile_approx
        self._profiles = [self._create_vertical_profile(e) for e in self.energies]

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

    def _get_full_profile(self, energy):
        """Get the vertical profile based on energies integration.
        If there are two energies e_0 and e_1, the profile at
        a specific angle will be calculated by integrating the energy
        range from e_0 - d_e, e_0 + d_e, where d_e = e_1 - e_0.
        """
        def _get_flux_wrapper(photon_energy, vertical_angle):
            """Get rid of quantities, because scipy.romberg
            cannot work with them.
            """
            flux = self.get_flux(photon_energy * q.keV, vertical_angle * q.rad).magnitude
            # Conversion is 1e3 * dE / E, 1e3 for the 0.1 % BW. Integration takes care of the dE so
            # we need to correct by 1e3 / E
            return flux * 1e3 / photon_energy

        def _get_flux_at_angle(angle, energy, d_energy):
            if len(self.energies) > 1:
                e_0 = energy - d_energy / 2.0
                e_1 = energy + d_energy / 2.0
                return integrate.romberg(_get_flux_wrapper, e_0, e_1, args=(angle,))
            else:
                return self.get_flux(energy * q.keV, angle * q.rad)

        get_profiles = np.vectorize(_get_flux_at_angle)

        energy = energy.rescale(q.keV).magnitude
        d_energy = self._d_energy.rescale(q.keV).magnitude

        return get_profiles(self._angles.rescale(q.rad).magnitude, energy, d_energy) / q.s

    def get_vertical_profile(self, energy):
        """Get flux profile at *energy*."""
        if energy in self.energies:
            # Lookup
            profile = self._profiles[np.where(self.energies == energy)[0][0]]
        else:
            profile = self._create_vertical_profile(energy)

        return profile

    def get_next_time(self, t_0, distance):
        """Get the next time when the source will have moved more than *distance*."""
        return self.trajectory.get_next_time_from_distance(t_0, distance)

    def _transfer(self, shape, pixel_size, energy, t=0 * q.s, queue=None, out=None):
        """Compute the flat field wavefield."""
        if queue is None:
            queue = cfg.OPENCL.queue

        ps = make_tuple(pixel_size)
        # Compute the shift in pixels
        y = self.trajectory.get_point(t)[1]
        shift = int((y / ps[0]).simplified.magnitude)

        # Shift and crop the profile
        profile = self.get_vertical_profile(energy).rescale(1 / q.s).magnitude
        start = self._mean_angle_index + shift - shape[0] / 2
        stop = self._mean_angle_index + shift + shape[0] / 2
        if start < 0 or stop > profile.shape[0]:
            raise ValueError('Height extends the source computed vertical profiles')

        profile = cl_array.to_device(queue, profile[start:stop].astype(cfg.PRECISION.np_float))
        if out is None:
            out = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_cplx)

        cfg.OPENCL.programs['physics'].make_flat(queue,
                                                 shape[::-1],
                                                 None,
                                                 out.data,
                                                 profile.data)

        return out

    def _create_vertical_profile(self, energy):
        if self.profile_approx:
            # Much faster but less precise.
            # dE / E = 1e-3 = 0.1 % BW, we need to convert it to the actual bandwidth of the
            # energies we use, so the result is 1e3 * dE / E
            bw_conv = 1e3 * (self._d_energy / energy).simplified.magnitude
            result = self.get_flux(energy, self._angles) * bw_conv
        else:
            # Full energy integration.
            result = self._get_full_profile(energy)

        return result

    def get_flux(self, photon_energy, vertical_angle):
        """Get the photon flux coming from the source consisting of photons
        with *photon_energy* and get it at the vertical observation angle
        *vertical_angle*.
        """
        gama = Quantity(self.electron_energy /
                        (qe.electron_mass * q.c ** 2)).simplified
        gama_psi = gama * vertical_angle.rescale(q.rad)
        norm_energy = photon_energy.rescale(self.critical_energy.units) / \
            self.critical_energy
        xi = Quantity(0.5 * norm_energy.magnitude *
                      (1.0 + gama_psi ** 2) ** (3.0 / 2)).magnitude

        # 1e-3 for 0.1 % BW
        return Quantity(BendingMagnet._SR_CONST * gama ** 2 *
                        self.el_current / q.elementary_charge *
                        norm_energy ** 2 *
                       (1.0 + gama_psi ** 2) ** 2 *
                       (special.kv(2.0 / 3, xi) ** 2 + gama_psi ** 2 /
                       (1.0 + gama_psi ** 2) * special.kv(1.0 / 3, xi) ** 2) *
                        self._angle_step.rescale(q.rad) ** 2 * 1e-3).simplified
