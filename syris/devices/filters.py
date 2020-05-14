"""Module for beam filters which cause light attenuation. Filters are assumed
to be homogeneous, thus no phase change effects are introduced when
a wavefield passes through them.
"""
import numpy as np
import quantities as q
import scipy.interpolate as interp
import syris.config as cfg
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength
from syris.util import get_gauss


class Filter(OpticalElement):

    """Beam frequency filter."""

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s


class GaussianFilter(OpticalElement):

    """Gaussian beam filter."""

    def __init__(self, energies, center, sigma, peak_transmission=1):
        """Create a Gussian beam filter for *energies* [keV], center it at *center* [keV] and use
        std *sigma* [keV]. *peak_transmission* specifies the transmitted intensity for energy
        *center*, i.e. this is the highest transmitted intensity.
        """
        if len(energies) < 4:
            raise ValueError("Number of energy points too low for interpolation")
        energies = energies.rescale(q.keV).magnitude
        center = center.rescale(q.keV).magnitude
        sigma = sigma.rescale(q.keV).magnitude
        profile = get_gauss(energies, center, sigma) * peak_transmission * q.keV
        self._tck = interp.splrep(energies, profile)

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _transfer(
        self,
        shape,
        pixel_size,
        energy,
        offset,
        exponent=False,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        """Transfer function implementation. Only *energy* is relevant because a filter has the same
        thickness everywhere.
        """
        coeff = interp.splev(energy.rescale(q.keV).magnitude, self._tck)
        eps = np.finfo(cfg.PRECISION.np_float).eps

        if exponent:
            result = np.log(coeff) / 2 if np.abs(coeff) > eps else -np.inf
        else:
            result = np.sqrt(coeff) if np.abs(coeff) > eps else 0.0

        return cfg.PRECISION.np_cplx(result)


class MaterialFilter(Filter):

    """Beam frequency filter."""

    def __init__(self, thickness, material):
        """Create a beam filter with projected *thickness* in beam direction
        and *material*.
        """
        self.thickness = thickness.simplified
        self.material = material

    def get_attenuation(self, energy):
        """Get attenuation at *energy*."""
        return (
            self.thickness * self.material.get_attenuation_coefficient(energy)
        ).simplified.magnitude

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _transfer(
        self,
        shape,
        pixel_size,
        energy,
        offset,
        exponent=False,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        """Transfer function implementation. Only *energy* is relevant because a filter has the same
        thickness everywhere.
        """
        lam = energy_to_wavelength(energy).simplified.magnitude
        thickness = self.thickness.simplified.magnitude
        ri = self.material.get_refractive_index(energy)
        result = -2 * np.pi / lam * thickness * (ri.imag + ri.real * 1j)

        if not exponent:
            result = np.exp(result)

        return result.astype(cfg.PRECISION.np_cplx)


class Scintillator(MaterialFilter):

    """Scintillator emits visible light when it is irradiated by X-rays."""

    def __init__(
        self,
        thickness,
        material,
        light_yields,
        energies,
        luminescence,
        wavelengths,
        optical_ref_index,
    ):
        """Create a scintillator with *light_yields* [1 / keV] at *energies*, *luminescence* are the
        portions of total emmitted photons per some portion of wavelengths [1 / nm] (they are
        normalized so that their integral is 1) with respect to visible light *wavelengths*,
        *optical_ref_index* is the refractive index between the scintillator material and air.
        """
        super(Scintillator, self).__init__(thickness, material)
        self._lights_yields = light_yields
        self._energies = energies
        self._wavelengths = wavelengths
        self._luminescence = luminescence / luminescence.sum() / self.d_wavelength
        self.opt_ref_index = optical_ref_index

        self._ly_tck = interp.splrep(
            self._energies.rescale(q.keV).magnitude,
            self._lights_yields.rescale(1 / q.keV).magnitude,
        )
        self._lum_tck = interp.splrep(
            self._wavelengths.rescale(q.nm).magnitude,
            self._luminescence.rescale(1 / q.nm).magnitude,
        )

    @property
    def wavelengths(self):
        """Wavelengths for which the emission is defined."""
        return self._wavelengths

    @property
    def d_wavelength(self):
        """Wavelength spacing."""
        return (self.wavelengths[1] - self.wavelengths[0]).rescale(q.nm)

    def get_light_yield(self, energy):
        """Get light yield at *energy* [1 / keV]."""
        return interp.splev(energy.rescale(q.keV).magnitude, self._ly_tck) / q.keV

    def get_luminescence(self, wavelength):
        """Get luminescence at *wavelength* [1 / nm]."""
        return interp.splev(wavelength.rescale(q.nm).magnitude, self._lum_tck) / q.nm

    def get_conversion_factor(self, energy):
        """Get the conversion factor to convert X-ray photons to visible light photons
        [dimensionless].
        """
        absorbed = 1 - np.exp(-self.get_attenuation(energy))
        ly = self.get_light_yield(energy)

        return absorbed * ly * energy.rescale(q.keV)
