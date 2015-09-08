"""Module for beam filters which cause light attenuation. Filters are assumed
to be homogeneous, thus no phase change effects are introduced when
a wavefield passes through them.
"""
import numpy as np
import quantities as q
import scipy.interpolate as interp


class Filter(object):

    """Beam frequency filter."""

    def __init__(self, thickness, material):
        """Create a beam filter with projected *thickness* in beam direction
        and *material*.
        """
        self.thickness = thickness.simplified
        self.material = material

    def get_attenuation(self, energy):
        """Get attenuation at *energy*."""
        return (self.thickness *
                self.material.get_attenuation_coefficient(energy)).simplified.magnitude


class Scintillator(Filter):

    """Scintillator emits visible light when it is irradiated by X-rays."""

    def __init__(self, thickness, material, light_yields, energies, luminescence, wavelengths,
                 optical_ref_index):
        """Create a scintillator with *light_yields* [1 / keV] at *energies*, *luminescence* are the
        portions of total emmitted photons [1 / nm] with respect to visible light *wavelengths*,
        *optical_ref_index* is the refractive index between the scintillator material and air.
        """
        super(Scintillator, self).__init__(thickness, material)
        self._lights_yields = light_yields
        self._energies = energies
        self._luminescence = luminescence
        self._wavelengths = wavelengths
        self.opt_ref_index = optical_ref_index

        self._ly_tck = interp.splrep(self._energies.rescale(q.keV).magnitude,
                                     self._lights_yields.rescale(1 / q.keV).magnitude)
        self._lum_tck = interp.splrep(self._wavelengths.rescale(q.nm).magnitude,
                                      self._luminescence)

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
