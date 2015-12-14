"""Detector composed of a scintillator, a lens and a camera."""
import numpy as np
from syris.physics import compute_collection


class Detector(object):

    """A detector consisting of a camera and an objective lens."""

    def __init__(self, scintillator, lens, camera):
        """Create a detector with *lens* and a *camera*."""
        self.scintillator = scintillator
        self.lens = lens
        self.camera = camera

    @property
    def pixel_size(self):
        return self.camera.pixel_size.simplified / self.lens.magnification

    def get_visible_attenuation(self, wavelengths=None):
        """Get the attenuation coefficient for visible light *wavelengths* [dimensionless]."""
        if wavelengths is None:
            wavelengths = self.camera.wavelengths
        if len(wavelengths) < 2:
            raise ValueError('Wavelengths must contain at least two values')
        d_lam = wavelengths[1] - wavelengths[0]
        luminescence = self.scintillator.get_luminescence(wavelengths) * d_lam
        qe = self.camera.get_quantum_efficiency(wavelengths)
        coeff = np.sum(luminescence * qe)
        na = self.lens.numerical_aperture
        coll = compute_collection(na, self.scintillator.opt_ref_index)

        return coeff * self.lens.transmission_eff * coll

    def convert(self, photons, energy, wavelengths=None):
        """Convert X-ray *photons* at *energy* to visible light photons with *wavelengths*."""
        if wavelengths is None:
            wavelengths = self.camera.wavelengths
        x_to_vis = self.scintillator.get_conversion_factor(energy).magnitude
        vis = self.get_visible_attenuation(wavelengths).magnitude

        return photons * x_to_vis * vis
