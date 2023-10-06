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
            raise ValueError("Wavelengths must contain at least two values")
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
        x_to_vis = self.scintillator.get_conversion_factor(energy).simplified.item()
        vis = self.get_visible_attenuation(wavelengths).simplified.item()

        return photons * x_to_vis * vis
