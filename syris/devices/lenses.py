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

"""Lenses used in experiments."""
import math


class Lens(object):

    """Class holding lenses."""

    def __init__(
        self,
        magnification,
        na=None,
        f_number=None,
        focal_length=None,
        transmission_eff=1,
        sigma=None,
    ):
        """Create a lens with *magnification*, numerical aperture *na*, *f_number*, *focal_length*,
        transmission efficiency *transmission_eff* and *sigma* (y, x) giving the standard deviation
        of the point spread function approximated by a Gaussian.
        If *na* is None, it is computed from *focal_length*, *magnification* and *f_number*.
        """
        if transmission_eff < 0 or transmission_eff > 1:
            raise ValueError("Transmission efficiency must be between 0 and 1.")

        if na is None and focal_length is None and f_number is None:
            raise ValueError("Either 'na' must be specified or both 'focal_length' and 'f_number'")
        self.f_number = f_number
        self.focal_length = focal_length
        self.magnification = magnification
        self.transmission_eff = transmission_eff
        self.sigma = sigma
        self.na = na

    @property
    def numerical_aperture(self):
        """Lens numerical aperture."""
        na = self.na
        if na is None:
            na = self._compute_numerical_aperture()

        return na

    def _compute_numerical_aperture(self):
        r"""The numerical aperture is given by the half-angle between
        the lens entrance and the object plane distance, i.e.

        .. math::
            :nowrap:

            \begin{eqnarray}
                NA & = & \sin (\theta) \\
                NA & = & \sin \left[ \arctan \left( \frac{D}{2 d_0}
                \right) \right] \\
                D & = & \frac{f}{N} \\
                M & = & \frac{f}{d_0 - f} \\
                NA & = & \sin \left[ \arctan \left( \frac{f}{2 N d_0}
                \right) \right].
            \end{eqnarray}


        Where :math:`D` is the lens aperture diameter, :math:`f` is the
        focal length, :math:`N` is the f-number, :math:`M` is the
        magnification and :math:`d_0` is the distance between the lens
        entrance and the object plane.
        """
        d_0 = self.focal_length + self.focal_length / self.magnification

        return math.sin(math.atan(self.focal_length / (2 * self.f_number * d_0)))
