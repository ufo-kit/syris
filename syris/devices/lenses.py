"""Lenses used in experiments."""
import math


class Lens(object):

    """Class holding lenses."""

    def __init__(self, f_number, focal_length, magnification,
                 transmission_eff, sigma):
        """Create a lens with *f_number*, *focal_length*,
        *magnification*, transmission efficiency *transmission_eff*
        and *sigma* (y, x) giving the standard deviation of the point
        spread function approximated by a Gaussian.
        """
        if transmission_eff < 0 or transmission_eff > 1:
            raise ValueError("Transmission efficiency must " +
                             "be between 0 and 1.")

        self.f_number = f_number
        self.focal_length = focal_length.simplified
        self.magnification = magnification
        self.transmission_eff = transmission_eff
        self.sigma = sigma

    @property
    def numerical_aperture(self):
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
        d_0 = (self.focal_length + self.focal_length / self.magnification)

        return math.sin(math.atan(self.focal_length /
                                  (2 * self.f_number * d_0)))
