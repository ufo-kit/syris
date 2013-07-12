"""Lenses used in experiments."""


class Lens(object):

    """Class holsing lenses."""

    def __init__(self, transmission_eff, numerical_aperture, sigma):
        """Create a lens with transmission efficiency *transmission_eff*,
        *numerical_aperture* and *sigma* (y, x) giving the standard
        deviation of the point spread function approximated by a Gaussian.
        """
        if transmission_eff < 0 or transmission_eff > 1:
            raise ValueError("Transmission efficiency must " +
                             "be between 0 and 1.")
        if numerical_aperture < 1:
            raise ValueError("Numerical aperture must be greater than 1.")
        self.transmission_eff = transmission_eff
        self.numerical_aperture = numerical_aperture
        self.sigma = sigma
