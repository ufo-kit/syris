"""
Module contains optical elements which consist of a graphical object and a material.
"""
import quantities as q
from syris.physics import transfer, energy_to_wavelength
from syris.util import make_tuple


class OpticalElement(object):

    """An optical element capable of producing a wavefield as a function of time."""

    def transfer(self, shape, pixel_size, energy, t=0 * q.s, queue=None, out=None):
        """Transfer function of the element on an image plane of size *shape*, use *pixel_size*,
        *energy*, time *t*. Use *queue* for OpenCL computations and *out* pyopencl array.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)

        return self._transfer(shape, pixel_size, energy, t=t, queue=queue, out=out)

    def _transfer(self, shape, pixel_size, energy, t=0 * q.s, queue=None, out=None):
        """Transfer function implementation."""
        raise NotImplementedError

    def get_next_time(self, t_0, distance):
        """Get next time at which the object will have traveled *distance*, the starting time is
        *t_0*.
        """
        raise NotImplementedError
