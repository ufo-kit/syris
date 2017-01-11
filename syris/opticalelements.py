"""
Optical Elements are entities capable of producing wavefields as a function of time.
"""
import quantities as q
from syris.physics import transfer, energy_to_wavelength
from syris.util import make_tuple


class OpticalElement(object):

    """An optical element capable of producing a wavefield as a function of time."""

    def transfer(self, shape, pixel_size, energy, exponent=False, offset=None, t=None,
                 queue=None, out=None, check=True, block=False):
        """Transfer function of the element on an image plane of size *shape*, use *pixel_size*,
        *energy*, *offset* is the physical spatial offset of the element as (y, x), transfer at time
        *t*. If *exponent* is true, compute the exponent of the transfer function without applying
        the wavenumber. Use *queue* for OpenCL computations and *out* pyopencl array. If *block* is
        True, wait for the kernel to finish. If *check* is True, the function is checked for
        aliasing artefacts.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)
        if offset is None:
            offset = (0, 0) * q.m

        return self._transfer(shape, pixel_size, energy, offset, exponent=exponent, t=t,
                              queue=queue, out=out, check=check, block=block)

    def _transfer(self, shape, pixel_size, energy, offset, exponent=False, t=None,
                  queue=None, out=None, check=True, block=False):
        """Transfer function implementation."""
        raise NotImplementedError

    def get_next_time(self, t_0, distance):
        """Get next time at which the object will have traveled *distance*, the starting time is
        *t_0*.
        """
        raise NotImplementedError
