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

"""
Optical Elements are entities capable of producing wavefields as a function of time.
"""
import quantities as q
import syris.config as cfg
from syris.util import make_tuple


class OpticalElement(object):

    """An optical element capable of producing a wavefield as a function of time."""

    def transfer(
        self,
        shape,
        pixel_size,
        energy,
        exponent=False,
        offset=None,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        """Transfer function of the element in real space on an image plane of size *shape*, use
        *pixel_size*, *energy*, *offset* is the physical spatial offset of the element as (y, x),
        transfer at time *t*. If *exponent* is true, compute the exponent of the transfer function
        without applying the wavenumber. Use *queue* for OpenCL computations and *out* pyopencl
        array. If *block* is True, wait for the kernel to finish. If *check* is True, the function
        is checked for aliasing artefacts.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)
        if offset is None:
            offset = (0, 0) * q.m
        if queue is None:
            queue = cfg.OPENCL.queue

        return self._transfer(
            shape,
            pixel_size,
            energy,
            offset,
            exponent=exponent,
            t=t,
            queue=queue,
            out=out,
            check=check,
            block=block,
        )

    def transfer_fourier(
        self, shape, pixel_size, energy, t=None, queue=None, out=None, block=False
    ):
        """Transfer function of the element in Fourier space of size *shape*, use *pixel_size*,
        *energy* and comput the function at time *t*. Use *queue* for OpenCL computations and *out*
        pyopencl array. If *block* is True, wait for the kernel to finish.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)
        if queue is None:
            queue = cfg.OPENCL.queue

        return self._transfer_fourier(
            shape, pixel_size, energy, t=t, queue=queue, out=out, block=block
        )

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
        """Transfer function implementation."""
        raise NotImplementedError

    def _transfer_fourier(
        self, shape, pixel_size, energy, t=None, queue=None, out=None, block=False
    ):
        """Transfer function implementation."""
        raise NotImplementedError

    def get_next_time(self, t_0, distance):
        """Get next time at which the object will have traveled *distance*, the starting time is
        *t_0*.
        """
        raise NotImplementedError
