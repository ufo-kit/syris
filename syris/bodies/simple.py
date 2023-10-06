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

"""A static body."""
import numpy as np
import quantities as q
import syris.config as cfg
import syris.gpu.util as g_util
from syris.bodies.base import Body
from syris.imageprocessing import crop, pad, rescale
from syris.util import make_tuple


class StaticBody(Body):

    """A static body is defined by its projected *thickness*, which is a quantity and it is
    always converted to meters, thus the :meth:`~Body.project` method always returns the
    projection in meters. *pixel_size* is the pixel size of the *thickness* and *material* is
    a :class:`syris.materials.Material` instance. Use OpenCL command *queue*.
    """

    def __init__(self, thickness, pixel_size, material=None, queue=None):
        super(StaticBody, self).__init__(material)
        self.thickness = g_util.get_array(thickness.simplified.magnitude, queue=queue)
        self.pixel_size = make_tuple(pixel_size, num_dims=2)

    def get_next_time(self, t_0, distance):
        """A simple body doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _project(self, shape, pixel_size, offset, t=None, queue=None, out=None, block=False):
        """Project thickness."""
        orig_shape = self.thickness.shape
        orig_region = (0, 0) + orig_shape
        end = ((offset + shape * pixel_size) / self.pixel_size).simplified.magnitude
        end = np.round(end).astype(int)
        start = np.round((offset / self.pixel_size).simplified.magnitude).astype(int)
        # numpy integers are not understood by pyopencl's rectangle copy
        end = [int(num) for num in end]
        start = [int(num) for num in start]

        cy, cx = (max(0, start[0]), max(0, start[1]))
        crop_region = (cy, cx, min(end[0], orig_shape[0]) - cy, min(end[1], orig_shape[1]) - cx)

        py, px = (abs(min(0, start[0])), abs(min(0, start[1])))
        pad_region = (py, px, end[0] - start[0], end[1] - start[1])

        proj = self.thickness
        if crop_region != orig_region:
            proj = crop(self.thickness, crop_region, block=block)
        if pad_region != (0, 0) + crop_region[2:]:
            proj = pad(proj, pad_region, block=block)
        if proj.shape != shape:
            proj = rescale(proj, shape, block=block)

        return proj


def make_grid(
    n, period, width=1 * q.m, thickness=1 * q.m, pixel_size=1 * q.m, material=None, queue=None
):
    """Make a rectangluar grid with shape (*n*, *n*), the bars are spaced *period* and are *width*
    in diameter. *thickness* is the projected thickness and *pixel_size*, *material* and *queue*,
    which is an OpenCL command queue, are used to create :class:`.StaticBody`.
    """
    ps = pixel_size.simplified.magnitude
    period = int(np.round(period.simplified.magnitude / ps))
    width = int(np.round(width.simplified.magnitude / ps))

    image = np.zeros((n, n), dtype=cfg.PRECISION.np_float)

    for i in range(-width // 2, width // 2):
        if i < 0:
            i = period + i
        image[i::period, :] = 1
        image[:, i::period] = 1

    return StaticBody(image * thickness, pixel_size, material=material, queue=queue)


def make_sphere(n, radius, pixel_size=1 * q.m, material=None, queue=None):
    """Make a sphere with image shape (*n*, *n*), *radius* and *pixel_size*. Sphere center is in n /
    2 + 0.5, which means between two adjacent pixels. *pixel_size*, *material* and *queue*, which is
    an OpenCL command queue, are used to create :class:`.StaticBody`.
    """
    pixel_size = make_tuple(pixel_size, num_dims=2)
    image = np.zeros((n, n), dtype=cfg.PRECISION.np_float)
    y, x = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
    x = (x + 0.5) * pixel_size[1].simplified.magnitude
    y = (y + 0.5) * pixel_size[0].simplified.magnitude
    radius = radius.simplified.magnitude
    valid = np.where(x ** 2 + y ** 2 < radius ** 2)
    image[valid] = 2 * np.sqrt(radius ** 2 - x[valid] ** 2 - y[valid] ** 2)

    return StaticBody(image * q.m, pixel_size, material=material, queue=queue)
