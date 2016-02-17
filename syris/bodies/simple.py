"""A static body."""
import numpy as np
import quantities as q
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

    def _project(self, shape, pixel_size, offset, t=0 * q.s, queue=None, out=None):
        """Project thickness."""
        orig_shape = self.thickness.shape
        orig_region = (0, 0) + orig_shape
        end = ((offset + shape * pixel_size) / self.pixel_size).simplified.magnitude
        end = np.round(end).astype(np.int)
        start = np.round((offset / self.pixel_size).simplified.magnitude).astype(np.int)
        # numpy integers are not understood by pyopencl's rectangle copy
        end = [int(num) for num in end]
        start = [int(num) for num in start]

        cy, cx = (max(0, start[0]), max(0, start[1]))
        crop_region = (cy, cx,
                       min(end[0], orig_shape[0]) - cy,
                       min(end[1], orig_shape[1]) - cx)

        py, px = (abs(min(0, start[0])), abs(min(0, start[1])))
        pad_region = (py, px, end[0] - start[0], end[1] - start[1])

        proj = self.thickness
        if crop_region != orig_region:
            proj = crop(self.thickness, crop_region)
        if pad_region != (0, 0) + crop_region[2:]:
            proj = pad(proj, pad_region)

        return rescale(proj, shape)
