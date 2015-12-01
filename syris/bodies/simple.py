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
    a :class:`syris.materials.Material` instance.
    """

    def __init__(self, thickness, pixel_size, material=None):
        super(StaticBody, self).__init__(material)
        self.thickness = g_util.get_array(thickness.simplified.magnitude)
        self.pixel_size = make_tuple(pixel_size, num_dims=2)

    def get_next_time(self, t_0, distance):
        """A simple body doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        """Project thickness."""
        if shape == self.thickness.shape and np.array_equal(pixel_size, self.pixel_size):
            result = self.thickness
        else:
            src_fov = self.thickness.shape * self.pixel_size
            dst_fov = shape * pixel_size
            fov_coeff = (dst_fov / src_fov).simplified.magnitude
            orig_shape = self.thickness.shape
            fov_shape = orig_shape * fov_coeff
            fov_shape = (int(np.ceil(fov_shape[0])), int(np.ceil(fov_shape[1])))
            # Do not use just one of them because it might be exactly 1
            representative = min(fov_coeff)
            if (fov_coeff[0] < 1) ^ (fov_coeff[1] < 1) and fov_coeff[0] != 1 and fov_coeff[1] != 1:
                raise ValueError('Cannot simultaneously crop and pad image')
            elif representative < 1:
                y_0 = (orig_shape[0] - fov_shape[0]) / 2
                x_0 = (orig_shape[1] - fov_shape[1]) / 2
                res = crop(self.thickness, (y_0, x_0) + fov_shape)
            else:
                y_0 = (fov_shape[0] - orig_shape[0]) / 2
                x_0 = (fov_shape[1] - orig_shape[1]) / 2
                res = pad(self.thickness, (y_0, x_0) + fov_shape)

            result = rescale(res, shape)

        return result

