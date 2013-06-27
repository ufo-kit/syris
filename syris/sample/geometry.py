"""Sample geometry including affine transformations and trajectory
specification.
"""
import numpy as np
from scipy import interpolate, integrate

# Constants.
X = 0
Y = 1
Z = 2
X_AX = (1, 0, 0)
Y_AX = (0, 1, 0)
Z_AX = (0, 0, 1)

AXES = {X: X_AX, Y: Y_AX, Z: Z_AX}


class Trajectory(object):
    """Class representing object's trajectory.

    Trajectory is a spline interpolated from a set of points.
    """
    def __init__(self, control_points, pixel_size):
        """Create trajectory with *control_points* as a list of (x,y,z)
        tuples representing control points of the curve (if only one is
        specified, the object does not move and its center is the control
        point specified). *pixel_size* is the minimum of the x, y, z
        pixel size.
        """
        self._control_points = control_points
        self._pixel_size = pixel_size
        if len(control_points) == 1:
            # the object does not move
            self._length = 0
            self._points = control_points
            return

        x, y, z = zip(*control_points)

        tck, vals = interpolate.splprep([x, y, z], s=0)
        self._length = integrate.romberg(_length_part,
                                         vals[0],
                                         vals[len(vals)-1],
                                         args=(tck,))

        # Compute points of the curve based on the curve length and
        # the smallest pixel size from x,y,z pixel sizes.
        # sqrt(12) factor to make sure the length of a step is < 1 voxel
        # the worst case is that two values are in the two most distanced
        # ends of two voxels, vx_1 - 0.5 in all directions and vx_2 - 0.5
        # in all directions (x,y,z), the distance between two such points is
        # then sqrt(12) which is twice the voxel's diagonal
        # TODO: assumes the distances are not larger then the diagonal, check!
        x_new, y_new, z_new = interpolate.splev(
            np.linspace(0, 1,
                        self._length*(1.0/self._pixel_size) *
                        np.sqrt(12)), tck)
        self._points = zip(x_new, y_new, z_new)

    @property
    def control_points(self):
        """Control points used for spline creation."""
        return self._control_points

    @property
    def pixel_size(self):
        """Pixel size."""
        return self._pixel_size

    @property
    def points(self):
        """Points of the spline."""
        return self._points

    @property
    def length(self):
        """Compute the length of the trajectory.

        @return: curve length in mm
        """
        return self._length


def _length_part(param, tck):
    """Compute length of a part of the parametrized curve with parameter
    *param* and a tuple *tck* consisting of knots, b-spline coefficients
    and the degree of the spline.
    """
    p_x, p_y, p_z = interpolate.splev(param, tck, der=1)
    return np.sqrt(p_x**2 + p_y**2 + p_z**2)
