"""Sample geometry including affine transformations and trajectory
specification.
"""
import numpy as np
from scipy import interpolate, integrate
import itertools
import logging

# Constants.
X = 0
Y = 1
Z = 2
X_AX = (1, 0, 0)
Y_AX = (0, 1, 0)
Z_AX = (0, 0, 1)

AXES = {X: X_AX, Y: Y_AX, Z: Z_AX}

LOGGER = logging.getLogger(__name__)


class BoundingBox(object):
    """Class representing a graphical object's bounding box."""
    def __init__(self, points):
        """Create a bounding box from the object border *points*."""
        self._points = points
        # 3D -> 2D for overlap
        self._points_z_proj = np.array([x[:-1] for x in points]) * points.units

    @property
    def points(self):
        """The object border points."""
        return self._points

    def get_projected_points(self, axis):
        """Get the points projection by releasing the specified *axis*."""
        if axis == X:
            return np.array([x[1:] for x in self._points])*self._points.units
        elif axis == Y:
            return np.array([x[::2] for x in self._points])*self._points.units
        elif axis == Z:
            return np.array([x[:-1] for x in self._points])*self._points.units

    def get_min(self, axis=X):
        """Get minimum along the specified *axis*."""
        return np.min(self._points.flatten()[axis::3])

    def get_max(self, axis=X):
        """Get maximum along the specified *axis*."""
        return np.max(self._points.flatten()[axis::3])

    def overlaps2d(self, fov):
        """Determine whether the 2D projected bounding box overlaps a
        given region specified by *fov* as ((y0, y1), (x0, x1)).
        """
        d_x, d_y = zip(*self._points_z_proj.rescale(fov[0][0].units))
        x_0, x_1 = min(d_x), max(d_x)
        y_0, y_1 = min(d_y), max(d_y)
        f_y, f_x = fov

        # x and y coordinate interval tests.
        return f_x[0] <= x_1 and f_x[1] >= x_0 and\
            f_y[0] <= y_1 and f_y[1] >= y_0

    def merge(self, other):
        """Merge with *other* bounding box."""
        x_min = min(self.get_min(X), other.get_min(X))
        x_max = max(self.get_max(X), other.get_max(X))
        y_min = min(self.get_min(Y), other.get_min(Y))
        y_max = max(self.get_max(Y), other.get_max(Y))
        z_min = min(self.get_min(Z), other.get_min(Z))
        z_max = max(self.get_max(Z), other.get_max(Z))

        self._points = np.array(list(itertools.product([x_min, x_max],
                                                       [y_min, y_max],
                                                       [z_min, z_max]))) *\
            x_min.units
        self._points_z_proj = np.array(
            [x[:-1] for x in self._points])*self._points.units

    def __repr__(self):
        return "BoundingBox(%s)" % (str(self))

    def __str__(self):
        return "(x0=%g,y0=%g,z0=%g) -> (x1=%g,y1=%g,z1=%g)" % \
            (self.get_min(X),
             self.get_min(Y),
             self.get_min(Z),
             self.get_max(X),
             self.get_max(Y),
             self.get_max(Z))


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

        points = zip(*control_points)

        tck, vals = interpolate.splprep([points[0], points[1], points[2]], s=0)
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
