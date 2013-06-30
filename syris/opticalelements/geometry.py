"""Geometry operations."""
import numpy as np
from numpy import linalg
import quantities as q
from scipy import interpolate, integrate
import itertools
import logging
import math

# Constants.
X = 0
Y = 1
Z = 2
X_AX = np.array([1, 0, 0])
Y_AX = np.array([0, 1, 0])
Z_AX = np.array([0, 0, 1])

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
        self._length = integrate.romberg(_length_curve_part,
                                         vals[0],
                                         vals[len(vals)-1],
                                         args=(tck,))

        # Compute points of the curve based on the curve length and
        # the smallest pixel size from x,y,z pixel sizes.
        # sqrt(12) factor to make sure the length of a step is < 1 voxel
        # the worst case is that two values are in the two most distanced
        # ends of two voxels, vx_1 - 0.5 in all directions and vx_2 - 0.5
        # in all directions (x,y,z), the distance between two such points is
        # then sqrt(12) which is twice the voxel's diagonal.
        # Assumes the distances are not larger then the diagonal.
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


def _length_curve_part(param, tck):
    """Compute length of a part of the parametrized curve with parameter
    *param* and a tuple *tck* consisting of knots, b-spline coefficients
    and the degree of the spline.
    """
    p_x, p_y, p_z = interpolate.splev(param, tck, der=1)
    return np.sqrt(p_x**2 + p_y**2 + p_z**2)


def length(vec):
    """Vector length.

    @param vec: a vector
    @return: vector length
    """
    sum_all = 0.0*vec.units**2
    for elem in vec:
        sum_all += elem**2

    return np.sqrt(sum_all)


def normalize(vec):
    """Normalize a vector *vec*."""
    if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
        return vec
    return np.array([x/float(length(vec)) for x in vec])*vec.units


def is_normalized(vec):
    """Test whether a vector is normalized.

    @param vec: a tuple
    """
    return length(vec) == 1.0*vec.units


def transform_vector(trans_matrix, vector):
    """Transform *vector* by the transformation matrix *trans_matrix* with
    dimensions (4,3) width x height.
    """
    return np.dot(trans_matrix, np.append(vector, 1)*vector.units)[:-1]


def translate(vec):
    """Translate the object by a vector *vec*. The transformation is
    in the backward form.
    """
    trans_matrix = np.identity(4)

    # minus because of the backward fashion
    trans_matrix[0][3] = -vec[0]
    trans_matrix[1][3] = -vec[1]
    trans_matrix[2][3] = -vec[2]

    return trans_matrix


def rotate(phi, axis, total_start=None):
    """Rotate the object by *phi* around vector *axis*, where
    *total_start* is the center of rotation point which results in
    transformation TRT^-1. The transformation is in the backward form.
    """
    if (not is_normalized(axis)):
        axis = normalize(axis)
    axis = axis.magnitude

    phi = phi.rescale(q.rad)
    sin = np.sin(phi)
    cos = np.cos(phi)
    v_x = axis[0]
    v_y = axis[1]
    v_z = axis[2]

    if total_start is not None:
        t_1 = translate(total_start)

    rot_matrix = np.identity(4)
    rot_matrix[0][0] = cos + pow(v_x, 2) * (1 - cos)
    rot_matrix[0][1] = v_x * v_y * (1 - cos) - v_z * sin
    rot_matrix[0][2] = v_x * v_z * (1 - cos) + v_y * sin
    rot_matrix[1][0] = v_x * v_y * (1 - cos) + v_z * sin
    rot_matrix[1][1] = cos + pow(v_y, 2) * (1 - cos)
    rot_matrix[1][2] = v_y * v_z * (1 - cos) - v_x * sin
    rot_matrix[2][0] = v_z * v_x * (1 - cos) - v_y * sin
    rot_matrix[2][1] = v_z * v_y * (1 - cos) + v_x * sin
    rot_matrix[2][2] = cos + pow(v_z, 2) * (1 - cos)

    if total_start is not None:
        t_2 = translate(-total_start)
        return np.dot(np.dot(t_2, linalg.inv(rot_matrix)), t_1)
    else:
        return linalg.inv(rot_matrix)


def scale(scale_vec):
    """Scale the object by scaling coefficients (kx, ky, kz)
    given by *sc_vec*. The transformation is in the backward form.
    """
    if (scale_vec[0] == 0 or scale_vec[1] == 0 or scale_vec[2] == 0):
        raise ValueError("All components of the scaling " +
                         "must be greater than 0")
    trans_matrix = np.identity(4)

    # 1/x because of the backward fashion
    trans_matrix[0][0] = 1.0 / scale_vec[0]
    trans_matrix[1][1] = 1.0 / scale_vec[1]
    trans_matrix[2][2] = 1.0 / scale_vec[2]

    return trans_matrix

def angle(vec_0, vec_1):
    """Angle between vectors *vec_0* and *vec_1*."""
    return math.atan2(length(np.cross(vec_0, vec_1)*q.dimensionless),
                          np.dot(vec_0, vec_1))*q.rad
