"""Geometry operations.

All the transformation operations are in the backward form, which means if
the order of operations is:
A = trans_1
B = trans_2
C = trams_3,
then in the forward form of the resulting transformation matrix would be
T = ABC yielding x' = ABCx = Tx. Backward form means that we calculate
the matrix in the form T^{-1} = C^{-1}B^{-1}A^{-1} = (ABC)^{-1}. Thus, we can
easily obtain x = T^{-1}x'.
"""
import numpy as np
from numpy import linalg
import quantities as q
from scipy import interpolate, integrate
import itertools
import logging
import math
from quantities.quantity import Quantity


X = 0
Y = 1
Z = 2
X_AX = np.array([1, 0, 0]) * q.dimensionless
Y_AX = np.array([0, 1, 0]) * q.dimensionless
Z_AX = np.array([0, 0, 1]) * q.dimensionless

AXES = {X: X_AX, Y: Y_AX, Z: Z_AX}

LOGGER = logging.getLogger(__name__)


class BoundingBox(object):

    """Class representing a graphical object's bounding box."""

    def __init__(self, points):
        """Create a bounding box from the object border *points*."""
        self._points = points.simplified
        # 3D -> 2D for overlap
        self._points_z_proj = np.array([x[:-1] for x in points]) * q.m

    @property
    def points(self):
        """The object border points."""
        return self._points

    def get_projected_points(self, axis):
        """Get the points projection by releasing the specified *axis*."""
        if axis == X:
            return np.array([x[1:] for x in self._points]) * q.m
        elif axis == Y:
            return np.array([x[::2] for x in self._points]) * q.m
        elif axis == Z:
            return np.array([x[:-1] for x in self._points]) * q.m

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
        d_x, d_y = zip(*self._points_z_proj.simplified)
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
                                                       [z_min, z_max]))) * q.m
        self._points_z_proj = np.array(
            [x[:-1] for x in self._points]) * q.m

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

    def __init__(self, points, traj_length, velocities=None,
                 v_0=0 * q.m / q.s):
        """Create trajectory from given *points* which represent (x,y,z)
        coordinates, *length* is the trajectory length, *velocities*
        is a list of (length, velocity) tuples, where the first component
        specifies the relative distance in which the object following
        the trajectory will achieve the given velocity. *v_0* is the initial
        velocity.
        """
        self._points = points.simplified
        self._length = traj_length.simplified
        self._v_0 = v_0.simplified
        self._length_velos = velocities

        if self._length_velos is not None:
            self._length_velos = [(dist.simplified, velo.simplified)
                                  for dist, velo in velocities]
            sum_dists = np.sum(zip(*self._length_velos)[0]) * q.m
            if sum_dists != self.length:
                raise ValueError("Specified velocities do not match the " +
                                 "trajectory length {0} != {1}".format(
                                     self.length, sum_dists))
            self._time_velos = self._distance_to_time()
            self._time = np.sum([time_velo[0] for
                                 time_velo in self._time_velos]) * q.s
        else:
            self._time_velos = None
            self._time = 0 * q.s

    def _get_point_index(self, abs_time):
        """Get index of a point in the points list at the time *abs_time*."""
        dist = self.get_distance(abs_time)
        if dist == 0:
            return 0

        return int(round(dist / self.length * (len(self.points) - 1)))

    def moved(self, t_0, t_1, pixel_size):
        """Return True if the trajectory moved between time *t_0* and *t_1*
        more than one pixel with respect to the given *pixel_size*.
        """
        p_0 = self.get_point(t_0)
        p_1 = self.get_point(t_1)

        return length(p_1 - p_0) > pixel_size

    def get_point(self, abs_time):
        """Get a point on the trajectory at the time *abs_time*."""
        return self.points[self._get_point_index(abs_time)]

    def get_direction(self, abs_time):
        """Get direction of the trajectory at the time *abs_time*. If p_0
        is the point at which the trajectory resides at *abs_time* and p_1
        is the point right after p_0, then the direction is defined as
        vector p_1 - p_0.
        """
        index = self._get_point_index(abs_time)
        if index == len(self.points) - 1:
            # If we are at the end of the trajectory we use the same direction
            # as for the previous point.
            index -= 1

        return normalize(self.points[index + 1] - self.points[index])

    def get_distance(self, abs_time):
        """Get distance from the beginning of the trajectory to the time
        given by *abs_time*.
        """
        if self._length_velos is None:
            return 0 * q.m
        total_time = 0 * q.s
        distance = 0 * q.m

        for i in range(len(self._time_velos)):
            t_1, v_1 = self._time_velos[i]
            v_0 = self._v_0 if i == 0 else self._time_velos[i - 1][1]
            total_time += t_1
            t_x = t_1 if abs_time >= total_time else abs_time - \
                total_time + t_1
            distance += _get_distance(t_x, t_1, v_0, v_1)
            if total_time >= abs_time:
                break

        return distance

    def _distance_to_time(self):
        """Convert (distance, velocity) pairs to (time, velocity) pairs."""
        velocities = []

        for i in range(len(self._length_velos)):
            s_1, v_1 = self._length_velos[i]
            v_0 = self._v_0 if i == 0 else self._length_velos[i - 1][1]
            t_1 = _get_time(s_1, v_0, v_1)
            velocities.append((t_1, v_1))

        return velocities

    @property
    def points(self):
        """Points of the spline."""
        return self._points

    @property
    def length(self):
        """Trajectory length."""
        return self._length

    @property
    def time(self):
        """Total time needed to travel the whole trajectory."""
        return self._time

    @property
    def length_profile(self):
        """Relative times and velocities as a list of tuples
        (relative_time, velocity).
        """
        return self._length_velos

    @property
    def time_profile(self):
        """Relative times and velocities as a list of tuples
        (relative_time, velocity).
        """
        return self._time_velos


def interpolate_points(control_points, pixel_size):
    """Create points by interpolating the *control_points*. Thanks to given
    *pixel_size* the resulting sampling has sub-voxel precision. Return
    a tuple (points, length), where length is the length of the created
    trajectory.
    """
    control_points = control_points.simplified

    if len(control_points) == 1:
        return control_points, 0 * q.m

    points = zip(*control_points)

    tck, vals = interpolate.splprep([points[0], points[1], points[2]], s=0)
    p_length = integrate.romberg(_length_curve_part,
                                 vals[0], vals[len(vals) - 1],
                                 args=(tck,)) * q.m

    # Compute points of the curve based on the curve length and pixel size.
    # sqrt(12) factor to make sure the length of a step is < 1 voxel
    # the worst case is that two values are in the two most distanced
    # ends of two voxels, vx_1 - 0.5 in all directions and vx_2 - 0.5
    # in all directions (x,y,z), the distance between two such points is
    # then sqrt(12) which is twice the voxel's diagonal.
    # Assumes the distances are not larger then the diagonal.
    size = p_length / pixel_size * np.sqrt(12)
    x_new, y_new, z_new = interpolate.splev(np.linspace(0, 1,
                                                        size.simplified), tck)

    return zip(x_new, y_new, z_new) * q.m, p_length


def _get_time(dist, v_0, v_1):
    """Get time needed for traveling distance *dist* with initial velocity
    *v_0* and ending velocity *v_1*.
    """
    if v_0 == v_1:
        return dist / v_0
    else:
        return 2 * dist / (v_0 + v_1)


def _get_distance(t_x, t_1, v_0, v_1):
    """Get distance at time *t_x* in a window taking *t_1* time with starting
    velocity *v_0* and ending with velocity *v_1*.
    """
    if v_0 == v_1:
        # Constant velocity.
        return v_0 * t_x
    else:
        accel = (v_1 - v_0) / t_1
        # Acceleration if accel > 0, otherwise deceleration.
        return v_0 * t_x + 0.5 * accel * t_x ** 2


def _length_curve_part(param, tck):
    """Compute length of a part of the parametrized curve with parameter
    *param* and a tuple *tck* consisting of knots, b-spline coefficients
    and the degree of the spline.
    """
    p_x, p_y, p_z = interpolate.splev(param, tck, der=1)
    return np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)


def length(vector):
    """Get length of a *vector*."""
    return np.sqrt(np.sum(vector ** 2))


def normalize(vector):
    """Normalize a *vector*."""
    if length(vector) == 0:
        if vector.__class__ == Quantity:
            vector = vector.magnitude
        return vector * q.dimensionless
    else:
        return vector / length(vector)


def is_normalized(vector):
    """Test whether a *vector* is normalized."""
    return length(vector) == 1.0 * vector.units


def transform_vector(trans_matrix, vector):
    """Transform *vector* by the transformation matrix *trans_matrix* with
    dimensions (4,3) width x height.
    """
    vector = vector.simplified

    return np.dot(trans_matrix, np.append(vector, 1) * vector.units)[:-1]


def translate(vec):
    """Translate the object by a vector *vec*. The transformation is
    in the backward form and the vector is _always_ transformed into meters.
    """
    vec = vec.simplified
    trans_matrix = np.identity(4)

    # minus because of the backward fashion
    trans_matrix[0][3] = -vec[0]
    trans_matrix[1][3] = -vec[1]
    trans_matrix[2][3] = -vec[2]

    return trans_matrix


def rotate(phi, axis, total_start=None):
    """Rotate the object by *phi* around vector *axis*, where
    *total_start* is the center of rotation point which results in
    transformation TRT^-1. The transformation is in the backward form and
    the angle is _always_ rescaled to radians.
    """
    axis = normalize(axis.simplified)

    phi = phi.simplified
    sin = np.sin(phi)
    cos = np.cos(phi)
    v_x = axis[0]
    v_y = axis[1]
    v_z = axis[2]

    if total_start is not None:
        total_start = total_start.simplified
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
    if (scale_vec[0] <= 0 or scale_vec[1] <= 0 or scale_vec[2] <= 0):
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
    vec_1 = vec_1.rescale(vec_0.units)

    return math.atan2(length(np.cross(vec_0, vec_1) * q.dimensionless),
                      np.dot(vec_0, vec_1)) * q.rad
