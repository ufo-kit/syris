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
from scipy import interpolate as interp, integrate as integ
import itertools
import logging
import math
from quantities.quantity import Quantity
from syris import math as smath


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

    @property
    def points(self):
        """The object border points."""
        return self._points

    @property
    def roi(self):
        """
        Return range of interest defined by the bounding box as
        (y_0, x_0, y_1, x_1).
        """

        return self.get_min(Y), self.get_min(X), \
            self.get_max(Y), self.get_max(X)

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

    def overlaps(self, other):
        """
        Determine if the bounding box XY-projection overlaps XY-projection
        of *other* bounding box.
        """
        x_interval_0 = self.get_min(X), self.get_max(X)
        y_interval_0 = self.get_min(Y), self.get_max(Y)
        x_interval_1 = other.get_min(X), other.get_max(X)
        y_interval_1 = other.get_min(Y), other.get_max(Y)

        return overlap(x_interval_0, x_interval_1) and \
            overlap(y_interval_0, y_interval_1)

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

    def __init__(self, control_points, time_dist=None, velocity=None):
        """
        TODO: describe when you are done!
        """
        self._control_points = control_points.simplified

        if time_dist is not None and velocity is not None:
            raise ValueError("Either times and distances or velocity " +
                             "must be given, but not both at the same time.")

        if len(self.control_points) == 1 or (time_dist is None and
                                             velocity is None):
            # Static trajectory or no velocity profile specified.
            self._tck = None
            self._u = None
            self._length = 0 * q.m
            self._times = self._distances = self._time_tck = None
        else:
            # Positions
            self._tck, self._u = \
                interp.splprep(zip(*control_points.simplified))
            self._length = self._get_length()

            # Derivatives of the spline
            x_d, y_d, z_d = interp.splev(self._u, self._tck, der=1)
            self._derivatives = np.array(zip(x_d, y_d, z_d))

            # Velocity profile
            if velocity is not None:
                # Constant velocity
                time_dist = get_constant_velocity(velocity,
                                                  self.length / velocity)

            t_0, s_0 = zip(*time_dist)
            t_0 = np.array(t_0)
            s_0 = np.array(s_0)
            # Time must be monotonic and distances cannot be negative
            if len(set(t_0)) != len(t_0) or np.any(t_0 != np.sort(t_0)):
                raise ValueError("Time must be strictly monotonically rising.")
            if np.any(t_0 < 0):
                raise ValueError("Time cannot be negative.")
            if np.any(s_0 < 0):
                raise ValueError("Distance cannot be negative.")

            t_units = time_dist[0][0].units
            d_units = time_dist[0][1].units
            t_0 = Quantity(t_0 * t_units).simplified.magnitude
            s_0 = Quantity(s_0 * d_units).simplified.magnitude
            self._times, self._distances = interpolate_1d(t_0, s_0, 100)
            self._time_tck = interp.splrep(self._times, self._distances)

    @property
    def control_points(self):
        """Control points used by the trajectory."""
        return tuple(self._control_points) * q.m

    @property
    def length(self):
        """Trajectory length."""
        return self._length

    @property
    def time(self):
        """Total time needed to travel the whole trajectory."""
        if self._times is not None:
            return (self._times[-1] - self._times[0]) * q.s
        else:
            return 0 * q.s

    def get_next_time(self, t_0, delta_distance):
        """
        Get next time at which the trajectory will have traveled
        *delta_distance*, the starting time is *t_0*.
        """
        t_0 = t_0.simplified.magnitude
        d_s = delta_distance.simplified.magnitude

        if t_0 < 0:
            raise ValueError("Time cannot be negative.")

        if t_0 > self.time or self._times is None:
            return None

        result = smath.difference_root(t_0, self._time_tck, d_s)

        if result is not None:
            result = result * q.s

        return result

    def get_point(self, abs_time):
        """Get a point on the trajectory at the time *abs_time*."""
        if abs_time < 0:
            raise ValueError("Time cannot be negative.")
        if abs_time > self.time:
            abs_time = self.time

        if self._tck is None:
            # Stationary trajectory.
            result = self._control_points[0]
        else:
            if self._times is None:
                # Stationary trajectory.
                result = self._control_points[0]
            else:
                result = interp.splev(self._get_u(abs_time), self._tck) * q.m

        return result

    def get_direction(self, abs_time):
        """
        Get direction of the trajectory at the time *abs_time*. It is the
        derivative of the trajectory at *abs_time*.
        """
        if self._times is None:
            res = np.array((0, 0, 0))
        else:
            res = normalize(np.array(interp.splev(self._get_u(abs_time),
                                                  self._tck, der=1)))

        return res * q.dimensionless

    def _get_next_time_angle(self, u_0, d_angle):
        """
        Get the next time when the trajectory will have rotated by
        more than d_angle.
        """
        x_s, y_s, z_s = interp.splev(u_0, self._tck, der=1)

        angles = vec_angle_fast((x_s, y_s, z_s), data) - d_angle
        angle_tck = interp.splrep(self._u, angles)

        u_1 = closest(interp.sproot(angle_tck), u_0)
#         if u_1 is not None:
#

    def _get_u(self, abs_time):
        """Get the spline parameter from the time *abs_time*."""
        dist = interp.splev(abs_time, self._time_tck)
        u = dist / self.length
        if u > 1:
            # If we go beyond the trajectory end, stay in it.
            u = 1

        return u

    def _get_length(self):
        """
        Get length of a parametric curve with knots, coefficients and spline
        degree tuple *tck* (most probably from scipy.interpolate.splprep). *u*
        is the parameter of the curve.
        """
        def part(u):
            der = interp.splev(u, self._tck, der=1)
            # for a 3D parametric curve the length is
            # sqrt((d_x/d_u)^2 + (d_y/d_u)^2 + (d_z/d_u)^2).
            return np.sqrt(der[0] ** 2 + der[1] ** 2 + der[2] ** 2)

        return integ.quad(part, self._u[0], self._u[-1])[0] * q.m


def closest(values, min_value):
    """Get the minimum greater value *min_value* from *values*."""
    bigger = np.where(values > min_value)[0]
    if len(bigger) == 0:
        return None
    else:
        return values[bigger[0]]


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
    return length(vector) == 1.0 * q.dimensionless


def transform_vector(trans_matrix, vector):
    """Transform *vector* by the transformation matrix *trans_matrix* with
    dimensions (4,3) width x height.
    """
    vector = vector.simplified

    return np.dot(trans_matrix, np.append(vector, 1) * vector.units)[:-1]


def overlap(interval_0, interval_1):
    """Check if intervals *interval_0* and *interval_1* overlap."""
    return interval_0[0] < interval_1[1] and interval_0[1] > interval_1[0]


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
    axis = normalize(axis)

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


def get_rotation_displacement(vec_0, vec_1, vec_length):
    """
    Get the displacement introduced by a rotation from vector *vec_0*
    to *vec_1* taking into account the rotation of a vector of length
    *vec_length*.
    """
    phi = angle(vec_0, vec_1)
    if phi > np.pi / 2 * q.rad:
        phi -= np.pi / 2 * q.rad

    return vec_length * np.tan(phi)


def interpolate_1d(x_0, y_0, size):
    """
    Interpolate function y = f(x) with *x_0*, *y_0* as control points
    and return the interpolated x_1 and y_1 arrays of *size*.
    """
    x_1 = np.linspace(x_0[0], x_0[-1], size)
    tck = interp.splrep(x_0, y_0)

    return x_1, interp.splev(x_1, tck)


def get_constant_velocity(v_0, duration):
    times = np.linspace(0 * duration.units, duration, 5)
    dist = v_0 * times

    return zip(times, dist)
