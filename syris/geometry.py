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
from __future__ import absolute_import
import math
import numpy as np
import quantities as q
from scipy import interpolate as interp
import itertools
import logging
from syris import math as smath
from quantities.quantity import Quantity


X = 0
Y = 1
Z = 2
X_AX = np.array([1, 0, 0]) * q.dimensionless
Y_AX = np.array([0, 1, 0]) * q.dimensionless
Z_AX = np.array([0, 0, 1]) * q.dimensionless

AXES = {X: X_AX, Y: Y_AX, Z: Z_AX}

LOG = logging.getLogger(__name__)


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

        self._points = make_points([x_min, x_max] * q.m, [y_min, y_max] * q.m,
                                   [z_min, z_max] * q.m)

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
            self._points = control_points[0].simplified
            self._length = 0 * q.m
            self._times = self._distances = self._time_tck = None
        else:
            # Extract x, y, z points
            points = zip(*control_points.simplified)

            # Positions
            tck, u = interp.splprep(points, s=0)
            self._tck, self._u = reinterpolate(tck, u, 1000)
            self._points = np.array(interp.splev(self._u, self._tck)) * q.m

            # Derivatives of the spline (x_d, y_d, z_d tuple)
            self._derivatives = interp.splev(self._u, self._tck, der=1)
            self._length = self._get_length() * q.m

            # Velocity profile
            if velocity is not None:
                # Constant velocity
                time_dist = get_constant_velocity(velocity, self.length / velocity)

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
            self._times, self._distances = interpolate_1d(t_0, s_0, 1000)
            self._time_tck = interp.splrep(self._times, self._distances)

    @property
    def control_points(self):
        """Control points used by the trajectory."""
        return tuple(self._control_points) * q.m

    @property
    def points(self):
        """Return interpolated points."""
        return self._points

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

    @property
    def parameter(self):
        return self._u

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
                result = interp.splev(self.get_parameter(abs_time), self._tck) * q.m

        return result

    def get_direction(self, abs_time, norm=True):
        """
        Get direction of the trajectory at the time *abs_time*. It is the
        derivative of the trajectory at *abs_time*. If *norm* is True, the
        direction vector will be normalized.
        """
        if self._times is None:
            res = np.array((0, 0, 0))
        else:
            res = np.array(interp.splev(self.get_parameter(abs_time),
                                        self._tck, der=1))
            if norm:
                res = normalize(res)

        return res * q.dimensionless

    def get_distances(self, distance=None):
        """Get the distances from the trajectory beginning to every consecutive point defined by the
        parameter. Take into account translation and rotation of an object with the furthest point
        from its center given by *distance*. If *distance* is None rotational component is not taken
        into account.
        """
        points = np.array(interp.splev(self._u, self._tck))
        initial_point = np.array(interp.splev(0, self._tck))
        distances = points - initial_point[:, np.newaxis]

        if distance is not None:
            angles = np.arctan(np.array(interp.splev(self._u, self._tck, der=1)))
            initial_angle = np.arctan(np.array(interp.splev(0, self._tck, der=1)))
            angle_diff = angles - initial_angle[:, np.newaxis]
            # sin(phi) = dx / distance => dx = distance * sin(phi)
            distances += distance.simplified.magnitude * np.sin(angle_diff)

        return distances

    def get_maximum_dt(self, furthest_point, distance):
        """
        Get the maximum time difference which moves the object no more than
        *distance*. Consider both rotational and translational displacement,
        when by the rotational one the *furhtest_point* is the most distant
        point from the center of the rotation.
        """
        if self._tck is None:
            # Stationary trajectory
            return None

        ds = self.get_maximum_du(furthest_point, distance) * self.length.simplified.magnitude
        # f' = ds / dt, f' = const. => dt = ds / f'
        derivative = max(np.abs(interp.splev(self._times, self._time_tck, der=1)))
        return ds / (derivative * self._times[-1])

    def get_maximum_du(self, furthest_point, distance):
        """
        Get the maximum parameter difference which moves the object no more than
        *distance*. Consider both rotational and translational displacement,
        when by the rotational one the *furhtest_point* is the most distant
        point from the center of the rotation.
        """
        return min(self._get_translation_du(distance),
                   self._get_rotation_du(furthest_point, distance))

    def _get_translation_du(self, distance):
        """Get the maximum du in order not to move more than *distance*."""
        return maximum_derivative_parameter(self._tck, self._u, distance.simplified.magnitude)

    def _get_rotation_du(self, furthest_point, distance):
        """
        Get the maximum du in order not to move angularly more than *distance*.
        *furthest_point* is the furthest point from the middle of the rotation
        which must not move more than *distance*.
        """
        furthest_point = furthest_point.simplified.magnitude
        distance = distance.simplified.magnitude

        du = np.gradient(self._u)
        max_sin = distance / furthest_point
        derivatives = interp.splev(self._u, self._tck, der=1)
        sines = np.array([np.abs(np.sin(np.gradient(np.arctan(der)))) for der in derivatives])
        dim = np.argmax(sines.max(axis=1))
        index = np.argmax(sines[dim])
        k = max_sin / sines[dim, index]

        return k * du[index]

    def get_parameter(self, abs_time):
        """Get the spline parameter from the time *abs_time*."""
        dist = interp.splev(abs_time.simplified.magnitude, self._time_tck)
        u = dist / self.length.magnitude
        if u > 1:
            # If we go beyond the trajectory end, stay in it.
            u = 1

        return u

    def get_next_time(self, t_0, u_0):
        """
        Get the next time when the trajectory parameter will be at position *u_0*.
        There can be multiple results but only the closest one is returned.
        """
        t, c, k = self._time_tck
        c = c - (self.length.magnitude * u_0)
        shifted_tck = t, c, k
        roots = interp.sproot(shifted_tck, mest=100)

        if len(roots) == 0:
            return np.inf * q.s

        return smath.supremum(t_0.simplified.magnitude, roots)

    def get_next_time_from_distance(self, t_0, distance, furthest_point=None):
        """
        Get time from *t_0* when the trajectory will have travelled more than *distance*, which is
        typically the pixel size. *furthest_point* is taken into account for rotational
        displacement.
        """
        if t_0 is None:
            return None

        points = self.get_distances(furthest_point)
        # Use the same parameter so the derivatives are equal
        distance_tck = interp.splprep(points, u=self.parameter, s=0)[0]

        def shift_spline(u_0, sgn):
            t, c, k = distance_tck
            initial_point = np.array(interp.splev(u_0, distance_tck))[:, np.newaxis]
            c = np.array(c) - initial_point + sgn * distance.simplified.magnitude

            return t, c, k

        t_1 = t_2 = np.inf
        u_0 = self.get_parameter(t_0)
        lower_tck = shift_spline(u_0, 1)
        upper_tck = shift_spline(u_0, -1)

        # Get the +/- distance roots (we can traverse the trajectory backwards)
        lower = interp.sproot(lower_tck)
        upper = interp.sproot(upper_tck)
        # Combine lower and upper into one list of roots for every dimension
        roots = [np.concatenate((lower[i], upper[i])) for i in range(3)]
        # Mix all dimensions, they are not necessary for obtaining the minimum
        # parameter difference
        roots = np.concatenate(roots)
        # Filter roots to get only the infimum and supremum based on u_0
        smallest = smath.infimum(u_0, roots)
        greatest = smath.supremum(u_0, roots)

        # Get next time for both directions
        if smallest is not None:
            t_1 = self.get_next_time(t_0, smallest)
            if t_1 is None:
                t_1 = np.inf
        if greatest is not None:
            t_2 = self.get_next_time(t_0, greatest)
            if t_2 is None:
                t_2 = np.inf

        # Next time is the smallest one which is greater than t_0.
        # Get a supremum and if the result is not infinity there
        # is a time in the future for which the trajectory moves
        # the associated object more than *distance*.
        closest_time = smath.supremum(t_0.simplified.magnitude, [t_1, t_2])

        return closest_time * q.s

    def _get_length(self, param_start=0, param_end=1):
        """
        Get spline length on the spline parameter interval *param_start*,
        *param_end*.
        """
        def part(x_d, y_d, z_d):
            # for a 3D parametric curve the length is
            # sqrt((d_x/d_u)^2 + (d_y/d_u)^2 + (d_z/d_u)^2).
            return np.sqrt(x_d ** 2 + y_d ** 2 + z_d ** 2)

        der_tck = interp.splrep(self._u, part(*self._derivatives))

        return interp.splint(param_start, param_end, der_tck)


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
    """Translate the object by a vector *vec*. The vector is _always_ transformed into meters."""
    vec = vec.simplified
    trans_matrix = np.identity(4)

    trans_matrix[0][3] = vec[0]
    trans_matrix[1][3] = vec[1]
    trans_matrix[2][3] = vec[2]

    return trans_matrix


def rotate(phi, axis, total_start=None):
    """Rotate the object by *phi* around vector *axis*, where *total_start* is the center of
    rotation point which results in transformation TRT. The angle is _always_ rescaled to radians.
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
    rot_matrix[0][0] = cos + v_x ** 2 * (1 - cos)
    rot_matrix[0][1] = v_x * v_y * (1 - cos) - v_z * sin
    rot_matrix[0][2] = v_x * v_z * (1 - cos) + v_y * sin
    rot_matrix[1][0] = v_x * v_y * (1 - cos) + v_z * sin
    rot_matrix[1][1] = cos + v_y ** 2 * (1 - cos)
    rot_matrix[1][2] = v_y * v_z * (1 - cos) - v_x * sin
    rot_matrix[2][0] = v_z * v_x * (1 - cos) - v_y * sin
    rot_matrix[2][1] = v_z * v_y * (1 - cos) + v_x * sin
    rot_matrix[2][2] = cos + v_z ** 2 * (1 - cos)

    if total_start is not None:
        t_2 = translate(-total_start)
        rot_matrix = np.dot(np.dot(t_2, rot_matrix), t_1)

    return rot_matrix

def scale(scale_vec):
    """Scale the object by scaling coefficients (kx, ky, kz) given by *sc_vec*."""
    if (scale_vec[0] <= 0 or scale_vec[1] <= 0 or scale_vec[2] <= 0):
        raise ValueError("All components of the scaling " +
                         "must be greater than 0")
    trans_matrix = np.identity(4)

    trans_matrix[0][0] = scale_vec[0]
    trans_matrix[1][1] = scale_vec[1]
    trans_matrix[2][2] = scale_vec[2]

    return trans_matrix


def angle(vec_0, vec_1):
    """Angle between vectors *vec_0* and *vec_1*."""
    vec_1 = vec_1.rescale(vec_0.units)

    return math.atan2(length(np.cross(vec_0, vec_1) * q.dimensionless),
                      np.dot(vec_0, vec_1)) * q.rad


def interpolate_1d(x_0, y_0, size):
    """
    Interpolate function y = f(x) with *x_0*, *y_0* as control points
    and return the interpolated x_1 and y_1 arrays of *size*.
    """
    x_1 = np.linspace(x_0[0], x_0[-1], size)
    tck = interp.splrep(x_0, y_0)

    return x_1, interp.splev(x_1, tck)


def reinterpolate(tck, u, n):
    """
    Arc length reinterpolation of a spline given by *tck* and parameter *u*
    to have *n* data points.
    """
    # First reinterpolate the arc length parameter
    u_tck = interp.splrep(np.linspace(0, 1, len(u)), u, s=0)
    new_u = interp.splev(np.linspace(0, 1, n), u_tck)

    x, y, z = interp.splev(new_u, tck)
    return interp.splprep((x, y, z), s=0)


def maximum_derivative_parameter(tck, u, max_distance):
    """Get the maximum possible du, for which holds that dx < *max_distance*."""
    derivatives = interp.splev(u, tck, der=1)
    du = np.gradient(u)
    distances = np.array([np.abs(derivative * du) for derivative in derivatives])
    max_indices = np.argmax(distances, axis=1)
    max_derivative = max([np.abs(derivatives[i][max_indices[i]]) for i in range(3)])
    # The desired du is given by max_distance / f'
    return max_distance / max_derivative


def derivative_fit(tck, u, max_distance):
    """
    Reinterpolate curve in a way that all the f' * du are smaller than *max_distance*.
    The original spline is given by *tck* and parameter *u*.
    """
    n = 1 / maximum_derivative_parameter(tck, u, max_distance)
    if n > len(u):
        tck, u = reinterpolate(tck, u, n)

    return tck, u


def get_constant_velocity(v_0, duration):
    times = np.linspace(0 * duration.units, duration, 5)
    dist = v_0 * times

    return zip(times, dist)


def get_rotation_displacement(d_0, d_1, length):
    """
    Return the displacement cause by rotation of a vector of some
    *length*. The *d_0* and *d_1* are the tangents at different
    points.
    """
    return np.abs(length * np.sin(np.arctan(d_1) - np.arctan(d_0)))


def make_points(x_ends, y_ends, z_ends):
    """Make 3D points out of minima and maxima given by *x_ends*, *y_ends* and *z_ends*."""
    return np.array(list(itertools.product(x_ends, y_ends, z_ends))) * x_ends.units
