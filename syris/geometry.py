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
import collections
import numpy as np
import quantities as q
from scipy import interpolate as interp
import itertools
import logging
from syris import math as smath
from syris.util import get_magnitude
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

    def __init__(self, control_points, pixel_size=None, furthest_point=None, time_dist=None,
                 velocity=None, num_points=None):
        """
        Construct a trajectory from *control_points* specified as [(x, y, z), ...]. Use *pixel_size*
        to interpolate such that the trajectory doesn't move more than 1 px between two time points.
        For this you need to specify *furthest_point* which is the furthest point from a body center
        and it is used to compute the rotational displacement component, it can be None. You specify
        the velocity profile by *time_dist*, which are tuples of (time, distance) or you can set
        constant *velocity*. *num_points* is the number of points used for interpolating the spline
        before it is used for determination of next times. If not specified, it is estimated
        automaticaly.
        """
        self._control_points = control_points.simplified
        self._velocity = velocity
        self._time_dist = time_dist
        self._pixel_size = pixel_size
        self._furthest_point = furthest_point
        self._num_points = num_points

        if time_dist is not None and velocity is not None:
            raise ValueError("time_dist and velocity can't be specified at the same time.")

        self._tck = None
        self._u = None
        self._points = control_points[0].simplified
        self._length = 0 * q.m
        self._times = self._distances = self._time_tck = None

        if pixel_size is not None:
            self.bind(pixel_size=pixel_size, furthest_point=furthest_point)

    def _interpolate(self):
        # Extract x, y, z points
        points = zip(*self._control_points)
        self._tck, self._u = interp.splprep(points, s=0)
        if self._num_points is None:
            # Reinterpolate with so many points that the trajectory doesn't move (both
            # translation and rotation taken into account) by more than pixel size between two
            # consecutive parameter values
            self._derivatives = interp.splev(self._u, self._tck, der=1)
            self._length = self._get_length() * q.m
            max_du = self.get_maximum_du()
            coeff = max(np.gradient(self.parameter)) / max_du
            # Use 4 times more points then approximated to try to fit the curve the best
            n = 4 * int(np.ceil(coeff * len(self.parameter))) if coeff > 1 else len(self.parameter)
        else:
            n = self._num_points

        LOG.debug('Using {} points for interpolation'.format(n))

        # Reinterpolate based on the updated number of points
        self._tck, self._u = reinterpolate(self._tck, self._u, n)
        self._points = np.array(interp.splev(self._u, self._tck)) * q.m
        self._derivatives = interp.splev(self._u, self._tck, der=1)
        self._length = self._get_length() * q.m

        # Velocity profile
        time_dist = None
        if self._velocity is not None:
            # Constant velocity
            duration = (self.length / self._velocity).simplified
            time_dist = get_constant_velocity(self._velocity, duration)
        elif self._time_dist is not None:
            time_dist = self._time_dist

        if time_dist is not None:
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
            self._times, self._distances = interpolate_1d(t_0, s_0, len(self.parameter))
            self._time_tck = interp.splrep(self._times, self._distances)

    def bind(self, pixel_size=None, furthest_point=None):
        """Bind the trajectory to a *pixel_size* to make sure two positions are not more than
        *pixel_size* apart between two time points. *furthest_point* is the furthest point from a
        body center used to compute rotational displacement and it can be None.
        """
        if pixel_size is not None:
            self._pixel_size = pixel_size
        self._furthest_point = furthest_point
        if self._pixel_size is None:
            raise ValueError('Pixel size must be set either here or before')
        if len(self.control_points) > 1:
            self._interpolate()

    @property
    def bound(self):
        """Return True if the trajectory is currently bound."""
        return self._pixel_size is not None

    @property
    def stationary(self):
        """Return True if the trajectory is stationary."""
        return len(self._control_points) == 1 or (self._velocity is None and
                                                  self._time_dist is None)

    @property
    def pixel_size(self):
        """Pixel size for which the trajectory is interpolated."""
        return self._pixel_size

    @property
    def furthest_point(self):
        """Furthest point for which the trajectory is interpolated."""
        return self._furthest_point

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
    def time_tck(self):
        """The tck tuple of :py:func:`scipy.interpolate.splrep` for time-distance spline."""
        return self._time_tck

    @property
    def times(self):
        """Return the time points for which the distance is defined."""
        return self._times * q.s

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

    def _evaluate(self, abs_time, der=0):
        """Evaluate trajectory's derivative *der* at time *abs_time*."""
        if abs_time < 0:
            raise ValueError("Time cannot be negative.")
        if abs_time > self.time:
            abs_time = self.time

        if self.stationary:
            result = np.array((0, 0, 0)) if der else self._control_points[0].magnitude
        else:
            if not self.bound:
                raise TrajectoryError('Trajectory not bound')
            result = interp.splev(self.get_parameter(abs_time), self._tck, der=der)

        return result

    def get_point(self, abs_time):
        """Get a point on the trajectory at the time *abs_time*."""
        return self._evaluate(abs_time, der=0) * q.m

    def get_direction(self, abs_time, norm=True):
        """
        Get direction of the trajectory at the time *abs_time*. It is the derivative of the
        trajectory at *abs_time*. If *norm* is True, the direction vector will be normalized.
        """
        direction = self._evaluate(abs_time, der=1) * q.dimensionless
        if norm:
            direction = normalize(direction)

        return direction

    def get_distances(self, u=None, u_0=None):
        """Get the distances from the trajectory beginning to every consecutive point defined by the
        parameter.
        """
        if not self.bound:
            raise TrajectoryError('Trajectory not bound')

        if u is None:
            u = self._u
        if u_0 is None:
            u_0 = 0

        points = np.array(interp.splev(u, self._tck))
        initial_point = np.array(interp.splev(u_0, self._tck))
        if isinstance(u, collections.Iterable):
            initial_point = initial_point[:, np.newaxis]
        distances = np.abs(points - initial_point)

        if self._furthest_point is not None and self._furthest_point > 0 * q.m:
            # Trajectory with no rotational displacement
            len_mag = self.length.simplified.magnitude
            der = np.array(interp.splev(u, self._tck, der=1)) / len_mag
            initial_der = np.array(interp.splev(u_0, self._tck, der=1)) / len_mag
            distances += get_rotation_displacement(der, initial_der,
                                                   self._furthest_point).simplified.magnitude

        return distances

    def get_maximum_dt(self, distance=None):
        """
        Get the maximum time difference which moves the object no more than *distance*. If distance
        is None the pixel size this trajectory is bound to is used. Consider both rotational and
        translational displacement.
        """
        if self.stationary:
            return None
        elif not self.bound:
            raise TrajectoryError('Trajectory not bound')
        if distance is None:
            distance = self._pixel_size

        # x(u) is the function of distance based on the parameter u. u(t) is the parameter as a
        # function of time. We need to find the maximum dt which doesn't move the body by more than
        # *distance*. We need to find (x(u(t)))' = dx / dt, then we set dx = *distance* and so
        # dt = *distance* / max((x(u(t)))') is the maximum allowed dt.
        # We compute the derivative (x(u(t)))' using the chain rule, (x(u(t)))' = x'(u(t)) * u'(t),
        # where x'(u(t)) means evaluating the distance spline derivative at points u(t) and u'(t) is
        # the time-distance spline derivative.
        length = self.length.simplified.magnitude
        distances = self.get_distances()

        # Create the function x(u)
        dtck = interp.splprep(distances, u=self.parameter, s=0)[0]
        # Evaluate x'(u(t)). The parameter is normalized to (0, 1), so rescale the real distances
        # by the trajectory length
        dx_ut = np.array(interp.splev(self._distances / length, dtck, der=1))
        # Evaluate u'(t) = du / dt, but the time-distance spline yields ds / dt. Since the
        # trajectory spline is arc-length parametrized, ds = du * length, so
        # du / dt = ds / (length * dt)
        du_dt = np.array(interp.splev(self._times, self._time_tck, der=1)) / length
        # (x(u(t)))' = x'(u(t)) * u'(t)
        max_f_der = np.max(np.abs(dx_ut * du_dt))
        # dt = *distance* / max|(x(u(t)))'|
        max_dt = distance.simplified.magnitude / max_f_der

        return max_dt

    def get_maximum_du(self, distance=None):
        """
        Get the maximum parameter difference which moves the object no more than *distance*. If
        distance is None the pixel size this trajectory is bound to is used. Consider both
        rotational and translational displacement.
        """
        if self.stationary:
            return 1
        elif self._tck is None:
            raise TrajectoryError('Trajectory not bound')
        if distance is None:
            distance = self._pixel_size

        # Get the distances as a combination of translation and rotation and find the du which
        # doesn't move the body by more than *distance*
        distances = self.get_distances()
        tck = interp.splprep(distances, u=self.parameter, s=0)[0]
        max_du = maximum_derivative_parameter(tck, self.parameter, distance.simplified.magnitude)

        return max_du

    def get_parameter(self, abs_time):
        """Get the spline parameter from the time *abs_time*."""
        dist = interp.splev(abs_time.simplified.magnitude, self._time_tck)
        u = dist / self.length.magnitude
        if u > 1:
            # If we go beyond the trajectory end, stay in it.
            u = 1

        return u

    def get_next_time(self, t_0):
        """Get time from *t_0* when the trajectory will have travelled more than pixel size."""
        if self.stationary:
            return np.inf * q.s
        elif not self.bound:
            raise TrajectoryError('Trajectory not bound')

        u_0 = self.get_parameter(t_0)
        distances = self.get_distances(u_0=u_0)
        # Use the same parameter for distances and trajectory so that we can directly use the
        # parameter for time calculation. This is however an approximation and if the distance
        # spline doesn't have enough points it can cause inaccuracies.
        dtck = interp.splprep(distances, u=self.parameter, s=0)[0]

        def shift_spline(sgn):
            t, c, k = dtck
            c = np.array(c) + sgn * self._pixel_size.simplified.magnitude

            return t, c, k

        t_1 = t_2 = np.inf
        lower_tck = shift_spline(1)
        upper_tck = shift_spline(-1)

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
            t_1 = self._get_next_time(t_0, smallest)
            if t_1 is None:
                t_1 = np.inf
        if greatest is not None:
            t_2 = self._get_next_time(t_0, greatest)
            if t_2 is None:
                t_2 = np.inf

        # Next time is the smallest one which is greater than t_0.
        # Get a supremum and if the result is not infinity there
        # is a time in the future for which the trajectory moves
        # the associated object more than *distance*.
        closest_time = smath.supremum(t_0.simplified.magnitude, [t_1, t_2])

        return closest_time * q.s

    def _get_next_time(self, t_0, u_0):
        """
        Get the next time when the trajectory parameter will be at position *u_0*.
        There can be multiple results but only the closest one is returned.
        """
        if self.stationary:
            return np.inf * q.s

        t, c, k = self._time_tck
        c = c - (self.length.magnitude * u_0)
        shifted_tck = t, c, k
        roots = interp.sproot(shifted_tck, mest=100)

        if len(roots) == 0:
            return np.inf * q.s

        return smath.supremum(t_0.simplified.magnitude, roots)

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


class TrajectoryError(Exception):

    """Exceptions related to trajectory."""

    pass


def closest(values, min_value):
    """Get the minimum greater value *min_value* from *values*."""
    bigger = np.where(values > min_value)[0]
    if len(bigger) == 0:
        return None
    else:
        return values[bigger[0]]


def length(vector):
    """Get length of a *vector*."""
    return np.sqrt(np.sum(vector ** 2, axis=0))


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
    """Angle between vectors *vec_0* and *vec_1*. The vectors might be 2D with 0 dimension
    specifying (x, y, z) components.
    """
    vec_0 = get_magnitude(vec_0)
    vec_1 = get_magnitude(vec_1)
    if vec_0.ndim == 2 and vec_1.ndim == 2:
        # The dot product would yield a matrix
        dot = length(vec_0 * vec_1)
    else:
        dot = np.dot(vec_0.T, vec_1)
    cross = np.cross(vec_0, vec_1, axis=0) * q.dimensionless
    lngth = length(cross)

    return np.arctan2(lngth, dot) * q.rad


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
    abs_derivatives = np.abs(np.array(interp.splev(u, tck, der=1)))

    # dx / du = f', max_distance / du = f' => du = max_distance / f'
    return max_distance / np.max(abs_derivatives)


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
    times = np.linspace(0 * duration.magnitude, duration.magnitude, 5) * duration.units
    dist = v_0 * times

    return zip(times, dist)


def get_rotation_displacement(d_0, d_1, length):
    """
    Return the displacement of a sphere with radius *length* caused by rotation around vectors *d_0*
    and *d_1*. The displacement is returned for every axis (x, y, z).
    """
    # return np.abs(length * (normalize(d_1) - normalize(d_0)))
    def get_displacement(axis, rot_axis, phi):
        """
        We first determine the local maximum rotation displacement in the plane parallel to the
        rotation plane (perpendicular to the rotation axis *rot_axis*, i.e. we work in local
        coordinates). We obtain the maximum when *phi* / 2 is aligned with an axis perpendicular to
        the *axis* we are interested in. We compute half of the local principal displacement (it is
        the displacement in one of the local principal axes direction) as the sine of the angle
        multiplied by the *length*, then we multiply by 2 to get the other half of the displacement.
        At the end we need to take into account that we have thus far worked with local principal
        axes (aligned with respect to *rot_axis*) and need to transform the displacement to the
        global coordinate system. This is obtained by taking the sine of the rotation axis and the
        principal axis angle and multiplying with the local displacement.
        """
        axis_sin = np.sin(angle(axis, rot_axis).rescale(q.rad).magnitude)
        # print axis
        # print rot_axis
        # print axis_sin, 'axis sin'
        # print phi, 'phi'

        return np.abs(axis_sin * 2 * get_magnitude(length) * np.sin(phi / 2))

    phi = angle(d_0, d_1)
    rot_axis = np.cross(d_0, d_1, axis=0) * q.dimensionless

    dx = get_displacement(AXES[X], rot_axis, phi)
    dy = get_displacement(AXES[Y], rot_axis, phi)
    dz = get_displacement(AXES[Z], rot_axis, phi)

    return (dx, dy, dz) * q.m


def make_points(x_ends, y_ends, z_ends):
    """Make 3D points out of minima and maxima given by *x_ends*, *y_ends* and *z_ends*."""
    return np.array(list(itertools.product(x_ends, y_ends, z_ends))) * x_ends.units
