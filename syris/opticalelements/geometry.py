"""Sample geometry including affine transformations and trajectory
specification.
"""
import numpy as np
from numpy import linalg
import quantities as q
from scipy import interpolate, integrate
from syris import config as cfg
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


class GraphicalObject(object):
    """Class representing an abstract graphical object."""
    def __init__(self, trajectory, v_0=0.0, max_velocity=None,
                 accel_dist_ratio=0.0, decel_dist_ratio=0.0):
        """Create a graphical object with a *trajectory*, where:

        * *v_0* - initial velocity
        * *max_velocity* - maximum velocity to achieve during movement
        * *accel_dist_ratio* - ratio between distance for which the object
            accelerates and total distance
        * *decel_dist_ratio* ratio between distance for which the object
            decelerates and total distance
        """
        self._trajectory = trajectory
        self._center = trajectory.points[0]

        # matrix holding transformation
        self._trans_matrix = np.identity(4, dtype=cfg.NP_FLOAT)

        # movement related attributes
        self._max_velocity = cfg.NP_FLOAT(max_velocity)
        self._v_0 = v_0

        # Last position as tuple consisting of a 3D point and a vector.
        self._last_position = None

        if accel_dist_ratio is not None and decel_dist_ratio is not None and\
                accel_dist_ratio + decel_dist_ratio > 1.0:
            raise ValueError("Acceleration and deceleration" +
                             "ratios sum must be <= 1.0")
        self._accel_dist_ratio = cfg.NP_FLOAT(accel_dist_ratio)
        self._decel_dist_ratio = cfg.NP_FLOAT(decel_dist_ratio)

        # Time needed to move along the whole trajectory.
        self._total_time = None
        self._set_movement_params()

    @property
    def position(self):
        """Current position."""
        return np.dot(linalg.inv(self._trans_matrix), (0, 0, 0, 1))[:-1]

    @property
    def last_position(self):
        """Last position."""
        return self._last_position

    @property
    def center(self):
        """Center."""
        return self._center

    @center.setter
    def center(self, cen):
        self._center = cen

    @property
    def total_time(self):
        """Total time needed to move along the whole trajectory."""
        return self._total_time

    @property
    def transformation_matrix(self):
        """Current transformation matrix."""
        return self._trans_matrix

    def clear_transformation(self):
        """Clear all transformations."""
        self._trans_matrix = np.identity(4, dtype=cfg.NP_FLOAT)

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, traj):
        self._trajectory = traj
        self._center = traj.points[0]
        self._set_movement_params()

    @property
    def v_0(self):
        """Initial velocity."""
        return self._v_0

    @property
    def max_velocity(self):
        """Maximum achieved velocity."""
        return self._max_velocity

    @property
    def accel_dist_ratio(self):
        """The ratio between the acceleration distance and the whole
        trajectory.
        """
        return self._accel_dist_ratio

    @property
    def decel_dist_ratio(self):
        """The ratio between the deceleration distance and the whole
        trajectory.
        """
        return self._decel_dist_ratio

    def _set_movement_params(self):
        """Set movement parameters."""
        v_0 = self._v_0.rescale(q.m/q.s)
        v_max = self._max_velocity.rescale(v_0)
        dist = self._trajectory.length.rescale(q.m)
        accel_dist = dist*self._accel_dist_ratio.rescale(dist)
        decel_dist = dist*self._decel_dist_ratio.rescale(dist)
        const_dist = dist-accel_dist-decel_dist.rescale(dist)

        if self._max_velocity <= 0 or self._trajectory.length == 0:
            self._acceleration = 0.0*q.m/q.s**2
            self._deceleration = 0.0*q.m/q.s**2
            self._accel_end_time = 0.0*q.s
            self._decel_start_time = 0.0*q.s
            return
        if self._accel_dist_ratio <= 0.0:
            accel = 0.0*q.m/q.s**2
            accel_time = 0.0*q.s
        else:
            # dist = 1/2at^2, v = at, ar...accel_dist_ratio
            # dist = 1/2vt => t = 2s/v => accel = v/t =
            # v/(2s/v) = v^2/2s = v^2/2(ar*dist)
            accel = (v_max**2 - v_0**2)/(2*accel_dist)
            accel_time = 2.0*accel_dist/(v_0 + v_max)
        if self._decel_dist_ratio <= 0.0:
            decel = 0.0*q.m/q.s**2
            decel_time = 0.0*q.s
            # total_time = accel_time + const_time
            total_time = accel_time + const_dist/v_max
        else:
            decel = (v_max**2 - v_0**2)/(2*decel_dist)
            # t_d = t_a + t_mv (mv...max_velocity) (t = dist/v)
            # t_d = t_a + dist/v = t_a + dist*(1-ar-dr)/mv
            decel_time = accel_time + const_dist/v_max

            # Do not stop, just return to the v_0 speed.
            total_time = decel_time + 2.0*decel_dist/(v_0 + v_max)

        self._acceleration = accel
        self._deceleration = decel
        self._accel_end_time = accel_time
        self._decel_start_time = decel_time
        self._total_time = total_time


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
