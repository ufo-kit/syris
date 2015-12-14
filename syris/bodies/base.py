"""A base module for pysical bodies, which are optical elements having spatial extent."""
import numpy as np
import quantities as q
import syris.config as cfg
import syris.geometry as geom
from quantities.quantity import Quantity
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength, transfer
from syris.util import make_tuple


class Body(OpticalElement):

    """An abstract body class with a *material*, which is a :class:`syris.materials.Material`
    instance.
    """

    def __init__(self, material=None):
        self.material = material

    def project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        """Project thickness at time *t* to the image plane of size *shape* which is either 1D and
        is extended to (n, n) or is 2D as HxW. *pixel_size* is the point size, also either 1D or
        2D. *queue* is an OpenCL command queue, *out* is the pyopencl array used for result.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)
        if queue is None:
            queue = cfg.OPENCL.queue

        return self._project(shape, pixel_size, t=t, queue=queue, out=None)

    def _project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        """Projection function implementation. *shape* and *pixel_size* are 2D."""
        raise NotImplementedError

    def _transfer(self, shape, pixel_size, energy, t=0 * q.s, queue=None, out=None):
        """Transfer function implementation based on a refractive index."""
        ri = self.material.get_refractive_index(energy)
        lam = energy_to_wavelength(energy)

        return transfer(self.project(shape, pixel_size, t=t), ri, lam, queue=queue, out=out)


class MovableBody(Body):

    """Class representing a movable body."""

    def __init__(self, trajectory, material=None, orientation=geom.Y_AX):
        """Create a body with a :class:`~syris.geometry.Trajectory` and *orientation*,
        which is an (x, y, z) vector specifying body's "up" vector.
        """
        super(MovableBody, self).__init__(material)
        self._trajectory = trajectory
        self._orientation = geom.normalize(orientation)
        self._center = trajectory.control_points[0].simplified

        # Matrix holding transformation.
        self.transform_matrix = np.identity(4, dtype=cfg.PRECISION.np_float)
        # Maximum body enlargement in any direction.
        self._scale_factor = np.ones(3)

        # Last position as tuple consisting of a 3D point and a vector giving
        # the body orientation.
        self._last_position = None

        # tck of the spline following the distances from the beginning of the
        # body's trajectory
        self._distance_tck = None

    def project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        """Project thickness at time *t* (if it is None no transformation is applied) to the image
        plane of size *shape* which is either 1D and is extended to (n, n) or is 2D as HxW.
        *pixel_size* is the point size, also either 1D or 2D. *queue* is an OpenCL command queue,
        *out* is the pyopencl array used for result.
        """
        if t is not None:
            self.move(t)

        return super(MovableBody, self).project(shape, pixel_size, t=t, queue=queue, out=None)

    @property
    def furthest_point(self):
        """
        The furthest point from body's center with respect to the
        scaling factor of the body.
        """
        raise NotImplementedError

    @property
    def bounding_box(self):
        """Bounding box defining the extent of the body."""
        raise NotImplementedError

    @property
    def position(self):
        """Current position."""
        return self.transform_matrix[:3, -1] * q.m

    @property
    def last_position(self):
        """Last position."""
        return self._last_position

    @property
    def center(self):
        """Center."""
        return self._center

    @center.setter
    def center(self, center):
        self._center = center.simplified

    @property
    def orientation(self):
        return self._orientation

    def clear_transformation(self):
        """Clear all transformations."""
        self.transform_matrix = np.identity(4, dtype=cfg.PRECISION.np_float)
        self._scale_factor = np.ones(3)

    @property
    def trajectory(self):
        return self._trajectory

    def get_rescaled_transform_matrix(self, units, coeff=1):
        """The last column of the transformation matrix holds displacement
        information has SI units, convert those to the *units* specified,
        apply coefficient *coeff* and return a copy of the matrix.
        """
        trans_mat = np.copy(self.transform_matrix)
        for i in range(3):
            trans_mat[i, 3] = coeff * Quantity(trans_mat[i, 3] * q.m).rescale(units)

        return trans_mat

    def apply_transformation(self, trans_matrix):
        """Apply transformation given by the transformation matrix
        *trans_matrix* on the current transformation matrix.
        """
        self.transform_matrix = np.dot(trans_matrix, self.transform_matrix)

    def get_next_time(self, t_0, distance):
        """
        Get time from *t_0* when the body will have travelled more than *distance*.
        """
        return self.trajectory.get_next_time_from_distance(t_0, distance, self.furthest_point)

    def get_maximum_dt(self, distance):
        """Get the maximum delta time for which the body will not
        move more than *distance* between any two time points.
        """
        return self.trajectory.get_maximum_dt(self.furthest_point, distance)

    def moved(self, t_0, t_1, distance):
        """
        Return True if the body moves more than *distance*
        in time interval *t_0*, *t_1*.
        """
        p_0 = self.trajectory.get_point(t_0)
        p_1 = self.trajectory.get_point(t_1)
        trans_displacement = np.abs(p_1 - p_0)

        d_0 = self.trajectory.get_direction(t_0, norm=False)
        d_1 = self.trajectory.get_direction(t_1, norm=False)
        rot_displacement = geom.get_rotation_displacement(d_0, d_1, self.furthest_point)
        total_displacement = trans_displacement + rot_displacement

        return max(total_displacement) > distance

    def move(self, abs_time):
        """Move to a position of the body in time *abs_time*."""
        self.clear_transformation()
        abs_time = abs_time.simplified
        p_0 = self.trajectory.get_point(abs_time).simplified
        vec = self.trajectory.get_direction(abs_time)

        # First translate to the point at time abs_time
        self.translate(p_0)

        # Then rotate about rotation axis given by trajectory direction
        # and body orientation.
        rot_ax = geom.normalize(np.cross(self._orientation, vec))
        angle = geom.angle(self._orientation, vec)
        self.rotate(angle, rot_ax)

    def translate(self, vec):
        """Translate the body by a vector *vec*."""
        self.transform_matrix = np.dot(self.transform_matrix, geom.translate(vec))

    def rotate(self, angle, axis, total_start=None):
        """Rotate the body by *angle* around vector *axis*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        self.transform_matrix = np.dot(self.transform_matrix, geom.rotate(angle, axis, total_start))


class CompositeBody(MovableBody):

    """Class representing a body consisting of more sub-bodies.  A composite body can be thought of
    as a tree structure with children representing another bodies.
    """

    def __init__(self, trajectory, orientation=geom.Y_AX, bodies=None):
        """*bodies* is a list of :py:class:`.MovableBody`."""
        super(CompositeBody, self).__init__(trajectory, orientation=orientation)
        if bodies is None:
            bodies = []
        self._bodies = []

        # Do not just assign bodies but let them all go through
        # add method, so the list is checked for correct input.
        for body in bodies:
            self.add(body)

        self._furthest_point = None
        self._dt = None
        self._saved_matrices = {}

    @property
    def bodies(self):
        """All bodies which are inside this composite body."""
        return tuple(self._bodies)

    @property
    def all_bodies(self):
        """All bodies inside this body recursively."""
        return self._all_bodies(False)

    def _all_bodies(self, primitive):
        res = set() if primitive else set([self])

        for body in self:
            if body.__class__ == CompositeBody:
                res.update(body._all_bodies(primitive))
            else:
                res.add(body)

        return tuple(res)

    @property
    def primitive_bodies(self):
        return self._all_bodies(True)

    @property
    def direct_primitive_bodies(self):
        """Return primitive bodies on the level immediately after this body's level."""
        primitive = []

        for body in self._bodies:
            if body.__class__ != CompositeBody:
                primitive.append(body)

        return primitive

    @property
    def time(self):
        """The total trajectory time of the body and all its subbodies."""
        return max([body.trajectory.time for body in self._all_bodies(False)])

    @property
    def furthest_point(self):
        """Furthest point is the greatest achievable distance to some primitive
        bodies plus the furthest point of the primitive body. This way we can
        put an upper bound on the distance travelled by any primitive body.
        """
        if self._furthest_point is None:
            self._determine_furthest_point()
        return self._furthest_point

    def _determine_furthest_point(self):
        """Calculate the furthest point based on all primitive bodies."""
        furthest = None

        for body in self.primitive_bodies:
            traj_dist = np.sqrt(np.sum(body.trajectory.points ** 2, axis=0))
            if len(body.trajectory.points.shape) == 2:
                # For non-stationary trajectory we take the maximum
                traj_dist = max(traj_dist)
            dist = traj_dist + body.furthest_point
            if furthest is None or dist > furthest:
                furthest = dist

        self._furthest_point = furthest

    def __len__(self):
        return len(self._bodies)

    def __getitem__(self, key):
        return self._bodies[key]

    def __iter__(self):
        return self.bodies.__iter__()

    def add(self, body):
        """Add a body *body*."""
        if body is self:
            raise ValueError("Cannot add self")
        if body in self._all_bodies(False):
            raise ValueError("Body {0} already contained".format(body))

        self._bodies.append(body)

    def remove(self, body):
        """Remove body *body*."""
        self._bodies.remove(body)

    def remove_all(self):
        """Remove all sub-bodies."""
        self._bodies = []

    def clear_transformation(self):
        """Clear all transformations."""
        MovableBody.clear_transformation(self)
        for body in self:
            body.clear_transformation()

    @property
    def bounding_box(self):
        """Get bounding box around all the bodies inside."""
        b_box = None
        for i in range(len(self)):
            if b_box is None:
                b_box = self[i].bounding_box
            else:
                b_box.merge(self[i].bounding_box)

        return b_box

    def translate(self, vec):
        """Translate all sub-bodies by a vector *vec*."""
        MovableBody.translate(self, vec)
        for body in self:
            body.translate(vec)

    def rotate(self, angle, vec, total_start=None):
        """Rotate all sub-bodies by *angle* around vector *vec*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        MovableBody.rotate(self, angle, vec, total_start)
        for body in self:
            body.rotate(angle, vec, total_start)

    def move(self, abs_time):
        """Move to a position of the body in time *abs_time*."""
        # Move the whole body.
        MovableBody.move(self, abs_time)
        for body in self:
            # Then move its sub-bodies.
            body.move(abs_time)

    def save_transformation_matrices(self):
        """Save transformation matrices of all bodies and return
        them in a dictionary {body: transform_matrix}.
        """
        for body in self.all_bodies:
            self._saved_matrices[body] = np.copy(body.transform_matrix)

    def restore_transformation_matrices(self):
        """Restore transformation matrices of all bodies."""
        for body, matrix in self._saved_matrices.iteritems():
            body.transform_matrix = matrix

        self._saved_matrices = {}

    def get_next_time(self, t_0, distance):
        """
        Get next time at which the body will have traveled
        *distance*, the starting time is *t_0*.
        """
        # First deterimne the real distance which is smaller by the
        # given one because the combination of body movements might
        # exceed the distance if the motion of bodies adds up
        # constructively.
        if self._dt is None:
            # Initialize
            dts = [body.get_maximum_dt(distance / len(self.all_bodies))
                   for body in self.all_bodies if body.trajectory.length > 0 * q.m]
            self._dt = np.min(dts) * q.s

        for current_time in np.arange(t_0, self.time + self._dt, self._dt) * q.s:
            if self.moved(t_0, current_time, distance):
                return current_time

        return np.inf * q.s

    def moved(self, t_0, t_1, distance):
        """Return True if the body moves between time *t_0* and *t_1* more than *distance*. We need
        to check all subbodies.  Moreover, simple trajectory distance between points at t_0 and t_1
        will not work because when the composite body moves more than one pixel, but the primitive
        body moves the exact opposite it results in no movement. We need to check also the composite
        body movement because it may cause some subbodies to rotate.
        """
        def move_and_save(abs_time):
            """Move primitive bodies to time *abs_time* and return
            their positions.
            """
            primitive = self.primitive_bodies
            self.clear_transformation()
            self.move(abs_time)

            positions = np.zeros((len(primitive), 3))
            for i in range(len(primitive)):
                positions[i] = primitive[i].position.simplified.magnitude

            return positions

        self.save_transformation_matrices()
        orig_positions = move_and_save(t_0)
        positions = move_and_save(t_1)
        self.restore_transformation_matrices()

        return np.max(np.abs(positions - orig_positions)) > distance.simplified.magnitude

    def __repr__(self):
        return "CompositeBody{0}".format(self.bodies)

    def __str__(self):
        return repr(self)
