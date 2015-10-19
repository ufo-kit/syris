"""
Graphical objects are used to construct an optical element's geometry.
"""
import itertools
import logging
import numpy as np
from numpy import linalg
from scipy import interpolate as interp
import quantities as q
from syris import config as cfg
from syris.physics import transfer, energy_to_wavelength
from syris.geometry import BoundingBox, get_rotation_displacement
import syris.geometry as geom
from syris import math as smath
import struct
from quantities.quantity import Quantity

LOG = logging.getLogger(__name__)


class GraphicalObject(object):

    """An abstract graphical object class."""

    def project(self, t=None):
        """Project thickness at time *t*."""
        raise NotImplementedError


class StaticGraphicalObject(GraphicalObject):

    """A static graphical object defined by its projected *thickness*."""

    def __init__(self, thickness):
        super(StaticGraphicalObject, self).__init__()
        self.thickness = thickness

    def project(self, t=None):
        """Project thickness."""
        return self.thickness


class MovableGraphicalObject(GraphicalObject):

    """Class representing an abstract graphical object."""

    def __init__(self, trajectory, orientation=geom.Y_AX):
        """Create a graphical object with a :class:`~syris.geometry.Trajectory` and *orientation*,
        which is an (x, y, z) vector specifying object's "up" vector.
        """
        super(MovableGraphicalObject, self).__init__()
        self._trajectory = trajectory
        self._orientation = geom.normalize(orientation)
        self._center = trajectory.control_points[0].simplified

        # Matrix holding transformation.
        self.transform_matrix = np.identity(4, dtype=cfg.PRECISION.np_float)
        # Maximum object enlargement in any direction.
        self._scale_factor = np.ones(3)

        # Last position as tuple consisting of a 3D point and a vector giving
        # the object orientation.
        self._last_position = None

        # tck of the spline following the distances from the beginning of the
        # object's trajectory
        self._distance_tck = None

    @property
    def furthest_point(self):
        """
        The furthest point from object's center with respect to the
        scaling factor of the object.
        """
        raise NotImplementedError

    @property
    def position(self):
        """Current position."""
        return linalg.inv(self.transform_matrix)[:3, -1] * q.m

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
            trans_mat[i, 3] = coeff * Quantity(trans_mat[i, 3] * q.m).\
                rescale(units)

        return trans_mat

    def apply_transformation(self, trans_matrix):
        """Apply transformation given by the transformation matrix
        *trans_matrix* on the current transformation matrix.
        """
        self.transform_matrix = np.dot(trans_matrix, self.transform_matrix)

    def get_next_time(self, t_0, distance):
        """
        Get time from *t_0* when the object will have travelled more than
        the *distance*.
        """
        if t_0 is None:
            return None

        if self._distance_tck is None:
            points = self.trajectory.get_distances(distance.simplified.magnitude)
            # Use the same parameter so the derivatives are equal
            self._distance_tck = interp.splprep(points, u=self.trajectory.parameter, s=0)[0]

        def shift_spline(u_0, sgn):
            t, c, k = self._distance_tck
            initial_point = np.array(interp.splev(u_0, self._distance_tck))[:, np.newaxis]
            c = np.array(c) - initial_point + sgn * distance.simplified.magnitude

            return t, c, k

        t_1 = t_2 = np.inf
        u_0 = self.trajectory.get_parameter(t_0)
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
            t_1 = self.trajectory.get_next_time(t_0, smallest)
            if t_1 is None:
                t_1 = np.inf
        if greatest is not None:
            t_2 = self.trajectory.get_next_time(t_0, greatest)
            if t_2 is None:
                t_2 = np.inf

        # Next time is the smallest one which is greater than t_0.
        # Get a supremum and if the result is not infinity there
        # is a time in the future for which the trajectory moves
        # the associated object more than *distance*.
        closest_time = smath.supremum(t_0.simplified.magnitude, [t_1, t_2])
        if closest_time == np.inf:
            return None

        return closest_time * q.s

    def get_maximum_dt(self, distance):
        """Get the maximum delta time for which the object will not
        move more than *distance* between any two time points.
        """
        return self.trajectory.get_maximum_dt(self.furthest_point, distance)

    def moved(self, t_0, t_1, distance):
        """
        Return True if the object moves more than *distance*
        in time interval *t_0*, *t_1*.
        """
        p_0 = self.trajectory.get_point(t_0)
        p_1 = self.trajectory.get_point(t_1)
        trans_displacement = np.abs(p_1 - p_0)

        d_0 = self.trajectory.get_direction(t_0, norm=False)
        d_1 = self.trajectory.get_direction(t_1, norm=False)
        rot_displacement = get_rotation_displacement(d_0, d_1, self.furthest_point)
        total_displacement = trans_displacement + rot_displacement

        return max(total_displacement) > distance

    def move(self, abs_time):
        """Move to a position of the object in time *abs_time*."""
        abs_time = abs_time.simplified
        p_0 = self.trajectory.get_point(abs_time).simplified
        vec = self.trajectory.get_direction(abs_time)

        # First translate to the point at time abs_time
        self.translate(p_0)

        # Then rotate about rotation axis given by trajectory direction
        # and object orientation.
        rot_ax = geom.normalize(np.cross(self._orientation, vec))
        angle = geom.angle(self._orientation, vec)
        self.rotate(angle, rot_ax)

    def translate(self, vec):
        """Translate the object by a vector *vec*."""
        self.transform_matrix = np.dot(
            geom.translate(vec), self.transform_matrix)

    def rotate(self, angle, axis, total_start=None):
        """Rotate the object by *angle* around vector *axis*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        self.transform_matrix = np.dot(geom.rotate(angle, axis, total_start),
                                       self.transform_matrix)


class MetaBall(MovableGraphicalObject):

    """"Metaball graphical object. Metaballs are smooth blobs formed
    by summing density functions representing particular objects.
    """

    def __init__(self, trajectory, radius, orientation=geom.Y_AX):
        """Create a metaobject with *radius*."""
        super(MetaBall, self).__init__(trajectory, orientation=orientation)
        if radius <= 0:
            raise ValueError("Radius must be greater than zero.")

        self._radius = radius.simplified

    @property
    def radius(self):
        return self._radius

    @property
    def furthest_point(self):
        """
        Furthest point is twice the radius because of the influence
        region of the metaball.
        """
        return 2 * self.radius * max(self._scale_factor)

    @property
    def bounding_box(self):
        """Bounding box of the metaball."""
        radius = self.radius.simplified.magnitude

        base = -2 * radius, 2 * radius
        points = list(itertools.product(base, base, base)) * q.m

        # Transform by the current transformation matrix.
        transformed = np.array([geom.transform_vector(
            linalg.inv(self.transform_matrix), points[i])
            for i in range(len(points))])

        return BoundingBox(transformed * q.m)

    def get_transform_const(self):
        """
        Precompute the transformation constant which does not change for
        x,y position.
        """
        a_x = self.transform_matrix[0][2]
        a_y = self.transform_matrix[1][2]
        a_z = self.transform_matrix[2][2]
        return a_x ** 2 + a_y ** 2 + a_z ** 2

    def pack(self, units):
        """Pack the object into a structure suitable for OpenCL kernels.
        Rescale the object using *units* first.
        """
        fmt = get_format_string("ffff")

        return struct.pack(fmt, self.position[0].rescale(units).magnitude,
                           self.position[1].rescale(units).magnitude,
                           self.position[2].rescale(units).magnitude,
                           self.radius.rescale(units).magnitude)

    def __repr__(self):
        return "MetaBall({0})".format(self.radius)

    def __str__(self):
        return repr(self)


class CompositeObject(MovableGraphicalObject):

    """
    Class representing an object consisting of more sub-objects.
    A composite object can be thought of as a tree structure with
    children representing another graphical objects.
    """

    def __init__(self, trajectory, orientation=geom.Y_AX, gr_objects=None):
        """*gr_objects* is a list of :py:class:`GraphicalObject`."""
        super(CompositeObject, self).__init__(trajectory, orientation=orientation)
        if gr_objects is None:
            gr_objects = []
        self._objects = []

        # Do not just assign gr_objects but let them all go through
        # add method, so the list is checked for correct input.
        for obj in gr_objects:
            self.add(obj)

        self._furthest_point = None
        self._dt = None
        self._saved_matrices = {}

    @property
    def objects(self):
        """All objects which are inside this composite object."""
        return tuple(self._objects)

    @property
    def all_objects(self):
        """All objects inside this object recursively."""
        return self._all_objects(False)

    def _all_objects(self, primitive):
        res = set() if primitive else set([self])

        for obj in self:
            if obj.__class__ == CompositeObject:
                res.update(obj._all_objects(primitive))
            else:
                res.add(obj)

        return tuple(res)

    @property
    def primitive_objects(self):
        return self._all_objects(True)

    @property
    def direct_primitive_objects(self):
        """
        Return primitive objects on the level immediately after
        this object's level.
        """
        primitive = []

        for obj in self._objects:
            if obj.__class__ != CompositeObject:
                primitive.append(obj)

        return primitive

    @property
    def time(self):
        """The total trajectory time of the object and all its subobjects."""
        return max([obj.trajectory.time for obj in self._all_objects(False)])

    @property
    def furthest_point(self):
        """Furthest point is the greatest achievable distance to some primitive
        object plus the furthest point of the primitive object. This way we can
        put an upper bound on the distance travelled by any primitive object.
        """
        if self._furthest_point is None:
            self._determine_furthest_point()
        return self._furthest_point

    def _determine_furthest_point(self):
        """Calculate the furthest point based on all primitive objects."""
        furthest = None

        for obj in self.primitive_objects:
            traj_dist = np.sqrt(np.sum(obj.trajectory.points ** 2, axis=0))
            if len(obj.trajectory.points.shape) == 2:
                # For non-stationary trajectory we take the maximum
                traj_dist = max(traj_dist)
            dist = traj_dist + obj.furthest_point
            if furthest is None or dist > furthest:
                furthest = dist

        self._furthest_point = furthest

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return self._objects.__iter__()

    def add(self, obj):
        """Add a graphical object *obj*."""
        if obj is self:
            raise ValueError("Cannot add self")
        if obj in self._all_objects(False):
            raise ValueError("Object {0} already contained".format(obj))

        self._objects.append(obj)

    def remove(self, obj):
        """Remove graphical object *obj*."""
        self._objects.remove(obj)

    def remove_all(self):
        """Remove all sub-objects."""
        self._objects = []

    def clear_transformation(self):
        """Clear all transformations."""
        MovableGraphicalObject.clear_transformation(self)
        for obj in self:
            obj.clear_transformation()

    @property
    def bounding_box(self):
        """Get bounding box around all the graphical objects inside."""
        b_box = None
        for i in range(len(self)):
            if b_box is None:
                b_box = self[i].bounding_box
            else:
                b_box.merge(self[i].bounding_box)

        return b_box

    def translate(self, vec):
        """Translate all sub-objects by a vector *vec*."""
        MovableGraphicalObject.translate(self, vec)
        for obj in self:
            obj.translate(vec)

    def rotate(self, angle, vec, total_start=None):
        """Rotate all sub-objects by *angle* around vector *vec*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        MovableGraphicalObject.rotate(self, angle, vec, total_start)
        for obj in self:
            obj.rotate(angle, vec, total_start)

    def move(self, abs_time):
        """Move to a position of the object in time *abs_time*."""
        # Move the whole object.
        MovableGraphicalObject.move(self, abs_time)
        for gr_object in self:
            # Then move its sub-objects.
            gr_object.move(abs_time)

    def save_transformation_matrices(self):
        """Save transformation matrices of all objects and return
        them in a dictionary {object: transform_matrix}.
        """
        for obj in self.all_objects:
            self._saved_matrices[obj] = np.copy(obj.transform_matrix)

    def restore_transformation_matrices(self):
        """Restore transformation matrices of all objects."""
        for obj, matrix in self._saved_matrices.iteritems():
            obj.transform_matrix = matrix

        self._saved_matrices = {}

    def get_next_time(self, t_0, distance):
        """
        Get next time at which the object will have traveled
        *distance*, the starting time is *t_0*.
        """
        # First deterimne the real distance which is smaller by the
        # given one because the combination of object movements might
        # exceed the distance if the motion of objects adds up
        # constructively.
        if self._dt is None:
            # Initialize
            self._dt = np.min([obj.get_maximum_dt(distance / len(self.all_objects))
                               for obj in self.all_objects]) * q.s

        for current_time in np.arange(t_0, self.time + self._dt, self._dt) * q.s:
            if self.moved(t_0, current_time, distance):
                return current_time

        return None

    def moved(self, t_0, t_1, distance):
        """
        Return True if the object moves between time *t_0* and *t_1*
        more than *distance*. We need to check all subobjects.
        Moreover, simple trajectory distance between points at t_0
        and t_1 will not work because when the composite object moves
        more than one pixel, but the primitive graphical object moves
        the exact opposite it results in no movement. We need to check
        also the composite object movement because it may cause some
        subobjects to rotate.
        """
        def move_and_save(abs_time):
            """Move primitive objects to time *abs_time* and return
            their positions.
            """
            primitive = self.primitive_objects
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

    def pack(self, units, coeff=1):
        """Pack all the sub-objects into a structure suitable for GPU
        calculation.
        """
        string = ""
        for gr_object in self._objects:
            string += gr_object.pack(units, coeff)

        return string

    def __repr__(self):
        return "CompositeObject{0}".format(self.objects)

    def __str__(self):
        return repr(self)


def get_moved_groups(objects, t_0, t_1, distance):
    """
    Filter only *objects* which truly move in the time interval
    *t_0*, *t_1* more than *distance*. Return a set of moved groups,
    where a group is defined by the last composite object which holds
    only primitive graphical objects. If a primitive object is in the
    *objects* it is included without further testing because if it
    didn't move it wouldn't be in the list.
    """
    moved = set([])

    for obj in objects:
        # Iterate over all root objects.
        if obj.__class__ == CompositeObject:
            # Add all last composites which truly moved
            moved.update([subobj for subobj in obj.get_last_composites()
                          if subobj.moved(t_0, t_1, distance)])
        else:
            # A primitive object wouldn't be in the list if it
            # didn't move.
            moved.add(obj)

    return moved


def get_format_string(string):
    """
    Get string in single or double precision floating point number
    format.
    """
    float_string = "f" if cfg.PRECISION.is_single() else "d"
    return string.replace("vf", float_string)
