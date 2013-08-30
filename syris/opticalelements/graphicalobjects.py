"""
Graphical objects
=================

There are two types of graphical objects:

    * **primitive** - children of :py:class:`GraphicalObject` but *not* \
        children of :py:class:`CompositeObject`
    * **composite** - can hold other graphical objects, including \
        another composites, these are the children of \
        :py:class:`CompositeObject`

"""
import itertools
import logging
import numpy as np
from numpy import linalg
import quantities as q
from syris import config as cfg
from syris.opticalelements.geometry import BoundingBox
import syris.opticalelements.geometry as geom
import struct
from quantities.quantity import Quantity

LOGGER = logging.getLogger(__name__)

OBJECT_ID = itertools.count().next


class GraphicalObject(object):

    """Class representing an abstract graphical object."""

    def __init__(self, trajectory, orientation=geom.Y_AX):
        """Create a graphical object with a *trajectory* and *orientation*,
        which is an (x, y, z) vector specifying object's "up" vector
        """
        self._trajectory = trajectory
        self._orientation = geom.normalize(orientation)
        self._center = trajectory.control_points[0].simplified

        # Matrix holding transformation.
        self.transform_matrix = np.identity(4, dtype=cfg.NP_FLOAT)
        # Maximum object enlargement in any direction.
        self._scale_factor = np.ones(3)

        # Last position as tuple consisting of a 3D point and a vector giving
        # the object orientation.
        self._last_position = None

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
        self.transform_matrix = np.identity(4, dtype=cfg.NP_FLOAT)
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

    def get_combined_displacement(self, t_0, t_1):
        """
        Get the displacement traveled between times *t_0* and *t_1*.
        Take into account both translation and rotation of the object.
        """
        trans_d = geom.length(self.trajectory.get_point(t_1) -
                              self.trajectory.get_point(t_0))
        rot_d = geom.get_rotation_displacement(
            self.trajectory.get_direction(t_0),
            self.trajectory.get_direction(t_1),
            self.furthest_point)

        return trans_d + rot_d

    def get_next_time(self, t_0, delta_distance):
        """
        Get next time at which the object will have traveled
        *delta_distance*, the starting time is *t_0*.
        """
        t_1 = self.trajectory.get_next_time(t_0, delta_distance)
        if t_1 is None:
            return None

        rot_d = geom.get_rotation_displacement(
            self.trajectory.get_direction(t_0),
            self.trajectory.get_direction(t_1),
            self.furthest_point)

        dist = rot_d + delta_distance
        d_t = (t_1 - t_0) / 2

        while dist > delta_distance:
            dist = self.get_combined_displacement(t_0, t_0 + d_t)
            d_t /= 2.0

        # Return the last bigger than *delta_distance*.
        return t_0 + 2 * d_t

    def moved(self, t_0, t_1, pixel_size):
        """Return True if the trajectory moved between time *t_0* and *t_1*
        more than one pixel with respect to the given *pixel_size*.
        """
        p_0 = self.trajectory.get_point(t_0)
        p_1 = self.trajectory.get_point(t_1)

        return geom.length(p_1 - p_0) > pixel_size / 2

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

    def scale(self, scale_vec):
        """Scale the object by scaling coefficients (kx, ky, kz)
        given by *sc_vec*.
        """
        self._scale_factor *= np.array(scale_vec)

        self.transform_matrix = np.dot(
            geom.scale(scale_vec), self.transform_matrix)


class MetaObject(GraphicalObject):

    """"Metaball-like graphical object. Metaballs are smooth blobs formed
    by summing density functions representing particular objects."""

    # Object type.
    TYPE = None

    def __init__(self, trajectory, radius, orientation=geom.Y_AX):
        """Create a metaobject with *radius*."""
        super(MetaObject, self).__init__(trajectory, orientation)
        if radius <= 0:
            raise ValueError("Radius must be greater than zero.")

        self._radius = radius.simplified

    @property
    def radius(self):
        return self._radius

    @property
    def bounding_box(self):
        """Bounding box of the object."""
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

    def pack(self, units, coeff=1):
        """Pack the object into a structure suitable for OpenCL kernels.
        Rescale the object using *units* first. *coeff* is a normalization
        factor for object's radius.
        """
        fmt = get_format_string("ifff" + 16 * "f")

        radius = coeff.rescale(1 / units).magnitude * \
            self.radius.rescale(units).magnitude

        # influence region = 2 * r, thus the coefficient guaranteeing
        # f(r) = 1 is
        # 1 / (R^2 - r^2)^2 = 1 / (4r^2 - r^2)^2 = 1 / 9r^4
        falloff_coeff = 1.0 / (9 * radius ** 4)

        trans_mat = self.get_rescaled_transform_matrix(units)
        return struct.pack(fmt, self.TYPE,
                           self.radius.rescale(units).magnitude,
                           falloff_coeff,
                           self.get_transform_const(),
                           *trans_mat.flatten())


class MetaBall(MetaObject):

    """Metaball graphical object."""
    TYPE = OBJECT_ID()

    def __init__(self, trajectory, radius, orientation=geom.Y_AX):
        super(MetaBall, self).__init__(trajectory, radius, orientation)

    @property
    def furthest_point(self):
        """
        Furthest point is twice the radius because of the influence
        region of the metaball.
        """
        return 2 * self.radius * max(self._scale_factor)

    def __repr__(self):
        return "MetaBall({0})".format(self.radius)

    def __str__(self):
        return repr(self)


class MetaCube(MetaObject):

    """Metacube graphical object."""
    TYPE = OBJECT_ID()

    def __init__(self, trajectory, radius, orientation=geom.Y_AX):
        super(MetaCube, self).__init__(trajectory, radius, orientation)


class CompositeObject(GraphicalObject):

    """
    Class representing an object consisting of more sub-objects.
    A composite object can be thought of as a tree structure with
    children representing actual graphical objects (non-composite
    only in leafs). This means that all direct children of a composite
    object are either all :py:class:`CompositeObject` instances
    or are all not. An example of a :py:class:`CompositeObject`
    and its children could be::

                            C          (0)
                           / \\
                          C   C        (1)
                         / \\   \\
                        P   P   P      (2)

    We see that the root has only composite children on level (1) and
    both its children have only primitive children on level (2).
    """

    def __init__(self, trajectory, orientation=geom.Y_AX, gr_objects=None):
        """*gr_objects* is a list of :py:class:`GraphicalObject`."""
        super(CompositeObject, self).__init__(trajectory, orientation)
        if gr_objects is None:
            gr_objects = []
        self._objects = []

        # Do not just assign gr_objects but let them all go through
        # add method, so the list is checked for correct input.
        for obj in gr_objects:
            self.add(obj)

    @property
    def objects(self):
        """All objects which are inside this composite object."""
        return tuple(self._objects)

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
    def total_time(self):
        """The total trajectory time of the object and all its subobjects."""
        return max([obj.trajectory.time for obj in self._all_objects(False)])

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
        if len(self) != 0:
            children_primitive = self[0].__class__ != CompositeObject
            obj_primitive = obj.__class__ != CompositeObject

            if children_primitive ^ obj_primitive:
                raise TypeError("Composite object direct children " +
                                "must be all of the same type")
        
        # enable bottom-up traversing
        obj.parent = self
        
        self._objects.append(obj)

    def remove(self, obj):
        """Remove graphical object *obj*."""
        self._objects.remove(obj)

    def remove_all(self):
        """Remove all sub-objects."""
        self._objects = []
        
    def get_last_composites(self):
        """
        Traverse the tree structure of the composite object and
        return a list of composite objects inside it which have only
        primitive children.
        """
        result = []
    
        def go_down(obj):
            """
            Go down in composite object's children list and look for
            instances which have children of type :py:class:`GraphicalObject`.
            Objects which are composed of purely primitive graphical objects
            are appended to the *result* list.
            """
            primitive = set([subobj for subobj in obj
                             if subobj.__class__ != CompositeObject])
            if primitive:
                result.append(obj)
            else:
                for comp_obj in set(obj) - primitive:
                    go_down(comp_obj)
    
        go_down(self)
    
        return result

    def clear_transformation(self):
        """Clear all transformations."""
        GraphicalObject.clear_transformation(self)
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
        GraphicalObject.translate(self, vec)
        for obj in self:
            obj.translate(vec)

    def rotate(self, angle, vec, total_start=None):
        """Rotate all sub-objects by *angle* around vector *vec*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        GraphicalObject.rotate(self, angle, vec, total_start)
        for obj in self:
            obj.rotate(angle, vec, total_start)

    def scale(self, scale_vec):
        """Scale all sub-objects by scaling coefficients (kx, ky, kz)
        given by *sc_vec*.
        """
        GraphicalObject.scale(self, scale_vec)
        for obj in self:
            obj.scale(scale_vec)

    def move(self, abs_time):
        """Move to a position of the object in time *abs_time*."""
        # Move the whole object.
        GraphicalObject.move(self, abs_time)
        for gr_object in self:
            # Then move its sub-objects.
            gr_object.move(abs_time)

    def get_next_time(self, t_0, delta_distance):
        """
        Get next time at which the object will have traveled
        *delta_distance*, the starting time is *t_0*.
        """
        t_1 = None

        # Get the smallest time from the primitive objects needed to travel
        # more than delta_distance. It will serve as an initial guess.
        for obj in self.primitive_objects:
            tmp = obj.get_next_time(t_0, delta_distance)
            if tmp is not None and (t_1 is None or tmp < t_1):
                t_1 = tmp

        if t_1 is None:
            return None

        d_t = t_1 - t_0

        # It might happen that in combination with trajectories of
        # parent objects the t_1 from the guess is actually too small
        # and we need to move forward in time.
        while not self.moved(t_0, t_0 + d_t, delta_distance):
            if t_0 + d_t > self.total_time:
                # The superposition of all subtrajectories makes this
                # object stationary from t_0 on.
                return None
            d_t *= 2

        # After we have made sure some objects will move, minimize
        # the movement down to delta_distance.
        d_t /= 2
        while self.moved(t_0, t_0 + d_t, delta_distance):
            d_t /= 2

        return t_0 + 2 * d_t

    def moved(self, t_0, t_1, delta_distance):
        """Return True if the trajectory moved between time *t_0* and *t_1*
        more than one pixel with respect to the given *delta_distance*.
        We need to check all subobjects. Moreover, simple trajectory
        distance between points at t_0 and t_1 will not work because
        when the composite object moves more than one pixel, but the
        primitive graphical object moves the exact opposite it results
        in no movement. We need to check also the composite object
        movement because it may cause some subobjects to rotate.
        """
        # Remember the current transformation matrices.
        matrix = np.copy(self.transform_matrix)
        matrices = {}
        for obj in self:
            matrices[obj] = np.copy(obj.transform_matrix)

        # Clear the transformation matrix
        self.clear_transformation()

        # Move to t_0.
        self.move(t_0)

        # Remember all primitive object positions.
        positions = {}
        orientations = {}
        for obj in self.primitive_objects:
            orientations[obj] = geom.transform_vector(obj.transform_matrix,
                                                      obj.orientation)
            positions[obj] = obj.position

        # Forget that movement and move to t_1.
        self.clear_transformation()
        self.move(t_1)

        # Check the displacements of all primitive objects.
        res = False
        for obj in self.primitive_objects:
            orientation = geom.transform_vector(obj.transform_matrix,
                                                obj.orientation)
            # Determine the maximum angle by which the object can move
            # and not move by more than *delta_distance*. The object moved
            # if the angle between the object pose at t_0 and t_1 combined
            # with the translation is larger than *delta_distance*.
            rot_d = geom.get_rotation_displacement(orientation,
                                                   orientations[obj],
                                                   obj.furthest_point)
            tran_d = geom.length(obj.position - positions[obj])
            if rot_d + tran_d > delta_distance:
                res = True
                break

        # Recover the transformation matrices.
        self.transform_matrix = matrix
        for obj in self:
            obj.transform_matrix = matrices[obj]

        return res

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

OBJECT_TYPES = {MetaCube.TYPE: "METACUBE",
                MetaBall.TYPE: "METABALL"}

def get_format_string(string):
    """
    Get string in single or double precision floating point number
    format.
    """
    float_string = "f" if cfg.single_precision() else "d"
    return string.replace("vf", float_string)
