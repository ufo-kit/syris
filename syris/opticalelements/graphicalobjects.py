"""Graphical objects."""
import itertools
import logging
import numpy as np
from numpy import linalg
import quantities as q
from syris import config as cfg
from syris.opticalelements.geometry import BoundingBox
import syris.opticalelements.geometry as geom
import struct

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
        self._center = trajectory.points[0].simplified

        # Matrix holding transformation.
        self._trans_matrix = np.identity(4, dtype=cfg.NP_FLOAT)

        # Last position as tuple consisting of a 3D point and a vector giving
        # the object orientation.
        self._last_position = None

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
    def center(self, center):
        self._center = center.simplified

    @property
    def orientation(self):
        return self._orientation

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

    def apply_transformation(self, trans_matrix):
        """Apply transformation given by the transformation matrix
        *trans_matrix* on the current transformation matrix.
        """
        self._trans_matrix = np.dot(trans_matrix, self._trans_matrix)

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
        self._trans_matrix = np.dot(geom.translate(vec), self._trans_matrix)

    def rotate(self, angle, axis, total_start=None):
        """Rotate the object by *angle* around vector *axis*, where
        *total_start* is the center of rotation point which results in
        transformation TRT^-1.
        """
        self._trans_matrix = np.dot(geom.rotate(angle, axis, total_start),
                                    self._trans_matrix)

    def scale(self, scale_vec):
        """Scale the object by scaling coefficients (kx, ky, kz)
        given by *sc_vec*.
        """
        self._trans_matrix = np.dot(geom.scale(scale_vec), self._trans_matrix)


class MetaObject(GraphicalObject):

    """"Metaball-like graphical object. Metaballs are smooth blobs formed
    by summing density functions representing particular objects."""

    # Object type.
    TYPE = None

    def __init__(self, trajectory, radius, blobbiness=None,
                 orientation=geom.Y_AX):
        """Create a metaobject with *radius* and *blobbiness* defining the
        distance after the object's radius until which it influences the
        scene.
        """
        super(MetaObject, self).__init__(trajectory, orientation)
        if radius <= 0:
            raise ValueError("Blobbiness must be greater than zero.")
        if blobbiness is None:
            blobbiness = radius
        elif blobbiness <= 0:
            raise ValueError("Blobbiness must be greater than zero.")

        self._radius = radius.simplified
        self._blobbiness = blobbiness.simplified

    @property
    def radius(self):
        return self._radius

    @property
    def blobbiness(self):
        """Influence region behind the radius of the object."""
        return self._blobbiness

    def get_transform_const(self):
        """Precompute the transformation constant which does not change for
        x,y position."""
        a_x = self._trans_matrix[0][2]
        a_y = self._trans_matrix[1][2]
        a_z = self._trans_matrix[2][2]
        return a_x ** 2 + a_y ** 2 + a_z ** 2

    def pack(self):
        """Pack the object into a structure suitable for OpenCL kernels."""
        fmt = get_format_string("iffff" + 16 * "f")

        return struct.pack(fmt, self.TYPE, self.radius.magnitude,
                           self.blobbiness.magnitude,
                           self.get_falloff_const(),
                           self.get_transform_const(),
                           *self._trans_matrix.flatten())


class MetaBall(MetaObject):

    """Metaball graphical object."""
    TYPE = OBJECT_ID()

    def __init__(self, trajectory, radius, blobbiness=None,
                 orientation=geom.Y_AX):
        super(MetaBall, self).__init__(trajectory, radius, blobbiness,
                                       orientation)

    def get_falloff_const(self):
        """Precompute mataball falloff curve constant which are the same
        for all the x,y coordinates. It ensures that f(x) = 1 <=> x = r."""
        influence = self._blobbiness + self._radius
        transformation_const = self.get_transform_const()

        a_x = self._trans_matrix[0][2]
        a_y = self._trans_matrix[1][2]
        a_z = self._trans_matrix[2][2]
        # Calculate the 1/(influence^2 - r^2)^2 coefficient.
        center_x = self._center[0]
        center_y = self._center[1]
        k_x = self._trans_matrix[0][0] * center_x + \
            self._trans_matrix[0][1] * center_y + \
            self._trans_matrix[0][3] * q.m
        k_y = self._trans_matrix[1][0] * center_x + \
            self._trans_matrix[1][1] * center_y + \
            self._trans_matrix[1][3] * q.m
        k_z = self._trans_matrix[2][0] * center_x + \
            self._trans_matrix[2][1] * center_y + \
            self._trans_matrix[2][3] * q.m

        roots = np.roots([transformation_const,
                          2 * k_x * a_x + 2 * k_y * a_y + 2 * k_z * a_z,
                          k_x * k_x + k_y * k_y + k_z * k_z - influence ** 2])
        influence_0 = (roots[1] - roots[0]) / 2 * q.m
        roots = np.roots([transformation_const,
                          2 * k_x * a_x + 2 * k_y * a_y + 2 * k_z * a_z,
                          k_x * k_x + k_y * k_y + k_z * k_z -
                          self.radius ** 2])
        r_0 = (roots[1] - roots[0]) / 2 * q.m

        return 1.0 / (influence_0 ** 2 - r_0 ** 2) ** 2


class MetaCube(MetaObject):

    """Metacube graphical object."""
    TYPE = OBJECT_ID()

    def __init__(self, trajectory, radius, blobbiness=None,
                 orientation=geom.Y_AX):
        super(MetaCube, self).__init__(trajectory, radius, blobbiness,
                                       orientation)

    def get_falloff_const(self):
        """There is no falloff constant for metacubes."""
        return 0 * q.m


class CompositeObject(GraphicalObject):

    """Class representing an object consisting of more sub-objects."""

    def __init__(self, trajectory, orientation=geom.Y_AX, gr_objects=[]):
        """*gr_objects* are the graphical objects which is this object
        composed of.
        """
        super(CompositeObject, self).__init__(self, trajectory, orientation)
        self._objects = gr_objects
        self._index = -1

    @property
    def objects(self):
        """All objects which are inside this composite object."""
        return self._objects

    def __len__(self):
        if self._objects:
            count = len(self._objects)
        else:
            count = 0

        return count

    def __getitem__(self, key):
        return self._objects[key]

    def __iter__(self):
        return self

    def next(self):  # @ReservedAssignment
        """Method needed for iteration over the object."""
        if (not self._objects or len(self._objects) == 0 or
                self._index + 1 == len(self._objects)):
            self._index = -1
            raise StopIteration

        self._index += 1

        return self._objects[self._index]

    def primitive_len(self):
        """Get number of graphical objects which are not composite objects."""
        return self._primitive_len(primitive_objects=[])

    def _primitive_len(self, primitive_objects=[]):
        """Internal primitive objects counter. *primitive_objects* is the
        accumulated list of primitive objects within this composite object."""
        for obj in self._objects:
            if obj.__class__ == CompositeObject:
                obj._primitive_len(primitive_objects)
            else:
                if obj not in primitive_objects:
                    primitive_objects.append(obj)

        return len(primitive_objects)

    def append(self, obj):
        """Add a graphical object *obj*."""
        self._objects.append(obj)

    def remove(self, obj):
        """Remove graphical object *obj*."""
        self._objects.remove(obj)

    def remove_all(self):
        """Remove all sub-objects."""
        self._objects = []

    def clear_transformation(self):
        """Clear all transformations."""
        super.clear_transformation(self)
        for obj in self:
            obj.clear_transformation()

    def get_bounding_box(self):
        """Get bounding box around all the graphical objects inside."""
        for i in range(len(self)):
            b_box = self[i].get_bounding_box()
            xmin = b_box.get_min(geom.X)
            ymin = b_box.get_min(geom.Y)
            zmin = b_box.get_min(geom.Z)
            xmax = b_box.get_max(geom.X)
            ymax = b_box.get_max(geom.Y)
            zmax = b_box.get_max(geom.Z)
            if i == 0:
                bpoints = [[xmin, ymin, zmin], [xmax, ymax, zmax]]
            else:
                if xmin < bpoints[0][0]:
                    bpoints[0][0] = xmin
                if ymin < bpoints[0][1]:
                    bpoints[0][1] = ymin
                if zmin < bpoints[0][2]:
                    bpoints[0][2] = zmin
                if xmax > bpoints[1][0]:
                    bpoints[1][0] = xmax
                if ymax > bpoints[1][1]:
                    bpoints[1][1] = ymax
                if zmax > bpoints[1][2]:
                    bpoints[1][2] = zmax

        return BoundingBox(list(itertools.product([xmin, xmax],
                                                  [ymin, ymax],
                                                  [zmin, zmax])))

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
        # First adjust the tranformation matrix.
        GraphicalObject.move(self, abs_time)
        for gr_object in self:
            # Then apply the transformation matrix on the sub-objects and
            # move also them according to their relative trajectories.
            gr_object.apply_transformation(self.transformation_matrix)
            gr_object.move(abs_time)

    def pack(self):
        """Pack all the sub-objects into a structure suitable for GPU
        calculation.
        """
        string = ""
        for gr_object in self._objects:
            string += gr_object.pack()

        return string

    def __repr__(self):
        return "CompositeObject(%s)" % (str(self))

    def __str__(self):
        return "center=" + repr(self._center) +\
            ", subobjects=" + repr(len(self._objects))

OBJECT_TYPES = {MetaCube.TYPE: "METACUBE",
                MetaBall.TYPE: "METABALL"}


def get_format_string(string):
    """Get string in single or double precision floating point number
    format."""
    float_string = "f" if cfg.single_precision() else "d"
    return string.replace("vf", float_string)
