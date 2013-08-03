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
        self._center = trajectory.points[0].simplified

        # Matrix holding transformation.
        self.transform_matrix = np.identity(4, dtype=cfg.NP_FLOAT)

        # Last position as tuple consisting of a 3D point and a vector giving
        # the object orientation.
        self._last_position = None

    @property
    def position(self):
        """Current position."""
        return np.dot(linalg.inv(self.transform_matrix), (0, 0, 0, 1))[:-1] * \
            self.center.units

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

    def moved(self, t_0, t_1, pixel_size):
        """Return True if the trajectory moved between time *t_0* and *t_1*
        more than one pixel with respect to the given *pixel_size*.
        """
        p_0 = self.trajectory.get_point(t_0)
        p_1 = self.trajectory.get_point(t_1)

        return geom.length(p_1 - p_0) > pixel_size

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
            raise ValueError("Blobbiness must be greater than zero.")

        self._radius = radius.simplified

    @property
    def radius(self):
        return self._radius

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

        radius = coeff * self.radius.rescale(units).magnitude
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


class MetaCube(MetaObject):

    """Metacube graphical object."""
    TYPE = OBJECT_ID()

    def __init__(self, trajectory, radius, orientation=geom.Y_AX):
        super(MetaCube, self).__init__(trajectory, radius, orientation)


class CompositeObject(GraphicalObject):

    """Class representing an object consisting of more sub-objects."""

    def __init__(self, trajectory, orientation=geom.Y_AX, gr_objects=[]):
        """*gr_objects* is a list of :py:class:`GraphicalObject`."""
        super(CompositeObject, self).__init__(trajectory, orientation)
        self._objects = gr_objects
        self._index = -1

    @property
    def objects(self):
        """All objects which are inside this composite object."""
        return tuple(self._objects)

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

    @property
    def primitive_objects(self):
        res = set()
        for obj in self:
            if obj.__class__ == self.__class__:
                res.update(obj.primitive_objects)
            else:
                res.add(obj)

        return tuple(res)

    def add(self, obj):
        """Add a graphical object *obj*."""
        if obj.__class__ == CompositeObject and self in obj.objects:
            raise ValueError("This instance is already inside the " +
                             "composite objects you are trying to add.")
        if obj not in self and obj is not self:
            self._objects.append(obj)

    def remove(self, obj):
        """Remove graphical object *obj*."""
        self._objects.remove(obj)

    def remove_all(self):
        """Remove all sub-objects."""
        self._objects = []

    def clear_transformation(self):
        """Clear all transformations."""
        GraphicalObject.clear_transformation(self)
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
        # Move the whole object.
        GraphicalObject.move(self, abs_time)
        for gr_object in self:
            # Then move its sub-objects.
            gr_object.move(abs_time)

    def moved(self, t_0, t_1, pixel_size):
        """Return True if the trajectory moved between time *t_0* and *t_1*
        more than one pixel with respect to the given *pixel_size*.
        """
        # We need to check all subobjects. Moreover, simple trajectory
        # distance between points at t_0 and t_1 will not work because
        # when the composite object moves more than one pixel, but the
        # primitive graphical object moves the exact opposite it results
        # in no movement. We are also interested only in primitive object
        # movement changes, because the composite objects do not show in
        # in the scene.
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
        for obj in self.primitive_objects:
            positions[obj] = obj.position

        # Forget that movement and move to t_1.
        self.clear_transformation()
        self.move(t_1)

        # Check the displacements of all primitive objects.
        res = False
        for obj in self.primitive_objects:
            if geom.length(obj.position - positions[obj]) > pixel_size:
                res = True
                break

        # Recover the transformation matrices.
        self.transform_matrix = matrix
        for obj in self:
            obj.transform_matrix = matrices[obj]

        return res

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
        return "Composite object(center=" + repr(self._center) +\
            ", subobjects=" + repr(len(self._objects)) + ")"

OBJECT_TYPES = {MetaCube.TYPE: "METACUBE",
                MetaBall.TYPE: "METABALL"}


def get_format_string(string):
    """Get string in single or double precision floating point number
    format."""
    float_string = "f" if cfg.single_precision() else "d"
    return string.replace("vf", float_string)
