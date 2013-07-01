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

object_id = itertools.count().next


class GraphicalObject(object):

    """Class representing an abstract graphical object."""

    def __init__(self, trajectory, orientation=geom.Y_AX, v_0=0.0 * q.m / q.s,
                 v_max=None, accel_dist_ratio=0.0,
                 decel_dist_ratio=0.0):
        """Create a graphical object with a *trajectory*, where:

        * *orientation* - (x, y, z) vector specifying object's "up" vector
        * *v_0* - initial velocity
        * *v_max* - maximum velocity to achieve during movement
        * *accel_dist_ratio* - ratio between distance for which the object
            accelerates and total distance
        * *decel_dist_ratio* ratio between distance for which the object
            decelerates and total distance
        """
        self._trajectory = trajectory
        self._orientation = orientation
        self._center = trajectory.points[0]

        # Matrix holding transformation.
        self._trans_matrix = np.identity(4, dtype=cfg.NP_FLOAT)

        # Movement related attributes.
        self._v_max = cfg.NP_FLOAT(v_max)
        self._v_0 = v_0

        # Last position as tuple consisting of a 3D point and a vector giving
        # the object orientation.
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
    def orientation(self):
        return self._orientation

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
    def v_max(self):
        """Maximum achieved velocity."""
        return self._v_max

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
        v_0 = self._v_0.rescale(q.m / q.s)
        v_max = self._v_max.rescale(v_0.units)
        dist = self._trajectory.length.rescale(q.m)
        accel_dist = dist * self._accel_dist_ratio.rescale(dist.units)
        decel_dist = dist * self._decel_dist_ratio.rescale(dist.units)
        const_dist = dist - accel_dist - decel_dist.rescale(dist.units)

        if self._v_max <= 0 or self._trajectory.length == 0:
            self._acceleration = 0.0 * q.m / q.s ** 2
            self._deceleration = 0.0 * q.m / q.s ** 2
            self._accel_end_time = 0.0 * q.s
            self._decel_start_time = 0.0 * q.s
            return
        if self._accel_dist_ratio <= 0.0:
            accel = 0.0 * q.m / q.s ** 2
            accel_time = 0.0 * q.s
        else:
            # dist = 1/2at^2, v = at, ar...accel_dist_ratio
            # dist = 1/2vt => t = 2s/v => accel = v/t =
            # v/(2s/v) = v^2/2s = v^2/2(ar*dist)
            accel = (v_max ** 2 - v_0 ** 2) / (2 * accel_dist)
            accel_time = 2.0 * accel_dist / (v_0 + v_max)
        if self._decel_dist_ratio <= 0.0:
            decel = 0.0 * q.m / q.s ** 2
            decel_time = 0.0 * q.s
            # total_time = accel_time + const_time
            total_time = accel_time + const_dist / v_max
        else:
            decel = (v_max ** 2 - v_0 ** 2) / (2 * decel_dist)
            # t_d = t_a + t_mv (mv...v_max) (t = dist/v)
            # t_d = t_a + dist/v = t_a + dist*(1-ar-dr)/mv
            decel_time = accel_time + const_dist / v_max

            # Do not stop, just return to the v_0 speed.
            total_time = decel_time + 2.0 * decel_dist / (v_0 + v_max)

        self._acceleration = accel
        self._deceleration = decel
        self._accel_end_time = accel_time
        self._decel_start_time = decel_time
        self._total_time = total_time

    def get_distance(self, abs_time):
        """Get the distance traveled from the beginning until the time
        given by *abs_time*.
        """
        v_0 = self._v_0.rescale(q.m / q.s)
        v_max = self._v_max.rescale(v_0.units)
        acc_end = self._accel_end_time.rescale(q.s)
        decel_start = self._decel_start_time.rescale(q.s)
        total_time = self._total_time.rescale(q.s)
        accel = self._acceleration.rescale(q.m / q.s ** 2)
        decel = self._deceleration.rescale(q.m / q.s ** 2)

        if self._accel_dist_ratio > 0.0 and abs_time <= self._accel_end_time:
            return v_0 * abs_time + accel * abs_time ** 2 / 2.0
        elif self._decel_dist_ratio > 0.0 and\
                abs_time < self._decel_start_time:
            return v_0 * acc_end + accel * acc_end ** 2 / 2.0 + \
                v_max * (abs_time - acc_end)
        else:
            if self._decel_start_time > 0.0:
                return v_0 * acc_end + 0.5 * accel * acc_end ** 2 + \
                    v_max * (decel_start - acc_end) + \
                    (abs_time - decel_start) * v_0 + \
                    0.5 * decel * (2 * total_time * (abs_time - decel_start) +
                                   decel_start ** 2 - abs_time ** 2)
            else:
                # no deceleration
                return self._v_0 * self._accel_end_time +\
                    (self._acceleration * self._accel_end_time ** 2) / 2.0 +\
                    self._v_max * (abs_time - self._accel_end_time)

    def get_trajectory_index(self, abs_time):
        """Get index into trajectory points at a specified time *abs_time*."""
        if self._trajectory is None or self._trajectory.length == 0:
            return 0

        # get the minimum delta s
        d_s = self._trajectory.length / len(self._trajectory.points)
        # index to the coordinates list computation
        index = int(round((self.get_distance(abs_time)) / d_s))
        if index >= len(self._trajectory.points):
            LOGGER.debug("Index to trajectories out of bounds: " +
                         "index=%d, max_i=%d, abs_time=%g, d_s=%g, max_d=%g." %
                         (index, len(self._trajectory.points) - 1, abs_time,
                          d_s, self._trajectory.length) + "The object " +
                         "reaches beyond its trajectory end by " +
                         "the specified time.")
            index = len(self._trajectory.points) - 1

        return index

    def move(self, abs_time):
        """Move to a position of the object in time *abs_time*."""
        index = self.get_trajectory_index(abs_time)
        p_0 = self._trajectory.points[index]

        if index + 1 >= len(self.trajectory.points):
            # end of trajectory, use the same inclination as in the position
            # before
            vec = geom.normalize(p_0 - self._trajectory.points[index - 1])
        else:
            vec = geom.normalize(self._trajectory.points[index + 1] - p_0)

        self.translate((p_0[0], p_0[1], p_0[2], 1))

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
        self._trans_matrix = np.dot(geom.rotate(angle, axis, total_start))

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
                 orientation=geom.Y_AX, v_0=0.0 * q.m / q.s, v_max=None,
                 accel_dist_ratio=0.0, decel_dist_ratio=0.0):
        """Create a metaobject with *radius* and *blobbiness* defining the
        distance after the object's radius until which it influences the
        scene.
        """
        super(MetaObject, self).__init__(trajectory, orientation, v_0,
                                         v_max, accel_dist_ratio,
                                         decel_dist_ratio)
        if radius <= 0:
            raise ValueError("Blobbiness must be greater than zero.")
        if blobbiness is None:
            self._blobbiness = radius
        elif blobbiness <= 0:
            raise ValueError("Blobbiness must be greater than zero.")

        self._radius = radius
        self._blobbiness = blobbiness

    def radius(self):
        return self._radius

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
        return struct.pack(fmt, self.TYPE, self.radius, self.blobbiness,
                           self.get_transform_const(),
                           self.get_falloff_const(),
                           tuple(self._trans_matrix.flatten()))


class MetaBall(MetaObject):

    """Metaball graphical object."""
    TYPE = object_id()

    def __init__(self, trajectory, radius, blobbiness=None,
                 orientation=geom.Y_AX, v_0=0.0 * q.m / q.s, v_max=None,
                 accel_dist_ratio=0.0, decel_dist_ratio=0.0):
        super(MetaBall, self).__init__(trajectory, radius, blobbiness,
                                       orientation, v_0, v_max,
                                       accel_dist_ratio, decel_dist_ratio)

    def get_falloff_const(self):
        """Precompute mataball falloff curve constant which are the same
        for all the x,y coordinates. It ensures that f(x) = 1 <=> x = r."""
        influence = float(self._blobbiness + self._radius[0])
        transformation_const = self.get_transform_const()

        a_x = self._trans_matrix[0][2]
        a_y = self._trans_matrix[1][2]
        a_z = self._trans_matrix[2][2]
        # Calculate the 1/(influence^2 - r^2)^2 coefficient.
        center_x = self._center[0]
        center_y = self._center[1]
        k_x = self._trans_matrix[0][0] * center_x + \
            self._trans_matrix[0][1] * center_y + self._trans_matrix[0][3]
        k_y = self._trans_matrix[1][0] * center_x + \
            self._trans_matrix[1][1] * center_y + self._trans_matrix[1][3]
        k_z = self._trans_matrix[2][0] * center_x + \
            self._trans_matrix[2][1] * center_y + self._trans_matrix[2][3]

        roots = np.roots([transformation_const,
                          2 * k_x * a_x + 2 * k_y * a_y + 2 * k_z * a_z,
                          k_x * k_x + k_y * k_y + k_z * k_z - influence ** 2])
        influence_0 = (roots[1] - roots[0]) / 2
        roots = np.roots([transformation_const,
                          2 * k_x * a_x + 2 * k_y * a_y + 2 * k_z * a_z,
                          k_x * k_x + k_y * k_y + k_z * k_z -
                          self.radius ** 2])
        r_0 = (roots[1] - roots[0]) / 2

        return 1.0 / (influence_0 ** 2 - r_0 ** 2) ** 2


class MetaCube(MetaObject):

    """Metacube graphical object."""
    TYPE = object_id()

    def __init__(self, trajectory, radius, blobbiness=None,
                 orientation=geom.Y_AX, v_0=0.0 * q.m / q.s, v_max=None,
                 accel_dist_ratio=0.0, decel_dist_ratio=0.0):
        super(MetaCube, self).__init__(trajectory, radius, blobbiness,
                                       orientation, v_0, v_max,
                                       accel_dist_ratio, decel_dist_ratio)

    def get_falloff_const(self):
        """There is no falloff constant for metacubes."""
        return 0 * self.radius.units


class CompositeObject(GraphicalObject):

    """Class representing an object consisting of more sub-objects."""

    def __init__(self, trajectory, orientation=geom.Y_AX, v_0=0.0 * q.m / q.s,
                 v_max=None, accel_dist_ratio=0.0,
                 decel_dist_ratio=0.0, gr_objects=[]):
        """*gr_objects* are the graphical objects which is this object
        composed of.
        """
        super(CompositeObject, self).__init__(self, trajectory, orientation,
                                              v_0, v_max,
                                              accel_dist_ratio,
                                              decel_dist_ratio)
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
        index = self.get_trajectory_index(abs_time)
        p_0 = self._trajectory.points[index]

        if index + 1 >= len(self.trajectory.points):
            # end of trajectory, use the same inclination as in the position
            # before
            vec = geom.normalize(p_0 - self._trajectory.points[index - 1])
        else:
            vec = geom.normalize(self._trajectory.points[index + 1] - p_0)

        GraphicalObject.translate(self, (p_0[0], p_0[1], p_0[2], 1))

        rot_ax = geom.normalize(np.cross(self._orientation, vec))
        angle = -geom.angle(self._orientation, vec)
        GraphicalObject.rotate(self, angle, rot_ax)
        for obj in self:
            m_diff = obj.center - self._center
            obj.translate((p_0[0], p_0[1], p_0[2], 1))
            obj.rotate(angle, rot_ax)
            obj.translate(m_diff)

        self._last_position = p_0, vec

        return True

    def pack(self):
        """Pack all the sub-objects into a structure suitable for GPU
        calculation.
        """
        string = ""
        for obj in self._objects:
            string += obj.pack()
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
