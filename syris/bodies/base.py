# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""A base module for pysical bodies, which are optical elements having spatial extent."""
import logging
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import syris.config as cfg
import syris.geometry as geom
from quantities.quantity import Quantity
from scipy.optimize import bisect
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength, transfer, transfer_many
from syris.util import make_tuple


LOG = logging.getLogger(__name__)


class Body(OpticalElement):

    """An abstract body class with a *material*, which is a :class:`syris.materials.Material`
    instance.
    """

    def __init__(self, material=None):
        self.material = material

    def project(self, shape, pixel_size, offset, t=None, queue=None, out=None, block=False, camera=None, parallel=True):
        """Project thickness at time *t* to the image plane of size *shape* which is either 1D and
        is extended to (n, n) or is 2D as HxW. *pixel_size* is the point size, also either 1D or 2D.
        *offset* is the physical spatial body offset as (y, x). *queue* is an OpenCL command queue,
        *out* is the pyopencl array used for result. If *block* is True, wait for the kernel to
        finish.
        """
        shape = make_tuple(shape, num_dims=2)
        pixel_size = make_tuple(pixel_size, num_dims=2)
        if offset is None:
            offset = (0, 0) * q.m
        if queue is None:
            queue = cfg.OPENCL.queue

        return self._project(camera, parallel=parallel)

    def _project(self, shape, pixel_size, offset, t=None, queue=None, out=None, block=False, camera=None, parallel=True):
        """Projection function implementation. *shape* and *pixel_size* are 2D."""
        raise NotImplementedError

    def _transfer(
        self,
        shape,
        pixel_size,
        energy,
        offset,
        exponent=False,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        """Transfer function implementation based on a refractive index."""
        # ri = self.material.get_refractive_index(energy)
        # lam = energy_to_wavelength(energy)
        # proj = self.project(
        #     shape, pixel_size, offset=offset, t=t, queue=queue, out=out, block=block
        # )

        # return transfer(
        #     proj, ri, lam, exponent=exponent, queue=queue, out=out, check=check, block=block
        # )


class MovableBody(Body):

    """Class representing a movable body."""

    def __init__(self, trajectory, material=None, orientation=geom.Y_AX, cache_projection=True, coordinate_system=None):
        """Create a body with a :class:`~syris.geometry.Trajectory` and *orientation*,
        which is an (x, y, z) vector specifying body's "up" vector. If *cache_projection* is True,
        the projection is computed only if the object moves between the last projection time and the
        desired time.
        """
        super(MovableBody, self).__init__(material)
        self._trajectory = trajectory
        self._orientation = geom.normalize(orientation)
        self._center = trajectory.control_points[0].simplified

        # Matrix holding transformation.
        self.transform_matrix = np.identity(4, dtype=cfg.PRECISION.np_float)
        # Maximum body enlargement in any direction.
        self._scale_factor = np.ones(3)

        if coordinate_system is None:
            self._coordinate_system = geom.CoordinateSystem(origin=self._center)
            self._child_cs = self._coordinate_system.add_symmetric_child ("viewport")
        else:
            self._coordinate_system = coordinate_system
            if self._coordinate_system.has_child("viewport"):
                self._child_cs = self._coordinate_system.children["viewport"]
            else:
                self._child_cs = self._coordinate_system.add_symmetric_child ("viewport")

        # Last position as tuple consisting of a 3D point and a vector giving
        # the body orientation.
        self._last_position = None

        # tck of the spline following the distances from the beginning of the
        # body's trajectory
        self._distance_tck = None

        self._cache_projection = cache_projection
        self.update_projection_cache()

    def project(self, shape, pixel_size, offset=None, t=None, queue=None, out=None, block=False, camera=None, parallel=True):
        """Project thickness at time *t* (if it is None no transformation is applied) to the image
        plane of size *shape* which is either 1D and is extended to (n, n) or is 2D as HxW.
        *pixel_size* is the point size, also either 1D or 2D. *offset* is the physical spatial body
        offset as (y, x). *queue* is an OpenCL command queue, *out* is the pyopencl array used for
        result. If *block* is True, wait for the kernel to finish.
        """
        pixel_size = make_tuple(pixel_size, 2)
        if offset is None:
            offset = (0, 0) * q.m
        if t is not None:
            self.move(t)

        if self.cache_projection:
            if (
                self._p_cache["time"] is None
                or np.any(self._p_cache["ps"] != pixel_size)
                or self._p_cache["shape"] != shape
                or np.any(self._p_cache["offset"] != offset)
            ):
                moved = True
                self.update_projection_cache(t=t, shape=shape, pixel_size=pixel_size, offset=offset)
            else:
                # 0.99 to make sure we recompute when next_time from cached time is current t
                moved = self.moved(
                    min(self._p_cache["time"], t),
                    max(self._p_cache["time"], t),
                    0.99 * min(pixel_size),
                    bind=False,
                )

            if moved:
                LOG.debug("{} computing projection at {}".format(self, t))
                self._p_cache["time"] = t
                self._p_cache["projection"] = super(MovableBody, self).project(
                    shape, pixel_size, offset=offset, t=t, queue=queue, out=out, block=block, camera=camera, parallel=parallel
                )
            projection = self._p_cache["projection"]
        else:
            projection = super(MovableBody, self).project(
                shape, pixel_size, offset=offset, t=t, queue=queue, out=out, block=block, camera=camera, parallel=parallel
            )

        return projection

    def bind_trajectory(self, pixel_size):
        """Bind trajectory for *pixel_size*."""
        if (
            self.trajectory.pixel_size != pixel_size
            or self.trajectory.furthest_point != self.furthest_point
        ):
            fmt = "Binding trajectory to pixel size {} and furthest point {}"
            LOG.debug(fmt.format(pixel_size, self.furthest_point))
            self.trajectory.bind(pixel_size=pixel_size, furthest_point=self.furthest_point)

    @property
    def child_cs(self):
        return self._child_cs

    @property
    def cache_projection(self):
        """Whether or not projection cache is being used."""
        return self._cache_projection

    @cache_projection.setter
    def cache_projection(self, value):
        """If *value* is True, cache projections for cases when the body doesn't move, otherwise
        not.
        """
        self._cache_projection = value
        self.update_projection_cache()

    def update_projection_cache(
        self, t=None, shape=None, pixel_size=None, offset=None, projection=None
    ):
        """Update projection cache with time *t*, *shape*, *pixel_size*, *offset* and
        *projection*."""
        self._p_cache = {
            "time": t,
            "shape": shape,
            "ps": pixel_size,
            "offset": offset,
            "projection": projection,
        }

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
        # return self.transform_matrix[:3, -1] * q.m
        return self._coordinate_system.origin

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
        # self.transform_matrix = np.dot(trans_matrix, self.transform_matrix)
        pass

    def get_next_time(self, t_0, pixel_size):
        """
        Get time from *t_0* when the body will have travelled more than *pixel_size*.
        """
        self.bind_trajectory(pixel_size)

        return self.trajectory.get_next_time(t_0)

    def get_maximum_dt(self, pixel_size):
        """Get the maximum delta time for which the body will not move more than *pixel_size*
        between any two time points.
        """
        self.bind_trajectory(pixel_size)

        return self.trajectory.get_maximum_dt(distance=pixel_size)

    def get_distance(self, t_0, t_1):
        """Return the maximum principal axes translational and rotational travelled distance in time
        interval *t_0*, *t_1*.
        """
        if not self.trajectory.bound:
            raise ValueError("Trajectory not bound")

        u_0 = self.trajectory.get_parameter(t_0)
        u_1 = self.trajectory.get_parameter(t_1)
        dist = self.trajectory.get_distances(u_0=u_0, u=u_1)

        return np.max(dist) * q.m

    def moved(self, t_0, t_1, pixel_size, bind=True):
        """
        Return True if the body moves more than *pixel_size* in time interval *t_0*, *t_1*. If
        *bind* is True bind the trajectory to the specified *pixel_size*, otherwise use the
        trajectory as-is to compute an estimate.
        """
        if bind or not self.trajectory.bound:
            self.bind_trajectory(pixel_size)

        p_0 = self.trajectory.get_point(t_0)
        p_1 = self.trajectory.get_point(t_1)
        trans_displacement = np.abs(p_1 - p_0)

        d_0 = self.trajectory.get_direction(t_0, norm=False)
        d_1 = self.trajectory.get_direction(t_1, norm=False)
        rot_displacement = geom.get_rotation_displacement(d_0, d_1, self.furthest_point)
        total_displacement = trans_displacement + rot_displacement

        return max(total_displacement) > pixel_size

    def _find_next_rotation_time(self, abs_time):
        if not self.trajectory.bound:
            raise geom.TrajectoryError("Trajectory not bound")
        orientation = self.orientation.simplified.magnitude
        t = np.copy(abs_time.simplified.magnitude) * q.s

        def compute_rotation_axis(current_time):
            vec = self.trajectory.get_direction(current_time)
            rot_ax = np.cross(orientation, vec)

            return rot_ax

        rot_ax = compute_rotation_axis(t)
        vec = self.trajectory.get_direction(t).simplified.magnitude
        angle = geom.angle(orientation, vec)

        if np.all(np.isclose(rot_ax, 0)) and not (
            np.all(np.isclose(vec, orientation)) or self.trajectory.stationary
        ):
            # Orientation does not coincide with trajectory direction and trajectory is not
            # stationary.
            dt = self.get_maximum_dt(self.trajectory.pixel_size)
            t += dt
            while t < self.trajectory.time and np.all(np.isclose(rot_ax, 0)):
                # Orientation and trajectory direction are opposite, the angle between them is 180
                # deg
                rot_ax = compute_rotation_axis(t)
                t += dt
            if t >= self.trajectory.time:
                # Orientation and trajectory direction don't deviate at all from abs_time forward,
                # just use z axis
                rot_ax = geom.Z_AX

        return (rot_ax, angle)

    def move(self, abs_time, clear=True):
        """Move to a position of the body in time *abs_time*. If *clear* is true clear the
        transformation matrix first.
        """
        if clear:
            self.clear_transformation()
        abs_time = abs_time.simplified
        p_0 = self.trajectory.get_point(abs_time).simplified

        # First translate to the point at time abs_time
        self.translate(p_0)

        # Then rotate about rotation axis given by trajectory direction
        # and body orientation.
        rot_ax, angle = self._find_next_rotation_time(abs_time)
        self.rotate(angle, rot_ax)

    def translate(self, vec, inherit=False):
        """Translate the body by a vector *vec*."""
        # self.transform_matrix = np.dot(self.transform_matrix, geom.translate(vec))
        self._coordinate_system.translate(vec, inherit=inherit)

    def rotate(self, angle, axis, shift=None, inherit=False):
        """Rotate the body by *angle* around vector *vec*, where *shift* is the translation which
        takes place before the rotation and -*shift* takes place afterward, resulting in the
        transformation TRT^-1.
        """
        # self.transform_matrix = np.dot(self.transform_matrix, geom.rotate(angle, axis, shift=shift))
        if shift is None:
            self._coordinate_system.rotate_euler_local(axis, angle, inherit=inherit)
        else:
            self._coordinate_system.rotate_euler(shift, axis, angle, inherit=inherit)

    def visualize(self, plotter, cmap="viridis"):
        self._coordinate_system.visualize(plotter, cmap=cmap)


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
        """Furthest point is 0 for composite object."""
        return max([body.furthest_point for body in self.primitive_bodies])

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

    def __len__(self):
        return len(self._bodies)

    def __getitem__(self, key):
        return self._bodies[key]

    def __iter__(self):
        return self.bodies.__iter__()

    def __repr__(self):
        strings = ", ".join([repr(item) for item in self.bodies[: min(3, len(self.bodies))]])
        if len(self.bodies) > 3:
            strings += ", ..."
        return "CompositeBody({})".format(strings)

    def __str__(self):
        return repr(self)

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

    def translate(self, vec):
        """Translate all sub-bodies by a vector *vec*."""
        MovableBody.translate(self, vec)
        for body in self:
            body.translate(vec)

    def rotate(self, angle, vec, shift=None):
        """Rotate the body by *angle* around vector *vec*, where *shift* is the translation which
        takes place before the rotation and -*shift* takes place afterward, resulting in the
        transformation TRT^-1. Sub-bodies are rotated with respect to their relative position to the
        composite body.
        """
        local_shift = None
        MovableBody.rotate(self, angle, vec, shift=shift)
        for body in self:
            local_shift = -body.trajectory.control_points[0]
            if shift is not None:
                local_shift += shift
            body.rotate(angle, vec, shift=local_shift)

    def move(self, abs_time, clear=True):
        """Move to a position of the body in time *abs_time*. If *clear* is true clear the
        transformation matrix first.
        """
        if clear:
            self.clear_transformation()
        # Move the whole body.
        abs_time = abs_time.simplified
        p_0 = self.trajectory.get_point(abs_time).simplified

        # First translate to the point at time abs_time
        self.translate(p_0)

        # Then rotate about rotation axis given by trajectory direction
        # and body orientation.
        rot_ax, angle = self._find_next_rotation_time(abs_time)

        # Don't rotate the sub-bodies around this body as in CompositeBody.rotate(), they will do it
        # in their own move() functions
        MovableBody.rotate(self, angle, rot_ax)
        for body in self:
            body.rotate(angle, rot_ax)

        # Recursive motion of the sub-bodies
        for body in self:
            # Then move its sub-bodies.
            body.move(abs_time, clear=False)

    def save_transformation_matrices(self):
        """Save transformation matrices of all bodies and return
        them in a dictionary {body: transform_matrix}.
        """
        for body in self.all_bodies:
            self._saved_matrices[body] = np.copy(body.transform_matrix)

    def restore_transformation_matrices(self):
        """Restore transformation matrices of all bodies."""
        for body, matrix in self._saved_matrices.items():
            body.transform_matrix = matrix

        self._saved_matrices = {}

    def bind_trajectory(self, pixel_size):
        """Bind trajectory for *pixel_size*."""
        for body in self.all_bodies:
            if (
                body.trajectory.pixel_size != pixel_size
                or body.trajectory.furthest_point != body.furthest_point
            ):
                if body == self:
                    self._dt = None
                fmt = "Binding trajectory to pixel size {} and furthest point {}"
                LOG.debug(fmt.format(pixel_size, body.furthest_point))
                body.trajectory.bind(pixel_size=pixel_size, furthest_point=body.furthest_point)

    def get_maximum_dt(self, pixel_size):
        """Get the maximum delta time for which the body will not move more than *pixel_size*
        divided by the number of bodies because their movement can sum up constructively.
        """
        self.bind_trajectory(pixel_size)

        if self._dt is None:
            if self.trajectory.stationary:
                dts = []
            else:
                dts = [MovableBody.get_maximum_dt(self, pixel_size / len(self.all_bodies))]
            dts += [
                body.get_maximum_dt(pixel_size / len(self.all_bodies))
                for body in self
                if not body.trajectory.stationary
            ]
            self._dt = np.min(dts) * q.s if dts else None

        return self._dt

    def get_next_time(self, t_0, pixel_size, xtol=1e-12):
        """
        Get next time at which the body will have traveled *pixel_size*, the starting time is *t_0*.
        *xtol* is the absolute tolerance for bisection passed to :py:func:`scipy.optimize.bisect`.
        """

        def func(t):
            """Objective function for time bisection. The
            maximum dt obtained from individual trajectories might not be precise enough, but is
            precise enough to deterimne when the *pixel_size* is overstepped. Thus, we compute the
            time when the overall trajectory doesn't move more than *pixel_size*, then the time it
            does move more than the *pixel_size* and bisect to obtain the movement by exactly
            *pixel_size*.
            """
            t = t * q.s
            # scipy's bisection gets the root at 0, thus we need to shift by *pixel_size*
            return (self.get_distance(t_0, t) - pixel_size).simplified.magnitude

        self.bind_trajectory(pixel_size)
        if self._dt is None:
            self.get_maximum_dt(pixel_size)

        if self._dt is None:
            # All bodies are stationary
            return np.inf * q.s

        for current_time in (
            np.arange(
                t_0.simplified.magnitude,
                self.time.simplified.magnitude,
                self._dt.simplified.magnitude,
            )
            * q.s
        ):
            if self.moved(t_0, current_time, pixel_size):
                return bisect(func, t_0, current_time, xtol=xtol) * q.s

        return np.inf * q.s

    def get_distance(self, t_0, t_1):
        """Return the translational and rotational travelled distance in time interval
        *t_0*, *t_1*.
        """
        if not self.trajectory.bound:
            raise ValueError("Trajectory not bound")
        # Place the point for computing the derivative very close
        dt = (t_1 - t_0) * 1e-7

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
        p_0 = move_and_save(t_0)
        p_1 = move_and_save(t_1)
        if t_0 + dt < self.time:
            d_0 = move_and_save(t_0 + dt) - p_0
        else:
            d_0 = p_0 - move_and_save(t_0 - dt)
        if t_1 + dt < self.time:
            d_1 = move_and_save(t_1 + dt) - p_1
        else:
            d_1 = p_1 - move_and_save(t_1 - dt)
        self.restore_transformation_matrices()

        trans = np.abs(p_1 - p_0) * q.m

        rot = []
        for i in range(len(self.primitive_bodies)):
            rot.append(
                geom.get_rotation_displacement(
                    d_0[i], d_1[i], self.primitive_bodies[i].furthest_point.simplified.magnitude
                )
            )
        rot = np.array(rot) * q.m

        return np.max(trans) + np.max(rot)

    def moved(self, t_0, t_1, pixel_size):
        """Return True if the body moves more than *pixel_size* in time interval *t_0*, *t_1*."""
        return self.get_distance(t_0, t_1) > pixel_size

    def _project(self, shape, pixel_size, offset, t=None, queue=None, out=None, block=False, camera=None, parallel=True):
        """Projection function implementation. *shape* and *pixel_size* are 2D."""
        if out is None:
            out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_float)
        for body in self.bodies:
            out += body.project(
                camera, parallel=parallel
            )

        return out

    def _transfer(
        self,
        shape,
        pixel_size,
        energy,
        offset,
        exponent=False,
        t=None,
        queue=None,
        out=None,
        check=True,
        block=False,
    ):
        """Transfer function implementation based on a refractive index."""
        if out is None:
            out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_cplx)
        else:
            # transmission_many adds values, make sure it start with a zeroed array
            out.fill(0)

        return transfer_many(
            self.bodies,
            shape,
            pixel_size,
            energy,
            offset=offset,
            exponent=exponent,
            queue=queue,
            out=out,
            t=t,
            check=check,
            block=block,
        )
