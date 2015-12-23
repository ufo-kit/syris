"""
Bodies based on isosurfaces.
"""
import itertools
import logging
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import quantities as q
import syris.gpu.util as g_util
import syris.geometry as geom
import struct
from syris import config as cfg
from syris.bodies.base import CompositeBody, MovableBody
from syris.geometry import BoundingBox

LOG = logging.getLogger(__name__)


class MetaBall(MovableBody):

    """"Metaball bodies are smooth blobs formed by summing density functions representing particular
    bodies.
    """

    def __init__(self, trajectory, radius, material=None, orientation=geom.Y_AX):
        """Create a metaball with *radius*."""
        if radius <= 0:
            raise ValueError("Radius must be greater than zero.")

        self._radius = radius.simplified
        super(MetaBall, self).__init__(trajectory, material=material, orientation=orientation)

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
        transformed = []
        for point in points:
            transformed.append(geom.transform_vector(self.transform_matrix, point))

        return BoundingBox(np.array(transformed) * q.m)

    def _project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        return project_metaballs(shape, pixel_size, self.pack(), queue=queue, out=out)

    def get_transform_const(self):
        """
        Precompute the transformation constant which does not change for
        x,y position.
        """
        a_x = self.transform_matrix[0][2]
        a_y = self.transform_matrix[1][2]
        a_z = self.transform_matrix[2][2]
        return a_x ** 2 + a_y ** 2 + a_z ** 2

    def pack(self):
        """Pack the body into a structure suitable for OpenCL kernels. Packed units are in
        meters.
        """
        fmt = get_format_string("ffff")

        return struct.pack(fmt, self.position[0].simplified.magnitude,
                           self.position[1].simplified.magnitude,
                           self.position[2].simplified.magnitude,
                           self.radius.simplified.magnitude)

    def __repr__(self):
        return "MetaBall({0})".format(self.radius)

    def __str__(self):
        return repr(self)


class MetaBalls(CompositeBody):

    """Composite body composed of metaballs."""

    def __init__(self, trajectory, metaballs, orientation=geom.Y_AX):
        super(MetaBalls, self).__init__(trajectory, orientation=orientation, bodies=metaballs)

    def _project(self, shape, pixel_size, t=0 * q.s, queue=None, out=None):
        """Projection implementation."""
        string = ''.join([body.pack() for body in self._bodies])

        return project_metaballs(shape, pixel_size, string, queue=queue, out=out)


def get_moved_groups(bodies, t_0, t_1, distance):
    """Filter only *bodies* which truly move in the time interval *t_0*, *t_1* more than *distance*.
    Return a set of moved groups, where a group is defined by the last composite body which holds
    only primitive bodies. If a primitive body is in the *bodies* it is included without further
    testing because if it didn't move it wouldn't be in the list.
    """
    moved = set([])

    for body in bodies:
        # Iterate over all root bodies.
        if body.__class__ == CompositeBody:
            # Add all last composites which truly moved
            moved.update([subbody for subbody in body.get_last_composites()
                          if subbody.moved(t_0, t_1, distance)])
        else:
            # A primitive body wouldn't be in the list if it
            # didn't move.
            moved.add(body)

    return moved


def get_format_string(string):
    """
    Get string in single or double precision floating point number
    format.
    """
    float_string = "f" if cfg.PRECISION.is_single() else "d"
    return string.replace("vf", float_string)


def project_metaballs(shape, pixel_size, body_string, queue=None, out=None):
    """Project packed metaballs *body_string* on an image plane with *shape*, *pixel_size*. Use
    OpenCL *queue* and *out* pyopencl Array instance for returning the result.
    """
    n, m = shape
    ps = pixel_size.simplified.magnitude
    num_bodies = len(body_string) / 4 / cfg.PRECISION.cl_float
    if not queue:
        queue = cfg.OPENCL.queue

    bodies_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY |
                           cl.mem_flags.COPY_HOST_PTR, hostbuf=body_string)
    pbodies_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                            size=m * n * cfg.MAX_META_BODIES * 4 * 7)
    left_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                         size=m * n * 2 * cfg.MAX_META_BODIES)
    right_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                          size=m * n * 2 * cfg.MAX_META_BODIES)
    if out is None:
        out = cl_array.Array(queue, shape, cfg.PRECISION.np_float)

    cfg.OPENCL.programs['geometry'].metaballs(cfg.OPENCL.queue,
                                              (m, n),
                                              None,
                                              out.data,
                                              bodies_mem,
                                              pbodies_mem,
                                              left_mem,
                                              right_mem,
                                              np.int32(num_bodies),
                                              cl_array.vec.make_int2(0, 0),
                                              cl_array.vec.make_int4(0, 0, m, n),
                                              g_util.make_vfloat2(ps[1], ps[0]),
                                              np.int32(True))

    return out
