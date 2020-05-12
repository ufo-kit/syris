import numpy as np
import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as g_util
from .graphics_util import derivative, f, sgn, Metaball
from syris.tests import default_syris_init, SyrisTest, opencl


@opencl
class TestThickness(SyrisTest):

    def setUp(self):
        default_syris_init()
        self.pixel_size = 1e-3
        self.precision_places = int(np.log10(1 / self.pixel_size))
        self.prg = g_util.get_program(g_util.get_metaobjects_source())
        self.poly_deg = 4
        self.roots_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                                   size=(self.poly_deg + 1) * cfg.PRECISION.cl_float)

    def get_thickness_addition(self, coeffs, roots, previous,
                               last_derivative_sgn):
        out_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE,
                            size=self.poly_deg * cfg.PRECISION.cl_float)
        roots_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_WRITE |
                              cl.mem_flags.COPY_HOST_PTR,
                              hostbuf=np.array(roots, dtype=cfg.PRECISION.np_float))
        coeffs_mem = cl.Buffer(cfg.OPENCL.ctx, cl.mem_flags.READ_ONLY |
                               cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=np.array(coeffs, dtype=cfg.PRECISION.np_float))

        self.prg.thickness_add_kernel(cfg.OPENCL.queue,
                                      (1,),
                                      None,
                                      out_mem,
                                      coeffs_mem,
                                      roots_mem,
                                      cfg.PRECISION.np_float(previous),
                                      np.int32(last_derivative_sgn))

        res = np.empty(4, dtype=cfg.PRECISION.np_float)
        cl.enqueue_copy(cfg.OPENCL.queue, res, out_mem)

        return res[:-1]

    def test_thickness(self):
        def test(coeffs, roots, expected_thickness, expected_previous,
                 previous=np.nan, last_derivative_sgn=-2,
                 expected_last_derivative_sgn=None):
            thickness_res, previous_res, last_derivative_sgn_res = \
                self.get_thickness_addition(coeffs,
                                            roots, previous,
                                            last_derivative_sgn)
            np.testing.assert_almost_equal(expected_thickness, thickness_res,
                                           decimal=self.precision_places)
            np.testing.assert_almost_equal(expected_previous, previous_res,
                                           decimal=self.precision_places)
            if expected_last_derivative_sgn is not None:
                np.testing.assert_almost_equal(expected_last_derivative_sgn,
                                               last_derivative_sgn_res)

        mb_0 = Metaball(1.0, 2.0, 0.0)
        coeffs = mb_0.get_coeffs(True)

        # Simple cases, no previous, no last accounted value.
        roots = 4 * [np.nan]
        test(coeffs, roots, 0.0, np.nan, np.nan)

        roots_base = [-2.5, -1, 1, 2.5, np.nan]

        for i in range(1, 5):
            roots[:i] = roots_base[:i]
            if i % 2 == 0:
                # Roots are coupled.
                test(coeffs, roots + [np.nan], i / 2 * 1.5, np.nan)
            else:
                # Previous is created.
                test(coeffs, roots + [np.nan], (i - 1) / 2 * 1.5, roots[i - 1],
                     expected_last_derivative_sgn=sgn(f(derivative(coeffs),
                                                        roots[i])))

        # Previous is not nan.
        roots = [-1, 1, 1.5, np.nan, np.nan]
        previous = -3
        test(coeffs, roots, 2, roots[1], previous=previous,
             last_derivative_sgn=f(derivative(coeffs), previous))

        roots = [-1, 1, 2.5, 4, np.nan]
        previous = -3
        test(coeffs, roots, 3.5, np.nan, previous=previous)

        # The leftmost root was coupled before, just skip the current
        # first root.
        roots = [1, 2, np.nan, np.nan, np.nan]
        previous = np.nan
        last_accounted = 1 - self.pixel_size / 2
        last_derivative_sgn = sgn(f(derivative(coeffs), last_accounted))
        test(coeffs, roots, 0, roots[1], previous, last_derivative_sgn)

        # The leftmost root existed before, but was not coupled.
        # Take the root into account.
        roots = [1, 2, np.nan, np.nan, np.nan]
        previous = 1 - self.pixel_size / 2
        last_derivative_sgn = sgn(f(derivative(coeffs), previous))
        test(coeffs, roots, 1, np.nan, previous, last_derivative_sgn)

        # Extremum in the middle of the new roots, but the surrounding
        # roots lie on the opposite sides, thus take them into account.
        roots = [-2.1, -self.pixel_size / 4, self.pixel_size / 4, 2.1, np.nan]
        test(coeffs, roots, roots[1] - roots[0] + roots[3] - roots[2], np.nan)

        # Make the middle roots to be on the same slope of the extremum,
        # thus do not take them both into account.
        # First root is not coupled with anything, thus must be coupled
        # with one of them.
        roots = [-1, self.pixel_size / 4, self.pixel_size / 2, 2.5, np.nan]
        test(coeffs, roots, roots[1] - roots[0], roots[-2])

        # Create some previous value in order for the first root to be
        # coupled with some previous, then one of the two roots is saved
        # as previous.
        roots = [-1, self.pixel_size / 4, self.pixel_size / 2, np.nan, np.nan]
        previous = -2.5
        test(coeffs, roots, roots[0] - previous, roots[1], previous=previous)

        # Multiple roots.
        roots = [1, 1, 1, 1, np.nan]
        test(coeffs, roots, 0, 1,
             expected_last_derivative_sgn=sgn(f(derivative(coeffs), 1)))

        # Multiple roots with previous.
        previous = 1
        last_derivative_sgn = sgn(f(derivative(coeffs), previous))
        test(coeffs, roots, 0, previous,
             expected_last_derivative_sgn=last_derivative_sgn)

        # Left multiple roots.
        roots = [-1, -1, 1.5, 1.7, np.nan]
        test(coeffs, roots, roots[2] - roots[0], np.nan)

        # Right multiple roots.
        roots = [-1, 1, 2.5, 2.5, np.nan]
        test(coeffs, roots, roots[1] - roots[0], roots[-2])

        # Stationary points as roots which are not extrema.
        roots = [-2, 2, np.nan, np.nan, np.nan]
        test(coeffs, roots, 4, np.nan, np.nan)

        # More than polynomial degree roots.
        roots = [-2.5, -1, 1, 2.5, 2.5]
        test(coeffs, roots, 3, np.nan)
