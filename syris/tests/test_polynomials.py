import numpy as np
import pyopencl as cl
from unittest import TestCase
import syris
from syris import config as cfg
from syris.gpu import util as g_util
from graphics_util import f, derivative, filter_close, np_roots, Metaball


class TestPolynomials(TestCase):

    def setUp(self):
        syris.init()
        self.poly_deg = 4
        self.coeffs = np.array([5, 87, -2, 37, 17], dtype=cfg.NP_FLOAT)

        self.coeffs_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_ONLY |
                                    cl.mem_flags.COPY_HOST_PTR,
                                    hostbuf=self.coeffs)
        self.scalar_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                                    size=cfg.CL_FLOAT)

        self.pixel_size = 1e-3
        self.precision_places = int(np.log10(1 / self.pixel_size))
        self.prg = g_util.get_program(g_util.get_metaobjects_source())
        self.roots_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                                   size=(self.poly_deg + 1) * cfg.CL_FLOAT)

    def get_roots(self, coeffs, interval, previous_coeffs=None,
                  next_coeffs=None):
        if previous_coeffs is None:
            previous_coeffs = (self.poly_deg + 1) * [np.nan]
        previous_coeffs_mem = get_coeffs_mem(previous_coeffs)

        if next_coeffs is None:
            next_coeffs = (self.poly_deg + 1) * [np.nan]
        next_coeffs_mem = get_coeffs_mem(next_coeffs)

        self.prg.roots_kernel(cfg.QUEUE,
                              (1,),
                              None,
                              self.roots_mem,
                              previous_coeffs_mem,
                              get_coeffs_mem(coeffs),
                              next_coeffs_mem,
                              g_util.make_vfloat2(interval[0], interval[1]),
                              cfg.NP_FLOAT(self.pixel_size))

        res = np.empty(5, dtype=cfg.NP_FLOAT)
        cl.enqueue_copy(cfg.QUEUE, res, self.roots_mem)

        return res

    def get_stationary(self, coeffs, interval):
        zero = np.inf
        x_zero = None
        x_data = np.arange(interval[0], interval[1], self.pixel_size)
        for x in x_data:
            y = np.abs(f(derivative(derivative(coeffs)), x))
            if y < zero:
                zero = y
                x_zero = x

        return x_zero

    def _test_result(self, coeffs, interval, expected_a=np.nan,
                     expected_b=np.nan, expected_c=np.nan, expected_d=np.nan,
                     expected_e=np.nan):

        roots = self.get_roots(coeffs, interval)
        ground_truth = [expected_a, expected_b, expected_c, expected_d,
                        expected_e]

        np.testing.assert_almost_equal(roots, ground_truth,
                                       decimal=self.precision_places)

    def test_one_metaball(self):
        # Regular two roots.
        for r in np.linspace(self.pixel_size / 4, 1.0 - self.pixel_size, 10):
            mb = Metaball(r, 1.0, 0.0)
            coeffs = mb.get_coeffs(True)
            interval = (mb.lower, mb.upper)
            self._test_result(coeffs, interval, -r, r)

        # One root left
        mb = Metaball(1.0, 2.0, 0.0)
        coeffs = mb.get_coeffs(True)
        interval = mb.lower, mb.center - mb.r / 2
        self._test_result(coeffs, interval, -1.0, np.nan)

        # One root right
        interval = mb.center + mb.r / 2, mb.upper
        self._test_result(coeffs, interval, 1, np.nan)

        # Make interval end points stationary.
        coeffs = mb.get_coeffs(True)
        interval = (mb.lower, 0)
        self._test_result(coeffs, interval, -1, np.nan)
        interval = (0, mb.upper)
        self._test_result(coeffs, interval, 1, np.nan)

        # Put interval end points to roots.
        # In right endpoint.
        interval = mb.lower, -1.0
        self._test_result(coeffs, interval, -1.0)
        # In left endpoint.
        interval = -1.0, 0.0
        self._test_result(coeffs, interval)
        # By epsilon too far -> no root expected
        interval = mb.lower, - 1 - self.pixel_size
        self._test_result(coeffs, interval)

        interval = 1.0, mb.upper
        # By epsilon too far -> no root expected
        interval = 1.0 + self.pixel_size, mb.upper
        self._test_result(coeffs, interval, np.nan, np.nan)

        # Left end point is a root.
        coeffs = coeffs - np.array([0, 0, 0, 0, f(coeffs, mb.center)])
        interval = 0.0, mb.upper
        self._test_result(coeffs, interval, 0.0)

        # Right end point is a root.
        interval = mb.lower, 0.0
        self._test_result(coeffs, interval, 0.0)

        # No roots
        coeffs = mb.get_coeffs(True)
        coeffs = coeffs - np.array([0, 0, 0, 0, f(coeffs, mb.center) + 1])
        interval = mb.lower, mb.upper
        self._test_result(coeffs, interval, np.nan, np.nan)

        # No roots on the left from the stationary point
        coeffs = mb.get_coeffs(True)
        interval = mb.lower, (mb.lower - mb.r) / 2
        self._test_result(coeffs, interval, np.nan, np.nan)

        # No roots on the right from the stationary point
        coeffs = mb.get_coeffs(True)
        interval = mb.upper - (mb.upper - mb.r) / 2, mb.upper
        self._test_result(coeffs, interval, np.nan, np.nan)

        # No roots because the metaball does not reach y = 1.
        mb = Metaball(0.0, 0.5, 0.0)
        coeffs = mb.get_coeffs(True)
        interval = mb.lower, mb.upper
        self._test_result(coeffs, interval, np.nan, np.nan)

    def test_stationary(self):
        def test(offset):
            mb = Metaball(1.0, 2.0, 0.0)
            # Root into the stationary point.
            coeffs = mb.get_coeffs(True)
            coeffs = coeffs - \
                np.array([0, 0, 0, 0, f(coeffs, mb.center) + offset])
            # Tip of the quartic is the stationary point.
            interval = mb.lower, mb.upper
            self._test_result(coeffs, interval, *np_roots(coeffs, interval))

            # Moreover, split the interval in the root
            interval = mb.lower, 0.0
            result = self.get_roots(coeffs, interval)
            ground_truth = np_roots(coeffs, interval)

            interval = 0.0, mb.upper
            result = np.sort(np.concatenate((result,
                                             self.get_roots(coeffs,
                                                            interval))))
            ground_truth = np.sort(np.concatenate((ground_truth,
                                                   np_roots(coeffs,
                                                            interval))))
            np.testing.assert_almost_equal(result, ground_truth,
                                           decimal=self.precision_places)

        # There are actually two roots.
        test(0.0)
        test(-1e-7)

        # One stationary root.
        test(1e-7)

    def test_flat(self):
        # Below 0.
        coeffs = np.array([1111.1110839844, -2222.2221679688,
                           1666.6354980469, -555.5244750977, 68.4366683960])
        interval = 0.4962593913, 0.5037406087
        self._test_result(coeffs, interval, np.nan, np.nan)

        # Above 0.
        coeffs = np.array([1111.1110839844, -2222.2221679688,
                           1666.6354980469, -555.5244750977, 70.4366683960])
        self._test_result(coeffs, interval, np.nan, np.nan)

        # No stationary point found.
        coeffs = np.array([13.7174215317, -54.8696861267,
                           82.3028259277, -54.8662834167, 12.7157211304])
        interval = 0.9921267033, 1.0078732967
        self._test_result(coeffs, interval, np.nan, np.nan)

    def test_convex(self):
        mb_0 = Metaball(1.0, 2.0, 0.0)
        mb_1 = Metaball(1.0, 2.0, 2.8)

        interval = mb_1.lower, mb_0.upper
        coeffs = mb_0.get_coeffs(False) + mb_1.get_coeffs(True)
        self._test_result(coeffs, interval, *np_roots(coeffs, interval))
        a, b = np_roots(coeffs, interval)[:2]
        mid = (a + b) / 2

        # Only left root.
        interval = mb_1.lower, mid
        self._test_result(coeffs, interval, a, np.nan)

        # Only right root.
        interval = mid, mb_0.upper
        self._test_result(coeffs, interval, b, np.nan)

        # Roots are interval ends.
        # The root must be picked up either in interval i or i + 1
        interval = a, b
        result = self.get_roots(coeffs, interval)
        np_result = np_roots(coeffs, interval)
        interval = b, mb_0.upper
        result = np.sort(np.concatenate((result,
                                         self.get_roots(coeffs, interval))))
        np_result = np.sort(np.concatenate((np_result,
                                            np_roots(coeffs, interval))))
        result = filter_close(result)
        np_result = filter_close(np_result)
        np.testing.assert_almost_equal(result, np_result,
                                       decimal=self.precision_places)

    def test_stationary_far_from_center(self):
        mb = Metaball(0.1, 2.0, 0.0)

        # Far left and right beyond the object.
        # Both roots.
        coeffs = mb.get_coeffs(True)
        interval = mb.lower, 1.5 * mb.r
        self._test_result(coeffs, interval, -mb.r, mb.r)

        # One root.
        interval = mb.lower, 0
        self._test_result(coeffs, interval, -mb.r, np.nan)

        # Far right
        interval = -1.5 * mb.r, mb.upper
        self._test_result(coeffs, interval, -mb.r, mb.r)

        # One root.
        interval = 0, mb.upper
        self._test_result(coeffs, interval, mb.r, np.nan)

    def test_small_on_big_metaball(self):
        mb_0 = Metaball(0.0, 0.5, 0.51)
        mb_1 = Metaball(1.5, 100.0, -1.5)

        # First interval, only big metaball, two roots.
        coeffs = mb_1.get_coeffs(True)
        interval = mb_1.lower, mb_0.lower
        self._test_result(coeffs, interval, *np_roots(coeffs, interval))

        # Last interval, nothing in it.
        interval = mb_0.upper, mb_1.upper
        self._test_result(coeffs, interval, np.nan, np.nan)

        # Small and big metaballs, two roots.
        coeffs = mb_0.get_coeffs(False) + mb_1.get_coeffs(True)
        interval = mb_0.lower, mb_0.upper
        self._test_result(coeffs, interval, *np_roots(coeffs, interval))

    def test_left_root_nan(self):
        # Left root nan and right not nan.
        coeffs = np.array([69.4444, -138.889, 81.9445, -12.5, -0.4375])
        interval = 0.4, 0.9
        self._test_result(coeffs, interval, 0.7, np.nan)

    def test_derivative_stationary(self):
        """Test the case when there are two roots and f' has stationary
        points in the interval end points.
        """
        mb = Metaball(1.0, 2.0, 0.0)
        coeffs = mb.get_coeffs(True)

        stat_left = self.get_stationary(coeffs, (mb.lower, 0))
        stat_right = self.get_stationary(coeffs, (0, mb.upper))
        self._test_result(coeffs, (mb.lower, stat_right), -1, 1)
        self._test_result(coeffs, (stat_left, mb.upper), -1, 1)
        self._test_result(coeffs, (stat_left, stat_right), -1, 1)

    def test_newton_fail(self):
        """The derivatives in the end points point inside the interval,
        but the derivative in middle of the interval points outside of it,
        thus the Newton-Raphson itself fails and the iteration must be
        interleaved with some other method.
        """
        coeffs = np.array([233.1961669922, -17.0919075012, -18.3071880341,
                           3.6484642029, -0.0150656402], dtype=cfg.NP_FLOAT)
        interval = -0.01, 0.2012305856
        self._test_result(coeffs, interval, *np_roots(coeffs, interval))

    def test_non_continuous_transition(self):
        coeffs_0 = np.array([6618.19824219, -12948.47167969, 9326.49609375,
                             -2927.98315430, 337.40744019], dtype=cfg.NP_FLOAT)
        interval_0 = 0.38356042, 0.45985207

        coeffs_1 = np.array([245280.20312500, -500687.46875000,
                             381868.15625000, -128970.43750000,
                             16275.70703125], dtype=cfg.NP_FLOAT)

        roots = self.get_roots(coeffs_0, interval_0, next_coeffs=coeffs_1)
        np.testing.assert_almost_equal(0.45985207, roots[0])

    def test_splitting_intervals_out_of_scope(self):
        previous = [852986.12500000, -464643.28125000, 94117.21875000,
                    -8391.04882812, 277.376]
        next_coeffs = [453.00000000, -484.18750000, 140.35156250,
                       -5.86132812, -0.933594]
        coeffs = [852175.93750000, -464475.93750000, 94121.78906250,
                  -8392.4165, 277.323]
        interval = 0.15565501, 0.157952

        roots = self.get_roots(coeffs, interval, previous_coeffs=previous,
                               next_coeffs=next_coeffs)
        np.testing.assert_almost_equal(roots, (self.poly_deg + 1) * [np.nan])

    def test_polynomial_evaluation(self):
        x = cfg.NP_FLOAT(18.75)
        res = np.empty(1, dtype=cfg.NP_FLOAT)
        for i in range(self.poly_deg + 1):
            self.prg.polynomial_eval_kernel(cfg.QUEUE,
                                           (1,),
                                            None,
                                            self.scalar_mem,
                                            self.coeffs_mem,
                                            np.int32(i),
                                            x)
            cl.enqueue_copy(cfg.QUEUE, res, self.scalar_mem)
            self.assertAlmostEqual(res[0], f(self.coeffs[:i + 1], x), places=1)

    def test_derivative_evaluation(self):
        x = cfg.NP_FLOAT(18.75)
        res = np.empty(1, dtype=cfg.NP_FLOAT)
        for degree in range(self.poly_deg + 1):
            self.prg.derivative_eval_kernel(cfg.QUEUE,
                                           (1,),
                                            None,
                                            self.scalar_mem,
                                            self.coeffs_mem,
                                            np.int32(degree),
                                            x)
            cl.enqueue_copy(cfg.QUEUE, res, self.scalar_mem)
            self.assertAlmostEqual(res,
                                   f(derivative(self.coeffs[: degree + 1])
                                     [:degree], x))

    def test_derivate(self):
        out_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                            size=(self.poly_deg + 1) * cfg.CL_FLOAT)
        res = np.empty(self.poly_deg + 1, dtype=cfg.NP_FLOAT)
        for degree in range(0, self.poly_deg + 1):
            self.prg.derivate_kernel(cfg.QUEUE,
                                     (1,),
                                     None,
                                     out_mem,
                                     self.coeffs_mem,
                                     np.int32(degree))
            cl.enqueue_copy(cfg.QUEUE, res, out_mem)
            np.testing.assert_almost_equal(res[:degree],
                                           derivative(self.coeffs[:degree + 1])
                                           [:degree])


def get_coeffs_mem(coeffs):
    return cl.Buffer(cfg.CTX, cl.mem_flags.READ_ONLY |
                     cl.mem_flags.COPY_HOST_PTR,
                     hostbuf=np.array(coeffs, dtype=cfg.NP_FLOAT))
