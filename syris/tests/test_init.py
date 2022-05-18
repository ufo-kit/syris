import logging
import pyopencl as cl
import pyopencl.cltypes as cltypes
import syris
import syris.physics
import syris.profiling
from syris import config as cfg
from syris.tests import SyrisTest


class TestInit(SyrisTest):
    def setUp(self):
        self.orig_platform_func = cl.get_platforms
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        cl.get_platforms = self.orig_platform_func

    def test_init(self):
        syris.init(profiling=True, loglevel=logging.DEBUG, double_precision=True)
        self.assertEqual(logging.DEBUG, syris.physics.LOG.getEffectiveLevel())
        self.assertNotEqual(cfg.OPENCL.ctx, None)
        self.assertEqual(cfg.PRECISION.cl_float, 8)
        self.assertNotEqual(syris.profiling.PROFILER, None)
        self.assertEqual(cfg.PRECISION.vfloat2, cltypes.double2)

    def test_no_opencl_init(self):
        """Initialization by broken OpenCL must work too, just the context and profiling not."""
        cl.get_platforms = lambda: None
        syris.init(profiling=False, loglevel=logging.DEBUG, double_precision=True)
        self.assertEqual(logging.DEBUG, syris.physics.LOG.getEffectiveLevel())
        self.assertEqual(cfg.PRECISION.cl_float, 8)
        self.assertEqual(cfg.PRECISION.vfloat2, cltypes.double2)
