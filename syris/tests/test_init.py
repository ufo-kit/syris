import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as g_util
import sys
from syris.tests.base import SyrisTest


def test_init():
    syris.init()
    assert cfg.OPENCL.ctx is not None
