import syris
from syris import config as cfg


def test_init():
    syris.init()
    assert cfg.OPENCL.ctx is not None
