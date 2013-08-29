import pyopencl as cl
import syris
from syris import config as cfg
from syris.gpu import util as g_util
import sys
from syris.tests.base import SyrisTest


def _get_queues(queue_kwargs=None):
    if queue_kwargs is None:
        queue_kwargs = {}

    return g_util.get_command_queues(g_util.get_cuda_context(),
                                     queue_kwargs=queue_kwargs)


class TestInit(SyrisTest):

    def test_no_queues(self):
        syris.init()
        self.assertEqual(cfg.QUEUE.__class__, cl.CommandQueue)
        self.assertEqual(cfg.CTX.__class__, cl.Context)

    def test_with_queues(self):
        queues = _get_queues()
        syris.init(queues)
        self.assertEqual(queues[0], cfg.QUEUE)

    def test_wrong_queues(self):
        sys.argv.append("--profile")
        queues = _get_queues()
        self.assertRaises(ValueError, syris.init, queues)
