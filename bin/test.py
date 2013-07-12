import logging
from matplotlib import pyplot as plt, cm
import numpy as np
import pyopencl as cl
import syris
from syris import physics, config as cfg
from syris.gpu import util as g_util

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    syris.init()
    mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE, size=4*4)
    ar = np.empty(4, np.float32)

    print cl.enqueue_copy(cfg.QUEUE, mem, ar)
        
#     plt.figure()
#     plt.imshow(res.magnitude, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
