import logging
from matplotlib import pyplot as plt, cm
import numpy as np
import pyopencl as cl
from pyopencl.array import vec
import quantities as q
import syris
from syris import physics, config as cfg
from syris.gpu import util as g_util
from syris import physics, imageprocessing as ip

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    syris.init()
    

#     plt.figure()
#     plt.imshow(res, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
