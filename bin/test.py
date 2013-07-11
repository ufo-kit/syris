import logging
from matplotlib import pyplot as plt, cm
import pyopencl as cl
import syris
from syris import physics, config as cfg

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    syris.init()
    print physics.CL_PRG

        
#     plt.figure()
#     plt.imshow(res.magnitude, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
