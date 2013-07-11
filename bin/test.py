from matplotlib import pyplot as plt, cm
import numpy as np
from numpy import linalg
import pyopencl as cl
import quantities as q
from syris import config as cfg
from syris.opticalelements import geometry as geom
from syris.gpu import util as gpu_util
import logging
from opticalelements.graphicalobjects import MetaBall
import time
import struct
from pyopencl.array import vec
from syris.devices.sources import BendingMagnet

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    ctx = gpu_util.get_cuda_context()
    queues = gpu_util.\
        get_command_queues(ctx,
                           queue_kwargs={"properties":
                                         cl.command_queue_properties.
                                         PROFILING_ENABLE})
    cfg.init(queues)
        
#     plt.figure()
#     plt.imshow(res.magnitude, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
