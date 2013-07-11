import logging
from matplotlib import pyplot as plt, cm
import pyopencl as cl
import syris
from syris import physics, config as cfg

LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    platforms = cl.get_platforms()[0]
    devices = platforms.get_devices()
    ctx = cl.Context(devices)
    queues = [cl.CommandQueue(ctx, devices[0], cl.command_queue_properties.PROFILING_ENABLE)]
    syris.init(queues)
    print physics.CL_PRG
    
    print cl.command_queue_properties.PROFILING_ENABLE
        
#     plt.figure()
#     plt.imshow(res.magnitude, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
