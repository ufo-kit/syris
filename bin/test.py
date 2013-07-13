import logging
from matplotlib import pyplot as plt, cm
import numpy as np
import pyopencl as cl
import quantities as q
import syris
from syris import physics, config as cfg
from syris.gpu import util as g_util
from syris import physics, imageprocessing as ip

LOGGER = logging.getLogger(__name__)

def kernels():
    return """
    __kernel void mulmy(__global float *out) {
        int ix = get_global_id(0);
        //int iy = get_global_id(1);
        
        out[ix] = ix;
    }
    """

if __name__ == '__main__':
    syris.init()
    
    prg = g_util.get_program(kernels())
    
    n = 512
    mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE, size=n**2*4)
    res = np.empty((2*n,2*n), dtype=np.float32)
    
    prg.mulmy(cfg.QUEUE,
            (n,n),
            None,
            mem)
    print res[:n, :n].__class__
    cl.enqueue_copy(cfg.QUEUE, res[:n, :n], mem)
        
    plt.figure()
    plt.imshow(res, origin="lower", cmap=cm.get_cmap("gray"),
               interpolation="nearest")
    plt.colorbar()
    plt.show()
