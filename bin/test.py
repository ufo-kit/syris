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

def kernels():
    return """
    __kernel void ones(__global float *out) {
        out[get_global_id(0)] = 1;
    }
    
    __kernel void mul(__global float *out) {
        int ix = get_global_id(0);
        
        ones(out);
        
        out[ix] += 4; 
    }
    """

if __name__ == '__main__':
    syris.init()
    
    prg = g_util.get_program(kernels())
    n = 4
    mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE, size=n*cfg.CL_FLOAT)
    
    prg.mul(cfg.QUEUE,
            (n,),
            None,
            mem)
    
    res = np.empty(n, dtype=cfg.NP_FLOAT)
    cl.enqueue_copy(cfg.QUEUE, res, mem)
    
    print res
        
#     plt.figure()
#     plt.imshow(res, origin="lower", cmap=cm.get_cmap("gray"),
#                interpolation="nearest")
#     plt.colorbar()
#     plt.show()
