import numpy as np
import pyopencl as cl
from syris import config as cfg
from syris.gpu import util as gpu_util
import logging

logger = logging.getLogger(__name__)


def kernels():
    return """
    __kernel void test_kernel(__global float* mem) {
        int ix = get_global_id(0);

        mem[ix] = ix;
    }
    """

if __name__ == '__main__':
    ctx = gpu_util.get_cuda_context()
    queues = gpu_util.\
        get_command_queues(ctx,
                           queue_kwargs={"properties":
                                         cl.command_queue_properties.
                                         PROFILING_ENABLE})
    cfg.init(queues)
    q = queues[0]
    n = 8
    mem = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=n*cfg.cl_float)
    prg = cl.Program(ctx, kernels()).build()

    gpu_util.execute(prg.test_kernel, q,
                     (n,),
                     None,
                     mem)

    res = np.empty(n, dtype=cfg.np_float)
    cl.enqueue_copy(q, res, mem)

    print res
