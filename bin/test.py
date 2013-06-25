from syris.util import init
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
    init()
    ctx = gpu_util.get_cuda_context()
    q = gpu_util.get_command_queues(ctx)[0]
    
    print ctx, q
    