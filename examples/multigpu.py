import time
import pyopencl as cl
import numpy as np
import syris
import syris.config as cfg
import syris.gpu.util as gutil


def get_kernel():
    return """
        kernel void foo (global float *output,
                         const int stop)
        {
            int idx = get_global_id (0);
            int idy = get_global_id (1);
            int i, j;
            float result;

            for (i = 0; i < stop; i++) {
                for (j = 0; j < stop; j++) {
                    result = log ((float) i * j) + idx * idy;
                }
            }

            output[idy * get_global_size (0) + idx] = result;
        }
        """


def main():
    syris.init()
    devices = cfg.OPENCL.devices
    ctx = cfg.OPENCL.ctx
    queues = cfg.OPENCL.queues
    prg = cl.Program(ctx, get_kernel()).build()

    n = 512
    shape = (n, n)
    num_items = 2 * len(queues)
    events = []

    def process(item, queue):
        data = cl.array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
        event = prg.foo(queue, shape[::-1], None, data.data, np.int32(n))
        events.append(event)
        # Wait for the event so that the command queue will not be scheduled before the work has
        # been done.
        event.wait()

    start = time.time()
    gutil.qmap(process, range(num_items))
    host_duration = time.time() - start

    print '------------------------------'
    print '     All duration: {:.2f} s'.format(host_duration)
    print '------------------------------'

    all_duration = 0
    for i, event in enumerate(events):
        duration = get_duration(event)
        all_duration += duration
        print 'Device {} duration: {:.2f} s'.format(i, duration)

    print '------------------------------'
    print '          Speedup: {:.2f} / {}'.format(all_duration / host_duration, len(devices))
    print '------------------------------'


def get_duration(event):
    return (event.profile.end - event.profile.start) * 1e-9


if __name__ == '__main__':
    main()
