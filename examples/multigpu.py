"""Show multi-device speedup on a problem of size n x m^k, where n is the number of pixels to
compute, m is the base number of operations per pixel powered to k.
"""
import argparse
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
            int n = get_global_size (0);
            int i;
            float result;

            for (i = 1; i < stop + 1; i++) {
                result = log ((float) i + idx);
            }

            output[idx] = result;
        }
        """


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=512 ** 2, help='Number of pixels (default 512^2)')
    parser.add_argument('--m', type=int, default=64, help='Number of pixel operations (default 64)')
    parser.add_argument('--k', type=float, default=1.0, nargs='*',
                        help='Time complexity, there are n x m^k operations in total '
                        '(default 1). The argument might have one (single run) or 3 numbers '
                        '(meaning start stop step) when a scan is performed.')

    args = parser.parse_args()
    m = args.k
    if not (len(m) == 1 or len(m) == 3):
        raise ValueError('--k must contain either 1 or 3 numbers')

    return args


def run(n, m, complexity, prg):
    devices = cfg.OPENCL.devices
    queues = cfg.OPENCL.queues

    stop = int(m ** complexity)
    complexity_fmt = 'complexity: {} x {}^{}, pixel operations: {}'
    print complexity_fmt.format(n, m, complexity, stop)
    num_items = 2 * len(queues)
    events = []

    def process(item, queue):
        data = cl.array.Array(queue, (n,), dtype=cfg.PRECISION.np_float)
        event = prg.foo(queue, (n, 1), None, data.data, np.int32(stop))
        events.append(event)
        # Wait for the event so that the command queue will not be scheduled before the work has
        # been done.
        event.wait()

    start = time.time()
    gutil.qmap(process, range(num_items))
    host_duration = time.time() - start

    print '-------------------------------'
    print '     All duration: {:.2f} s'.format(host_duration)
    print '-------------------------------'

    all_duration = 0
    for i, event in enumerate(events):
        duration = get_duration(event)
        all_duration += duration
        print 'Device {} duration: {:.2f} s'.format(i, duration)

    speedup = all_duration / host_duration
    print '-------------------------------'
    print '    Mean duration: {:.2f} s'.format(all_duration / len(devices))
    print '-------------------------------'
    print '          Speedup: {:.2f} / {}'.format(speedup, len(devices))
    print '-------------------------------'

    return speedup


def main():
    args = parse_args()
    syris.init()
    prg = cl.Program(cfg.OPENCL.ctx, get_kernel()).build()

    if len(args.k) == 1:
        complexities = args.k
    else:
        complexities = np.arange(args.k[0], args.k[1] + args.k[2], args.k[2])
    print 'Complexities: {}'.format(complexities)

    results = []
    for complexity in complexities:
        results.append(run(args.n, args.m, complexity, prg))

    print
    print '==============================='
    for i, result in enumerate(results):
        print 'Complexity: {:.2f}, speedup: {:.2f}'.format(complexities[i], result)
    print '==============================='


def get_duration(event):
    return (event.profile.end - event.profile.start) * 1e-9


if __name__ == '__main__':
    main()
