"""Code speed on a specific platform."""
import logging
import os
import time
import numpy as np
import quantities as q
import syris
import syris.config as cfg
import syris.gpu.util as gutil
from syris.bodies.simple import make_sphere
from syris.materials import make_fromfile
from syris.physics import propagate
from util import get_default_parser


LOG = logging.getLogger(__name__)


def propagate_one(unused, queue, shape, energies, distance, ps, spheres):
    # Make sure we use the sample created by this *queue*
    sample = spheres[queue]
    return propagate([sample], shape, energies, distance, ps,
                     mollified=False, queue=queue, block=True).real.get()


def run(n, ps, num_runs, queues):
    LOG.info('n: %d', n)
    shape = (n, n)
    distance = 5 * q.m
    material = make_fromfile(os.path.join('examples', 'data', 'pmma_5_30_kev.mat'))
    energies = np.linspace(material.energies[0].magnitude,
                           material.energies[-1].magnitude,
                           100, endpoint=False) * material.energies.units
    spheres = {queue: make_sphere(n, n / 4 * ps, pixel_size=ps, material=material, queue=queue)
               for queue in queues}

    durations = []
    for i in range(num_runs):
        st = time.time()
        gutil.qmap(propagate_one, range(len(queues)), queues=queues,
                   args=(shape, energies, distance, ps, spheres))
        durations.append((time.time() - st) / len(queues))
        LOG.info('%d. run, duration: %.2f s', i + 1, durations[-1])

    mean = np.mean(durations)
    std = np.std(durations)
    LOG.info('Mean duration: %.2f s', mean)
    LOG.info('std: %.2f s', std)

    return mean, std


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument('--n', type=int, default=[1024], nargs='+',
                        help='List of number of pixels in one dimension')
    parser.add_argument('--num-devices', type=int, default=1,
                        help='Number of compute devices to use')
    parser.add_argument('--pixel-size', type=float, default=1, help='Pixel size [um]')
    parser.add_argument('--platform', type=str, help='Platform name substring')
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--output', type=str, help='Output file name')

    return parser.parse_args()


def main():
    args = parse_args()
    syris.init(platform_name=args.platform, loglevel='INFO')
    if args.num_devices > len(cfg.OPENCL.queues):
        fmt = "There are only {} devices available for platform '{}'"
        raise ValueError(fmt.format(len(cfg.OPENCL.devices), args.platform))
    queues = cfg.OPENCL.queues[:args.num_devices]

    result = []
    for n in args.n:
        mean, std = run(n, args.pixel_size * q.um, args.runs, queues)
        result.append((n, mean, std))

    if args.output:
        np.savetxt(args.output, result, fmt='%g',
                   header='n\tmean duration [s]\tstd [s], number of runs: {}'.format(args.runs),
                   delimiter='\t')

    return result


if __name__ == '__main__':
    main()
