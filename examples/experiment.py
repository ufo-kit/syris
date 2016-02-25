"""Experiment example."""
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import scipy.misc
import syris
from syris.bodies.mesh import make_cube, Mesh
from syris.devices.cameras import Camera
from syris.devices.detectors import Detector
from syris.devices.lenses import Lens
from syris.devices.filters import Scintillator
from syris.devices.sources import BendingMagnet
from syris.geometry import Trajectory
from syris.gpu.util import get_host
from syris.experiments import Experiment
from syris.materials import make_fromfile
from trajectory import create_sample, make_circle
from util import show


def get_flat(shape, energies, detector, source):
    image = np.zeros(shape)

    for e in energies:
        flat = (abs(source.transfer(shape, detector.pixel_size, e)) ** 2).get()
        det = detector.convert(flat, e)
        image += det * detector.camera.exp_time.simplified.magnitude

    return detector.camera.get_image(image, shot_noise=False, amplifier_noise=False, psf=False)


def get_material(name):
    """Load material from file *name*."""
    return make_fromfile(os.path.join('examples', 'data', name))


def main():
    syris.init()
    n = 256
    shape = (n, n)
    energies = np.arange(5, 30) * q.keV
    vis_wavelengths = np.arange(500, 700) * q.nm
    args = parse_args()

    camera = Camera(11 * q.um, .1, 500, 23, 32, shape, fps=1000 / q.s,
                    quantum_efficiencies=0.5 * np.ones(len(vis_wavelengths)),
                    wavelengths=vis_wavelengths, dtype=np.float32)
    lens = Lens(3, f_number=1.4, focal_length=50 * q.mm, transmission_eff=0.7, sigma=None)
    luag = get_material('luag.mat')
    scintillator = Scintillator(50 * q.um,
                                luag,
                                14 * np.ones(len(energies)) / q.keV,
                                energies,
                                np.ones(len(vis_wavelengths)) / (len(vis_wavelengths) * q.nm),
                                vis_wavelengths,
                                1.84)
    detector = Detector(scintillator, lens, camera)
    mb = create_sample(n, detector.pixel_size, velocity=20 * q.mm / q.s)
    mb_2 = create_sample(n, detector.pixel_size, velocity=10 * q.mm / q.s)
    mb.material = get_material('pmma_5_30_kev.mat')
    mb_2.material = mb.material

    cube = make_cube() / q.m * 30 * detector.pixel_size + 0.1 * detector.pixel_size
    fov = detector.pixel_size * n
    circle = make_circle().magnitude * fov / 30000 + fov / 2
    tr = Trajectory(circle, velocity=10 * q.um / q.s)
    glass = get_material('glass.mat')
    mesh = Mesh(cube, tr, material=glass)
    # mesh.material = mb.material
    mesh.bind_trajectory(detector.pixel_size)
    source_trajectory = Trajectory([(n / 2, n / 2, 0)] * detector.pixel_size)
    bm = BendingMagnet(2.5 * q.GeV, 100 * q.mA, 1.5 * q.T, 30 * q.m, energies,
                       (200, 800) * q.um, detector.pixel_size, source_trajectory)
    ex = Experiment([bm, mb, mb_2, mesh], bm, detector, 0 * q.m)

    for sample in ex.samples:
        if sample != bm:
            sample.trajectory.bind(detector.pixel_size)

    if args.show_flat:
        show(get_flat(shape, energies, detector, bm), title='Counts')
        plt.show()

    if args.conduct:
        if args.output is not None and not os.path.exists(args.output):
            os.makedirs(args.output, mode=0o755)

        t_0 = 0 * q.s
        if args.num_images:
            t_1 = args.num_images / camera.fps
        else:
            t_1 = ex.time

        st = time.time()
        mpl_im = None
        for i, proj in enumerate(ex.make_sequence(t_0, t_1)):
            image = get_host(proj)

            if args.show:
                if mpl_im is None:
                    plt.figure()
                    mpl_im = plt.imshow(image)
                    plt.show(False)
                else:
                    mpl_im.set_data(image)
                    plt.draw()

            if args.output:
                path = os.path.join(args.output, 'projection_{:>05}.png').format(i)
                scipy.misc.imsave(path, image)

        print 'Maximum intensity:', image.max()
        print 'Duration: {} s'.format(time.time() - st)

    plt.show()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Output directory for moving objects.')
    parser.add_argument('--show', action='store_true', help='Show images as they are produced')
    parser.add_argument('--show-flat', action='store_true', help='Show a flat field image')
    parser.add_argument('--conduct', action='store_true', help='Conduct the experiment')
    parser.add_argument('--num-images', type=int, help='Number of images to produce')

    return parser.parse_args()


if __name__ == '__main__':
    main()
