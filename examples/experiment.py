# Copyright (C) 2013-2023 Karlsruhe Institute of Technology
#
# This file is part of syris.
#
# This library is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library. If not, see <http://www.gnu.org/licenses/>.

"""Experiment example."""
import imageio
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from syris.bodies.mesh import make_cube, Mesh
from syris.devices.cameras import Camera, make_pco_dimax
from syris.devices.detectors import Detector
from syris.devices.lenses import Lens
from syris.devices.filters import MaterialFilter, Scintillator
from syris.devices.sources import make_topotomo
from syris.geometry import Trajectory
from syris.gpu.util import get_host
from syris.experiments import Experiment
from syris.math import fwnm_to_sigma
from .trajectory import create_sample, make_circle
from .util import get_default_parser, get_material, show


def get_flat(
    shape,
    energies,
    detector,
    source,
    filters=(),
    shot_noise=False,
    amplifier_noise=False,
    psf=False,
):
    image = np.zeros(shape)

    for e in energies:
        u = source.transfer(shape, detector.pixel_size, e, t=0 * q.s)
        for oe in filters:
            u *= oe.transfer(shape, detector.pixel_size, e, t=0 * q.s)
        flat = (abs(u) ** 2).get()
        det = detector.convert(flat, e)
        image += det * detector.camera.exp_time.simplified.magnitude

    return detector.camera.get_image(
        image, shot_noise=shot_noise, amplifier_noise=amplifier_noise, psf=psf
    )


def make_devices(n, energies, camera=None, highspeed=True, scintillator=None):
    """Create devices with image shape (*n*, *n*), X-ray *energies*, *camera* and use the high speed
    setup if *highspeed* is True.
    """
    shape = (n, n)
    dE = energies[1] - energies[0]

    if not camera:
        vis_wavelengths = np.arange(500, 700) * q.nm
        camera = Camera(
            11 * q.um,
            0.1,
            500,
            23,
            32,
            shape,
            exp_time=1 * q.ms,
            fps=1000 / q.s,
            quantum_efficiencies=0.5 * np.ones(len(vis_wavelengths)),
            wavelengths=vis_wavelengths,
            dtype=np.float32,
        )
    else:
        vis_wavelengths = camera.wavelengths.rescale(q.nm)

    x = vis_wavelengths.rescale(q.nm).magnitude
    if scintillator == "lso" or not (scintillator or highspeed):
        sigma = fwnm_to_sigma(50)
        emission = np.exp(-((x - 545) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        lso = get_material("lso_5_30_kev.mat")
        scintillator = Scintillator(
            13 * q.um,
            lso,
            36 * np.ones(len(energies)) / q.keV,
            energies,
            emission / q.nm,
            vis_wavelengths,
            1.82,
        )
    elif scintillator == "luag" or (not scintillator and highspeed):
        sigma = fwnm_to_sigma(50)
        emission = np.exp(-((x - 450) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
        luag = get_material("luag.mat")
        scintillator = Scintillator(
            50 * q.um,
            luag,
            14 * np.ones(len(energies)) / q.keV,
            energies,
            emission / q.nm,
            vis_wavelengths,
            1.84,
        )

    if highspeed:
        # High speed setup
        lens = Lens(3, f_number=1.4, focal_length=50 * q.mm, transmission_eff=0.7, sigma=None)
    else:
        # High resolution setup
        lens = Lens(9, na=0.28, sigma=None)

    detector = Detector(scintillator, lens, camera)
    source_trajectory = Trajectory([(n / 2, n / 2, 0)] * detector.pixel_size)
    bm = make_topotomo(dE=dE, trajectory=source_trajectory, pixel_size=detector.pixel_size)

    return bm, detector


def make_topo_tomo_flat(args, highspeed=True, scintillator=None):
    syris.init(device_index=0, loglevel="INFO")
    dimax = make_pco_dimax()
    n = dimax.shape[0]
    energies = np.arange(5, 30) * q.keV

    bm, detector = make_devices(
        n, energies, camera=dimax, highspeed=highspeed, scintillator=scintillator
    )

    # Customize the setup for the 2013_03_07-08 experiment
    air = MaterialFilter(1 * q.m, get_material("air_5_30_kev.mat"))
    filters = [air]
    dimax.bpp = 32
    dimax.dtype = np.float32
    dimax.fps = 450 / q.s
    detector.lens.f_number = 2.8
    bm.el_current = 130 * q.mA

    # Compte the flat field
    flat = get_flat(
        dimax.shape, energies, detector, bm, filters=filters, shot_noise=True, amplifier_noise=True
    )
    fmt = "min: {}, max: {}, mean: {}, middle row std: {}"
    print(fmt.format(flat.min(), flat.max(), flat.mean(), flat[n // 2].std()))

    show(flat)
    plt.show()

    return flat


def make_motion(args):
    syris.init()
    n = 256
    shape = (n, n)
    energies = np.arange(5, 30, 1) * q.keV
    bm, detector = make_devices(n, energies)
    mb = create_sample(n, detector.pixel_size, velocity=20 * q.mm / q.s)
    mb_2 = create_sample(n, detector.pixel_size, velocity=10 * q.mm / q.s)
    mb.material = get_material("pmma_5_30_kev.mat")
    mb_2.material = mb.material

    cube = make_cube() / q.m * 30 * detector.pixel_size + 0.1 * detector.pixel_size
    fov = detector.pixel_size * n
    circle = make_circle().magnitude * fov / 30000 + fov / 2
    tr = Trajectory(circle, velocity=10 * q.um / q.s)
    glass = get_material("glass.mat")
    mesh = Mesh(cube, tr, material=glass)
    ex = Experiment([bm, mb, mb_2, mesh], bm, detector, 0 * q.m, energies)

    for sample in ex.samples:
        if sample != bm:
            sample.trajectory.bind(detector.pixel_size)

    if args.show_flat:
        show(get_flat(shape, energies, detector, bm), title="Counts")
        plt.show()

    if args.conduct:
        if args.output is not None and not os.path.exists(args.output):
            os.makedirs(args.output, mode=0o755)

        t_0 = 0 * q.s
        if args.num_images:
            t_1 = args.num_images / detector.camera.fps
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
                path = os.path.join(args.output, "projection_{:>05}.png").format(i)
                imageio.imwrite(path, image)

        print("Maximum intensity:", image.max())
        print("Duration: {} s".format(time.time() - st))

    plt.show()


def main():
    """Parse command line arguments and execute one of the experiments."""
    parser = get_default_parser(__doc__)
    subparsers = parser.add_subparsers(help="sub-command help", dest="sub-commands", required=True)

    motion = subparsers.add_parser("motion", help="An experiment with motion")
    motion.add_argument("--output", type=str, help="Output directory for moving objects.")
    motion.add_argument("--show", action="store_true", help="Show images as they are produced")
    motion.add_argument("--show-flat", action="store_true", help="Show a flat field image")
    motion.add_argument("--conduct", action="store_true", help="Conduct the experiment")
    motion.add_argument("--num-images", type=int, help="Number of images to produce")
    motion.set_defaults(_func=make_motion)

    flat = subparsers.add_parser("flat", help="Quantitatively correct flat field computation")
    flat.set_defaults(_func=make_topo_tomo_flat)

    args = parser.parse_args()
    args._func(args)


if __name__ == "__main__":
    main()
