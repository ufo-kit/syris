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

"""An X-ray source example."""
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
import syris.imageprocessing as ip
from syris.geometry import Trajectory
from syris.devices.sources import BendingMagnet, FixedSpectrumSource
from .util import get_default_parser, show


def make_triangle(n=128):
    lin = np.linspace(0, 2, n, endpoint=False)
    y = np.abs(lin - 1)
    z = np.zeros(n)

    return list(zip(z, y, z)) * q.mm


def run_bending_magnet():
    syris.init()

    ps = 1 * q.um
    dE = 1 * q.keV
    energies = np.arange(5, 30, dE.magnitude) * q.keV
    cp = make_triangle(n=16) * 1e-1
    tr = Trajectory(cp, velocity=10 * q.um / q.s, pixel_size=ps)

    bm = BendingMagnet(2.5 * q.GeV, 100 * q.mA, 1.5 * q.T, 30 * q.m, dE, (200, 800) * q.um, ps, tr)

    # Flat at time = 0
    flat_0 = (abs(bm.transfer((512, 256), ps, energies[0], t=0 * q.s)) ** 2).real.get()
    # Flat at half the time
    flat_1 = (abs(bm.transfer((512, 256), ps, energies[0], t=tr.time / 2)) ** 2).real.get()

    plt.subplot(121)
    plt.imshow(flat_0)
    plt.subplot(122)
    plt.imshow(flat_1)
    plt.show()


def run_fixed():
    syris.init()
    n = 512
    ps = 1 * q.um
    energies = np.arange(5, 30) * q.keV
    y, x = np.mgrid[-n // 2 : n // 2, -n // 2 : n // 2]
    flux = np.exp(-(x ** 2 + y ** 2) / (100 ** 2)) / q.s
    weights = np.arange(1, len(energies) + 1)[:, np.newaxis, np.newaxis]
    flux = weights * flux
    traj = Trajectory([(n / 2, n / 2, 0)] * ps)
    source = FixedSpectrumSource(energies, flux, 30 * q.m, (100, 500) * q.um, traj, pixel_size=ps)

    im = ip.compute_intensity(source.transfer((n, n), ps, 5 * q.keV)).get()
    show(im, title="Original sampling")
    im = ip.compute_intensity(source.transfer((2 * n,) * 2, ps / 2, 5 * q.keV)).get()
    show(im, title="2x supersampled")
    plt.show()


def main():
    parser = get_default_parser(__doc__)
    subparsers = parser.add_subparsers(help="sub-command help", dest="sub-commands", required=True)

    bm = subparsers.add_parser("bm", help="BendingMagnet example")
    bm.set_defaults(_func=run_bending_magnet)

    fixed = subparsers.add_parser(
        "fixed", help="FixedSpectrumSource example with Gaussian intensity profile"
    )
    fixed.set_defaults(_func=run_fixed)

    args = parser.parse_args()
    args._func()


if __name__ == "__main__":
    main()
