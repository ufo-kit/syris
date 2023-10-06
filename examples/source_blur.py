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

"""Source blur example."""
import matplotlib.pyplot as plt
import quantities as q
import syris
import syris.imageprocessing as ip
from syris.geometry import Trajectory
from syris.bodies.simple import make_sphere
from syris.devices.sources import make_topotomo
from syris.physics import propagate
from .util import get_default_parser, get_material, show


def main():
    args = parse_args()
    syris.init()
    n = 1024
    d = args.propagation_distance * q.m
    shape = (n, n)
    ps = 1 * q.um
    energy = 20 * q.keV
    tr = Trajectory([(n / 2, n / 2, 0)] * ps, pixel_size=ps)
    sample = make_sphere(n, n / 30 * ps, pixel_size=ps, material=get_material("air_5_30_kev.mat"))

    bm = make_topotomo(pixel_size=ps, trajectory=tr)
    print("Source size FWHM (height x width): {}".format(bm.size.rescale(q.um)))

    intensity = propagate([sample], shape, [energy], d, ps).get()
    incoh = bm.apply_blur(intensity, d, ps).get()

    region = (n // 4, n // 4, n // 2, n // 2)
    intensity = ip.crop(intensity, region).get()
    incoh = ip.crop(incoh, region).get()

    show(intensity, title="Coherent")
    show(incoh, title="Applied source blur")
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument(
        "--propagation-distance", type=float, default=2, help="Propagation distance [m]"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
