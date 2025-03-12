# Copyright (C) 2013-2025 Karlsruhe Institute of Technology
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

"""
Show refraction effect by propagating an object behind a wedge.
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import quantities as q
import syris
from numpy.fft import fft2, ifft2
from syris.bodies.simple import make_sphere, StaticBody
from syris.materials import Material
from syris.physics import propagate
from .util import get_default_parser, get_material, show


def main():
    args = parse_args()
    syris.init()
    # Propagate to 50 cm
    d = 0.5 * q.m
    n = 4096
    shape = (n, n)
    material = get_material("pmma_5_30_kev.mat")
    # Pure phase object
    material_phase = Material(
        "pmma_phase",
        material.refractive_indices.real + 0j,
        material.energies
    )
    energy = 29 * q.keV
    ps = 0.2 * q.um  # Pixel size
    shift = 16 * ps  # Shift by 24 pixels

    # Wedge gradient multiplied by delta defines the diffraction angle:
    # tan(angle) = d/dx delta T(x) = shift / d
    delta = material_phase.get_refractive_index(energy).real
    dT = (shift * ps / delta / d).simplified.magnitude
    angle = delta * dT / ps.simplified.magnitude
    print(f"Wedge diffraction angle: {np.arctan(angle) * 1e6} urad")

    wedge = np.tile(np.arange(n // 2) * dT, [n // 2, 1])
    wedge = np.pad(wedge, n // 4)
    wedge = StaticBody(wedge * q.m, ps, material=material_phase)
    u_wedge = wedge.transfer(shape, ps, energy).get()

    sample = make_sphere(n, n // 8 * ps, pixel_size=ps, material=material)
    im_sample = propagate([sample], shape, [energy], d, ps).get()
    im = propagate([sample, wedge], shape, [energy], d, ps).get()

    c = ifft2(np.conj(fft2(im - 1)) * fft2(im_sample - 1)).real
    dy, dx = np.unravel_index(c.argmax(), c.shape)
    print(f"Desired shift: dx: {(shift / ps).simplified.magnitude} pixels, dy: 0 pixels")
    print(f"Computed shift: dx: {dx} pixels, dy: {dy} pixels")

    if args.output_directory:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory, exist_ok=True)
        imageio.imwrite(os.path.join(args.output_directory, "image.tif"), im_sample)
        imageio.imwrite(os.path.join(args.output_directory, "image-wedge.tif"), im)

    show(im, title="Sphere and wedge", vmin=im_sample.min(), vmax=im_sample.max())
    show(im_sample, title="Only sphere")
    show(u_wedge.real, title="Real part of wedge transmission function")
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Output directory for the two X-ray projections image.tif and image-wedge.tif"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
