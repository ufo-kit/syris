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
Show magnification due to spherical incident wave with three approaches:
    1. Spherical incident wave
    2. Parabolic incident wave
    3. Plane incident wave and rescaled pixel size and propagation distance
"""

import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import syris
from numpy.fft import fft2, ifft2, fftfreq
from syris.bodies.simple import make_sphere
from syris.physics import energy_to_wavelength
from .util import get_default_parser


def crop(image):
    n = image.shape[0]

    return image[n // 4:3 * n // 4, n // 4: 3 * n // 4]


def main():
    _ = parse_args()
    syris.init()
    n = 4096  # number of pixels
    ps = 0.2e-6  # pixel size in meters
    M = 1.25  # magnification
    d = 1  # propagation distance in meters
    energy = 15 * q.keV
    lam = energy_to_wavelength(energy).simplified.magnitude
    beta = 1e-9
    delta = 1e-7
    k = 2 * np.pi / lam
    l = d / (M - 1)     # source - sample distance in meters
    x = np.arange(-n // 2, n // 2) * ps
    x, y = np.meshgrid(x, x)
    r = np.sqrt(x ** 2 + y ** 2 + l ** 2)
    f = fftfreq(n) / ps
    f, g = np.meshgrid(f, f)
    fg = np.sqrt(f ** 2 + g ** 2)
    print(f"source - sample: {l} m, sample - detector: {d} m")

    # Spherical incident wave which magnifies by M -> Kirchhof-Fresnel diffraction
    # Incident wave field with unit amplitude and spherical phase profile
    u_inc = np.exp(1j * k * r)
    # Sphere with n / 8 pixels radius
    proj = make_sphere(n, n // 8 * ps * q.m, pixel_size=ps * q.m).project((n, n) , ps * q.m).get()
    # Tranmission function
    u_obj = np.exp(-k * proj * (beta + 1j * delta))
    # Wavefield immediatly after the object
    u = u_inc * u_obj

    # Propagator
    prop = np.exp(1j * k * (d * np.sqrt(1 - (lam * f) ** 2 - (lam * g) ** 2)))
    # Butterworth filter suppresses high frequencies and makes the result a little nicer to look at
    butt = 1 / (1 + (ps * fg / 0.25) ** (2 * 10))
    # Convolution and conversion to intensity
    im_sphere = crop(np.abs(ifft2(fft2(u) * prop * butt)) ** 2)

    # Parabolic incident wave and Fresnel diffraction
    # r ~ l + x^2 / (2 * l) -> keeps only second order wrt x, same for the propagator
    # This is an unusual form with the prefactor in so that you can compare u_inc and u_inc_parabola
    # directly
    u_inc_parabola = np.exp(1j * k * (l + (x ** 2 + y ** 2) / (2 * l)))
    u = u_inc_parabola * u_obj

    # This is an unusual form with the prefactor in so that you can compare prop and prop_fresnel
    # directly
    prop_fresnel = np.exp(1j * k * d * (1 - (lam * f) ** 2 / 2 - (lam * g) ** 2 / 2))
    im_parabola = crop(np.abs(ifft2(fft2(u) * prop_fresnel * butt)) ** 2)

    # Plane wave and no rescaling -> no magnification
    im_noscale = crop(np.abs(ifft2(fft2(u_obj) * prop_fresnel * butt)) ** 2)

    # Plane incident wave, magnification achieved by ps = ps / M and d = d / M
    ps /= M
    d /= M
    f *= M
    g *= M
    proj = make_sphere(
        n,
        M * n // 8 * ps * q.m,
        pixel_size=ps * q.m
    ).project((n, n) , ps * q.m).get()
    u_obj = np.exp(-k * proj * (beta + 1j * delta))
    prop_fresnel_plane = np.exp(1j * k * d * (1 - (lam * f) ** 2 / 2 - (lam * g) ** 2 / 2))
    im_plane = crop(np.abs(ifft2(fft2(u_obj) * prop_fresnel_plane * butt)) ** 2 / M ** 2)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0, 0].imshow(im_noscale)
    ax[0, 0].set_title("Plane incident wave")

    ax[0, 1].imshow(im_plane)
    ax[0, 1].set_title("Plane incident wave + rescaling")

    ax[1, 0].imshow(im_sphere)
    ax[1, 0].set_title("Spherical incident wave")

    ax[1, 1].imshow(im_parabola)
    ax[1, 1].set_title("Parabolic incident wave")
    plt.tight_layout()

    plt.figure()
    plt.plot(im_sphere[n // 4], label="Spherical incident wave")
    plt.plot(im_parabola[n // 4], label="Parabolic indicent wave")
    plt.plot(im_plane[n // 4], label="Plane incident wave + rescaling")
    plt.plot(im_noscale[n // 4], label="Plane incident wave")
    plt.grid()
    plt.legend()
    plt.show()


def parse_args():
    parser = get_default_parser(__doc__)

    return parser.parse_args()


if __name__ == "__main__":
    main()
