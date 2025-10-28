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
Visualization of the Angular spectrum method.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft
from .util import show


def main():
    # All quantities are multiples of pixel size
    lam = 32
    k = 2 * np.pi / lam
    # Spatial wavelength stemming from the angle of the plane wave normal and z axis
    lam_s = 64
    ks = 2 * np.pi / lam_s
    n = 256
    d = 4  # Propagation distance
    x = np.arange(n)
    x, z = np.meshgrid(x, x)

    # x direction cosine
    kx = lam / lam_s
    # z direction cosine
    kz = np.sqrt(1 - kx ** 2)
    # 3D plane wave at y=0, so actually 2D computation
    u = np.exp(1j * k * (kx * x + kz * z))
    show(u.real, title="xz cut through a 3D plane wave")

    # Full Kirchhof-Fresnel propagator for frequency 1 / lam_s
    prop = np.exp(1j * kz * d * k)
    # Fourier transform of our plane wave at z=0 (2D plane wave projection)
    up = fft(u[0])
    # Multiply the corresponding frequency by the propagator (all other frequencies are 0 because
    # input is a plane wave at an angle, so we only need to multiply one point at index n / lam_s)
    up[int(n / lam_s)] *= prop
    # Inverse Fourier transform gives the plane wave at z=4
    up = ifft(up)

    print("Phase shift of the 3D plane wave at z=4:", k * d * kz)
    # Phase shift of the 2D "projected" plane wave with wavenumber ks = 2 Pi / lam_s.
    # Phase shift is ks * d * tan(alpha) = ks * d * sin(alpha) / kx = ks * d * Sqrt(1 - kx^2) / kx
    print("Phase shift of the 2D plane wave at z=4:", ks * d * np.sqrt(1 - kx ** 2) / kx)

    plt.figure()
    plt.plot(ks * x[0], u[0].real, label="Plane wave at z=0")
    plt.plot(ks * x[0], u[d].real, label="Plane wave at z=4")
    plt.plot(ks * x[0], up.real, label="Plane wave at z=0 propagated to z=4")
    plt.plot(ks * x[0], (up[0].real,) * n, color="gray", label="Visual guide for phase shift")
    plt.ylim(-1.25, 1.5)
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
