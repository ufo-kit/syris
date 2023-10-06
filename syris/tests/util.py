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

import numpy as np
from syris.util import get_magnitude, make_tuple


def get_gauss_2d(shape, sigma, pixel_size=None, fourier=False):
    shape = make_tuple(shape)
    sigma = get_magnitude(make_tuple(sigma))
    if pixel_size is None:
        pixel_size = (1, 1)
    else:
        pixel_size = get_magnitude(make_tuple(pixel_size))

    if fourier:
        i = np.fft.fftfreq(shape[1]) / pixel_size[1]
        j = np.fft.fftfreq(shape[0]) / pixel_size[0]
        i, j = np.meshgrid(i, j)

        return np.exp(-2 * np.pi ** 2 * ((i * sigma[1]) ** 2 + (j * sigma[0]) ** 2))
    else:
        x = (np.arange(shape[1]) - shape[1] // 2) * pixel_size[1]
        y = (np.arange(shape[0]) - shape[0] // 2) * pixel_size[0]
        x, y = np.meshgrid(x, y)
        gauss = np.exp(-(x ** 2) / (2.0 * sigma[1] ** 2) - y ** 2 / (2.0 * sigma[0] ** 2))

        return np.fft.ifftshift(gauss)
