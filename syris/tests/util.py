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
