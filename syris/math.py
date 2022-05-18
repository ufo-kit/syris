"""Math helper functions."""

import numpy as np
from scipy import interpolate as interp


def difference_root(x_0, tck, y_d):
    """
    Given a function :math:`f(x) = y`, find :math:`x_1` for which holds
    :math:`|f(x_1) - f(x_0)| = y_d`. *x_0* is the starting :math:`x_0`
    and :math:`f(x)` is defined by spline coefficients *tck*.
    """

    def get_root(up=1):
        y_s = interp.splev(x_0, tck) + up * y_d
        t, c, k = tck
        # Adjust spline coefficients to be able to find f(x) = 0.
        c = c - y_s

        return closest(interp.sproot((t, c, k)), x_0)

    # Function can be ascending or descending and since we are interested
    # in the difference, x_1 can be greater or less than x_0, so check
    # both directions.
    top = get_root()
    bottom = get_root(-1)

    return top if top is not None and top < bottom or bottom is None else bottom


def closest(values, min_value):
    """Get the minimum greater value greater than *min_value* from *values*."""
    bigger = np.where(values > min_value)[0]
    if len(bigger) == 0:
        return None
    else:
        return values[bigger[0]]


def get_surrounding_points(points, threshold):
    """
    Get the closest points around a *threshold* from both sides, left
    and right. If one of the sides is empty than None is returned
    on its place.
    """
    left_points = points[np.where(points < threshold)]
    right_points = points[np.where(points > threshold)]

    left = None if len(left_points) == 0 else np.max(left_points)
    right = None if len(right_points) == 0 else np.min(right_points)

    return left, right


def match_range(x_points, y_points, x_target):
    """Match the curve :math:`f(x) = y` to *x_target* points by interpolation
    of *x_points* and *y_points*.
    """
    x_points = x_points.simplified
    y_points = y_points.simplified
    tck = interp.splrep(x_points, y_points)

    return interp.splev(x_target.rescale(x_points.units), tck) * y_points.units


def supremum(x_0, data):
    """Return the smallest point from *data* which is greater than *x_0*."""
    srt = np.copy(data)
    srt.sort()
    greater_indices = np.where(srt - x_0 > 0)[0]

    if len(greater_indices) == 0:
        return None
    return srt[min(greater_indices)]


def infimum(x_0, data):
    """Return the greatest point from *data* which is less than *x_0*."""
    srt = np.copy(data)
    srt.sort()
    smaller_indices = np.where(srt - x_0 < 0)[0]

    if len(smaller_indices) == 0:
        return None
    return srt[max(smaller_indices)]


def sigma_to_fwnm(sigma, n=2):
    """Return Gaussian full width at n-th maximum given by *sigma* and *n*."""
    return sigma * 2 * np.sqrt(2 * np.log(n))


def fwnm_to_sigma(fwnm, n=2):
    """Return Gaussian sigma from full width at *n*-th maximum *fwnm*."""
    return fwnm / (2 * np.sqrt(2 * np.log(n)))


def fftfreq(n, pixel_size):
    """Compute spatial frequencies for a 2D grid (*n*, *n*) with spacing *pixel_size*. Returns
    spatial frequencies f as (f_y, f_x).
    """
    frequencies = np.fft.fftfreq(n)

    return np.meshgrid(frequencies, frequencies) / pixel_size
