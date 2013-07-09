"""Math helper functions."""
from scipy import interpolate


def match_range(x_points, y_points, x_target):
    """Match the curve :math:`f(x) = y` to *x_target* points by interpolation
    of *x_points* and *y_points*.
    """
    x_points = x_points.simplified
    y_points = y_points.simplified
    tck = interpolate.splrep(x_points, y_points)

    return interpolate.splev(x_target.rescale(x_points.units), tck) * \
        y_points.units
