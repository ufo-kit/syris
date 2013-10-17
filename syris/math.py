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
        t, c, k = np.copy(tck)
        # Adjust spline coefficients to be able to find f(x) = 0.
        c -= y_s

        return closest(interp.sproot((t, c, k)), x_0)

    # Function can be ascending or descending and since we are interested
    # in the difference, x_1 can be greater or less than x_0, so check
    # both directions.
    top = get_root()
    bottom = get_root(-1)

    return top if top is not None and top < bottom or bottom \
        is None else bottom


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

    return interp.splev(x_target.rescale(x_points.units), tck) * \
        y_points.units
