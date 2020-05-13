import numpy as np
import quantities as q


def is_close(a, b, eps):
    return abs(a - b) < eps


def filter_close(array):
    """Filter close points."""
    result = []

    for i in range(len(array) - 1):
        if not is_close(array[i], array[i + 1], np.finfo(np.float32).eps):
            result.append(array[i])
        else:
            result.append(np.nan)
    result.append(array[-1])

    return np.sort(result)


def np_roots(coeffs, interval):
    roots = np.concatenate((np.roots(coeffs), [np.nan]))

    return filter_np_roots(roots, interval)


def filter_np_roots(roots, interval):
    result = []
    for root in roots:
        if root.real > interval[0] and root.real <= interval[1] and root.imag == 0:
            result.append(root.real)
        else:
            result.append(np.nan)

    return np.sort(result)


def derivative(coeffs):
    res = np.empty(len(coeffs) - 1)

    for i in range(len(coeffs))[:-1]:
        exponent = len(coeffs) - i - 1
        res[i] = exponent * coeffs[i]

    return res


def f(coeffs, x):
    return np.sum(coeffs * np.array([x ** (len(coeffs) - i - 1) for i in range(len(coeffs))]))


def sgn(val):
    return 0 if val == 0 else 1 if val > 0 else -1


def get_linear_points(direction, start=(0, 0, 0), num=4):
    res = []
    for i in range(num):
        point = np.copy(start)
        point[direction] += i
        res.append(point)

    return np.array(res) * q.mm


class Metaball(object):
    def __init__(self, r, R, c):
        self.r = r
        self.R = R
        self.c = c
        if r != 0:
            self.coeff = 1.0 / (R ** 2 - r ** 2) ** 2
        else:
            self.coeff = 1
        self.lower = self.c - self.R
        self.upper = self.c + self.R
        self.center = (self.lower + self.upper) / 2

    def get_metaball(self, x):
        res = (self.R ** 2 - (x - self.c) ** 2) ** 2 * self.coeff

        res[np.where(np.abs(x - self.c) > self.R)] = 0

        return res

    def get_coeffs(self, final=False):
        abs_val = 1 if final else 0

        return np.array(
            [
                self.coeff,
                -4 * self.coeff * self.c,
                self.coeff * (-2 * self.R ** 2 + 6 * self.c ** 2),
                self.coeff * (4 * self.R ** 2 * self.c - 4 * self.c ** 3),
                self.coeff * (self.R ** 4 - 2 * self.R ** 2 * self.c ** 2 + self.c ** 4) - abs_val,
            ]
        )
