"""Utility functions for untis tests."""
import numpy as np


def compare_array(array, delta):
    return not np.any(array > delta)


def compare_arrays(a_1, a_2, delta):
    diff = a_1 - a_2

    return compare_array(diff, delta)
