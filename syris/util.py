"""Utility functions."""
import numpy as np


def make_tuple(value, num_dims=2):
    """Make a tuple from *value* in case it is a scalar, otherwise leave it as it was."""
    # Don't test for class because quantities would throw it off
    try:
        value[0]
    except:
        if hasattr(value, 'magnitude'):
            value = (value.magnitude,) * num_dims * value.units
        else:
            value = (value,) * num_dims
    else:
        if len(value) != num_dims:
            raise ValueError("Value is a tuple already and with different dimensions")

    return value


def get_magnitude(value):
    """If value is a quantity, simplify it to base units and return the magnitude, otherwise do
    nothing.
    """
    if hasattr(value, 'magnitude'):
        value = value.simplified.magnitude

    return value


def next_power_of_two(n):
    """Get next power of two for number *n*."""
    return 2 ** int(np.ceil(np.log2(n)))
