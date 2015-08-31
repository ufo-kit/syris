"""Utility functions."""


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
