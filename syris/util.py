"""Utility functions."""
import numpy as np
import scipy.misc


TIF_32 = True


try:
    import tifffile
except:
    TIF_32 = False


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
        if not hasattr(value, 'magnitude') and hasattr(value[0], 'magnitude'):
            # Convert tuple of quantities to quantity of a tuple
            unit = value[0].units
            value = [item.rescale(unit).magnitude for item in value] * unit

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


def save_image(filename, image):
    """Save *image* to *filename*."""
    if (filename.endswith('tif') or filename.endswith('.tiff')) and TIF_32:
        tifffile.imsave(filename, image.astype(np.float32))
    else:
        scipy.misc.imsave(filename, image)


def read_image(filename):
    """Load image from *filename*."""
    if (filename.endswith('tif') or filename.endswith('.tiff')) and TIF_32:
        image = tifffile.imread(filename)
    else:
        image = scipy.misc.imread(filename, image)

    return image


def get_gauss(x, center, sigma, normalized=False):
    """Get 1D gaussian function over *x* centered around *center* and std *sigma*. If *normalized*
    is True, the sum of the function is 1.
    """
    y = np.exp(-(x - float(center)) ** 2 / (2 * sigma ** 2))
    if normalized:
        # Don't use 1/(2 sigma^2) because if the peak is too broad the sum won't be 1
        y /= np.sum(y)

    return y
