"""Utility functions."""

TIF_32 = True

import numpy as np
import scipy.misc
try:
    import tifffile
except:
    TIF_32 = False


def save_image(filename, image):
    """Save *image* to *filename*."""
    if (filename.endswith('tif') or filename.endswith('.tiff')) and TIF_32:
        tifffile.imsave(filename, image.astype(np.float32))
    else:
        scipy.misc.imsave(filename, image)
