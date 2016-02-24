"""Example utility functions."""

import matplotlib.pyplot as plt


def show(image, title=''):
    """Show *image* with *title* on its own figure."""
    plt.figure()
    plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.show(False)
