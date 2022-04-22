"""Example utility functions."""

import os
import argparse
import matplotlib.pyplot as plt
from syris.materials import make_fromfile


def get_default_parser(description):
    """Default argument parser with *description*"""
    return argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


def show(image, title=""):
    """Show *image* with *title* on its own figure."""
    plt.figure()
    image = plt.imshow(image)
    plt.title(title)
    plt.colorbar()
    plt.show(block=False)

    return image


def get_material(name):
    """Load material from file *name*."""
    return make_fromfile(os.path.join(os.path.dirname(__file__), "data", name))
